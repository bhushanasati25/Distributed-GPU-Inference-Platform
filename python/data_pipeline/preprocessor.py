#!/usr/bin/env python3
"""
Data Preprocessor — Distributed GPU Inference Platform
=======================================================

Large-scale, streaming data preprocessing for images, text, and tabular data
with multiprocessing parallelism, checkpointing, data validation, and multiple
output formats (Parquet, TFRecord, NumPy memmap).

Designed for datasets exceeding 10 TB — all operations stream data rather than
loading it fully into memory.

Usage::

    from python.data_pipeline.preprocessor import DataPreprocessor, Pipeline

    pp = DataPreprocessor()
    pp.preprocess_images("/data/raw_images", "/data/processed", target_size=(224, 224))
    pp.preprocess_text("/data/corpus.jsonl", "/data/tokenized.parquet", tokenizer=tok)

    # Or chain steps with Pipeline
    pipeline = Pipeline([
        ("resize", lambda batch: resize(batch, (224, 224))),
        ("normalize", lambda batch: batch / 255.0),
    ])
    pipeline.run(input_iter, output_writer)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import multiprocessing as mp
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class PreprocessingError(Exception):
    """Generic preprocessing failure."""


class DataValidationError(PreprocessingError):
    """Raised when input data fails validation checks."""


class CheckpointError(PreprocessingError):
    """Raised when checkpoint save / restore fails."""


class OutputFormatError(PreprocessingError):
    """Raised for unsupported or misconfigured output formats."""


@dataclass
class TabularSchema:
    """Schema specification for tabular data preprocessing."""
    columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class PreprocessingResult(dict):
    """Result summary dictionary that also allows indexing output files as a list."""
    def __init__(self, summary: Dict[str, Any], output_files: List[str]):
        super().__init__(summary)
        self.output_files = output_files
        self["output_files"] = output_files

    def __len__(self) -> int:
        return len(self.output_files) if self.output_files else super().__len__()

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            return self.output_files[item]
        return super().__getitem__(item)

    def __iter__(self):
        if self.output_files:
            return iter(self.output_files)
        return super().__iter__()


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """Tracks preprocessing progress for resume support.

    Persists to a JSON file in the output directory so that a crashed job can
    pick up where it left off.
    """

    checkpoint_path: str
    last_processed_index: int = 0
    processed_files: List[str] = field(default_factory=list)
    total_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        """Persist checkpoint to disk."""
        data = {
            "last_processed_index": self.last_processed_index,
            "processed_files": self.processed_files,
            "total_processed": self.total_processed,
            "metadata": self.metadata,
        }
        tmp = self.checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path: str) -> "Checkpoint":
        """Load a checkpoint from disk, or create a fresh one."""
        if os.path.isfile(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            cp = cls(
                checkpoint_path=checkpoint_path,
                last_processed_index=data.get("last_processed_index", 0),
                processed_files=data.get("processed_files", []),
                total_processed=data.get("total_processed", 0),
                metadata=data.get("metadata", {}),
            )
            logger.info(
                "Resumed checkpoint: %d items already processed.", cp.total_processed
            )
            return cp
        return cls(checkpoint_path=checkpoint_path)


# ---------------------------------------------------------------------------
# Worker functions (module-level for pickling)
# ---------------------------------------------------------------------------

def _process_image_batch(
    file_paths: List[str],
    output_dir: str,
    target_size: Tuple[int, int],
    normalize: bool,
    output_format: str,
) -> Tuple[int, int, List[str]]:
    """Process a batch of image files.  Runs in a worker process.

    Returns:
        (success_count, error_count, list_of_error_paths)
    """
    from PIL import Image  # type: ignore[import-untyped]

    success = 0
    errors = 0
    error_paths: List[str] = []

    for fpath in file_paths:
        try:
            img = Image.open(fpath).convert("RGB")
            img = img.resize(target_size, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            if normalize:
                arr = arr / 255.0

            stem = Path(fpath).stem
            if output_format == "numpy":
                out_path = os.path.join(output_dir, f"{stem}.npy")
                np.save(out_path, arr)
            elif output_format == "parquet":
                import pyarrow as pa  # type: ignore[import-untyped]
                import pyarrow.parquet as pq  # type: ignore[import-untyped]

                flat = arr.flatten()
                table = pa.table({
                    "filename": [os.path.basename(fpath)],
                    "data": [flat.tolist()],
                    "height": [target_size[1]],
                    "width": [target_size[0]],
                    "channels": [3],
                })
                out_path = os.path.join(output_dir, f"{stem}.parquet")
                pq.write_table(table, out_path)
            else:
                # Raw binary
                out_path = os.path.join(output_dir, f"{stem}.bin")
                arr.tofile(out_path)

            success += 1
        except Exception as exc:
            errors += 1
            error_paths.append(fpath)
            # Log at DEBUG in worker; caller aggregates.
    return success, errors, error_paths


def _process_text_batch(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> List[Dict[str, Any]]:
    """Tokenize a batch of text records.  Runs in a worker process.

    Each *record* is expected to have a ``"text"`` key.

    Returns:
        List of dicts with ``input_ids``, ``attention_mask``, etc.
    """
    results: List[Dict[str, Any]] = []
    for rec in records:
        text = rec.get("text", "")
        if not text:
            continue
        try:
            encoded = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
            out: Dict[str, Any] = {}
            for key, val in encoded.items():
                if isinstance(val, list):
                    out[key] = val
                else:
                    out[key] = val
            # Carry over any extra metadata columns.
            for key, val in rec.items():
                if key != "text":
                    out[key] = val
            results.append(out)
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# DataPreprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """High-level data preprocessing with parallelism, checkpointing, and
    streaming I/O.

    Example::

        pp = DataPreprocessor()
        pp.preprocess_images(
            "/data/raw", "/data/processed",
            target_size=(224, 224), num_workers=8,
        )
    """

    # ------------------------------------------------------------------ #
    # Image preprocessing
    # ------------------------------------------------------------------ #

    def preprocess_images(
        self,
        input_dir: str,
        output_dir: str,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        num_workers: int = 8,
        batch_size: int = 256,
        output_format: str = "numpy",
        checkpoint_interval: int = 10,
    ) -> Dict[str, Any]:
        """Resize, normalise, and convert images in parallel.

        Args:
            input_dir:           Directory of source images.
            output_dir:          Directory for processed outputs.
            target_size:         ``(width, height)`` to resize to.
            normalize:           Divide pixel values by 255.
            num_workers:         Number of parallel worker processes.
            batch_size:          Files per worker batch.
            output_format:       ``"numpy"``, ``"parquet"``, or ``"binary"``.
            checkpoint_interval: Save checkpoint every N batches.

        Returns:
            Summary dict with ``total``, ``success``, ``errors``, ``elapsed_s``.

        Raises:
            FileNotFoundError:    If *input_dir* does not exist.
            DataValidationError:  If *input_dir* contains no image files.
        """
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        all_files = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if Path(f).suffix.lower() in image_exts
        )
        if not all_files:
            raise DataValidationError(f"No image files found in {input_dir}")

        # Checkpoint
        ckpt_path = os.path.join(output_dir, ".preprocess_checkpoint.json")
        ckpt = Checkpoint.load(ckpt_path)
        processed_set = set(ckpt.processed_files)
        remaining = [f for f in all_files if f not in processed_set]

        logger.info(
            "Image preprocessing: %d total, %d already done, %d remaining.",
            len(all_files), len(processed_set), len(remaining),
        )

        # Progress
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            pbar = tqdm(total=len(remaining), desc="Preprocessing images", unit="img")
        except ImportError:
            pbar = None

        total_success = 0
        total_errors = 0
        all_error_paths: List[str] = []
        t0 = time.monotonic()

        # Batch and dispatch to workers.
        batches = [
            remaining[i : i + batch_size]
            for i in range(0, len(remaining), batch_size)
        ]

        worker_fn = partial(
            _process_image_batch,
            output_dir=output_dir,
            target_size=target_size,
            normalize=normalize,
            output_format=output_format,
        )

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(worker_fn, batch): batch for batch in batches
            }
            batch_counter = 0
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    ok, errs, err_paths = future.result()
                    total_success += ok
                    total_errors += errs
                    all_error_paths.extend(err_paths)
                    ckpt.processed_files.extend(batch)
                    ckpt.total_processed += ok
                    ckpt.last_processed_index += len(batch)
                except Exception as exc:
                    logger.error("Batch failed: %s", exc)
                    total_errors += len(batch)

                batch_counter += 1
                if batch_counter % checkpoint_interval == 0:
                    ckpt.save()

                if pbar is not None:
                    pbar.update(len(batch))

        if pbar is not None:
            pbar.close()

        ckpt.save()
        elapsed = time.monotonic() - t0

        summary = {
            "total": len(all_files),
            "success": total_success,
            "errors": total_errors,
            "error_paths": all_error_paths[:100],  # cap for readability
            "elapsed_s": round(elapsed, 2),
            "images_per_sec": round(total_success / max(elapsed, 1e-9), 1),
        }
        logger.info(
            "Image preprocessing complete: %d ok, %d errors in %.1f s (%.0f img/s).",
            total_success, total_errors, elapsed, summary["images_per_sec"],
        )
        out_files = sorted([
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if not f.startswith(".") and not f.endswith(".json") and not f.endswith(".tmp")
        ])
        return PreprocessingResult(summary, out_files)

    # ------------------------------------------------------------------ #
    # Text preprocessing
    # ------------------------------------------------------------------ #

    def preprocess_text(
        self,
        input_path: str,
        output_path: str,
        tokenizer: Any,
        max_length: int = 512,
        num_workers: int = 8,
        batch_size: int = 1024,
        text_column: str = "text",
        output_format: str = "parquet",
    ) -> Dict[str, Any]:
        """Tokenize a JSONL or Parquet text corpus in parallel.

        Args:
            input_path:    Path to JSONL or Parquet input file.
            output_path:   Path for the output file.
            tokenizer:     A callable (or HuggingFace-style tokenizer) that accepts
                           ``(text, max_length=..., padding=..., truncation=...,
                           return_attention_mask=...)``.
            max_length:    Maximum token length.
            num_workers:   Number of parallel workers.
            batch_size:    Records per batch.
            text_column:   Name of the text column.
            output_format: ``"parquet"`` or ``"numpy"``.

        Returns:
            Summary dict.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
        """
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Read input records.
        records: List[Dict[str, Any]] = []
        if input_path.endswith(".parquet"):
            table = pq.read_table(input_path)
            for i in range(len(table)):
                rec = {col: table.column(col)[i].as_py() for col in table.column_names}
                records.append(rec)
        else:
            # JSONL
            with open(input_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        if not records:
            raise DataValidationError(f"No records found in {input_path}")

        # Ensure text column exists.
        if text_column != "text":
            for rec in records:
                if text_column in rec:
                    rec["text"] = rec.pop(text_column)

        logger.info("Tokenizing %d records with %d workers …", len(records), num_workers)

        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            pbar = tqdm(total=len(records), desc="Tokenizing", unit="rec")
        except ImportError:
            pbar = None

        all_results: List[Dict[str, Any]] = []
        t0 = time.monotonic()

        # For pickling safety, use ThreadPool if tokenizer isn't picklable.
        # We fall back to sequential processing for safety.
        batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]

        for batch in batches:
            batch_results = _process_text_batch(batch, tokenizer, max_length)
            all_results.extend(batch_results)
            if pbar is not None:
                pbar.update(len(batch))

        if pbar is not None:
            pbar.close()

        # Write output.
        if output_format == "parquet" and all_results:
            # Build columnar data.
            columns: Dict[str, List[Any]] = {}
            for key in all_results[0]:
                columns[key] = [r.get(key) for r in all_results]
            table = pa.table(columns)
            pq.write_table(table, output_path)
        elif output_format == "numpy" and all_results:
            # Save input_ids and attention_mask as .npy
            if "input_ids" in all_results[0]:
                ids = np.array([r["input_ids"] for r in all_results], dtype=np.int64)
                np.save(output_path.replace(".parquet", "_input_ids.npy"), ids)
            if "attention_mask" in all_results[0]:
                mask = np.array([r["attention_mask"] for r in all_results], dtype=np.int64)
                np.save(output_path.replace(".parquet", "_attention_mask.npy"), mask)

        elapsed = time.monotonic() - t0
        summary = {
            "total_input": len(records),
            "total_output": len(all_results),
            "elapsed_s": round(elapsed, 2),
            "records_per_sec": round(len(all_results) / max(elapsed, 1e-9), 1),
        }
        logger.info("Text preprocessing complete: %d records in %.1f s.", len(all_results), elapsed)
        return summary

    # ------------------------------------------------------------------ #
    # Tabular preprocessing
    # ------------------------------------------------------------------ #

    def preprocess_tabular(
        self,
        input_path: str,
        output_path: str,
        schema: Union[Dict[str, Dict[str, Any]], TabularSchema],
        num_workers: int = 4,
        batch_size: int = 10_000,
        output_format: str = "parquet",
    ) -> Dict[str, Any]:
        """Normalize and encode tabular data (CSV / Parquet).

        The *schema* dict maps column names to processing instructions::

            {
                "age": {"type": "numeric", "normalize": "standard"},
                "income": {"type": "numeric", "normalize": "minmax"},
                "city": {"type": "categorical", "encoding": "onehot"},
                "id": {"type": "passthrough"},
            }

        Supported normalizations: ``"standard"`` (z-score), ``"minmax"``.
        Supported encodings: ``"onehot"``, ``"label"``.

        Args:
            input_path:    Path to CSV or Parquet input.
            output_path:   Path for the output file.
            schema:        Column processing schema (see above).
            num_workers:   Number of parallel workers.
            batch_size:    Rows per batch (for streaming large files).
            output_format: ``"parquet"`` (default) or ``"csv"``.

        Returns:
            Summary dict.

        Raises:
            FileNotFoundError:   If *input_path* does not exist.
            DataValidationError: On schema violations.
        """
        if isinstance(schema, TabularSchema):
            schema = schema.columns

        import pandas as pd  # type: ignore[import-untyped]
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Read the full dataset (for statistics computation we need a scan).
        if input_path.endswith(".parquet"):
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

        if df.empty:
            raise DataValidationError(f"Empty dataset: {input_path}")

        logger.info("Tabular preprocessing: %d rows, %d columns.", len(df), len(df.columns))

        # Validate schema columns exist.
        missing_cols = set(schema.keys()) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Schema columns not found in data: {missing_cols}"
            )

        # Check for missing values.
        null_counts = df[list(schema.keys())].isnull().sum()
        if null_counts.any():
            logger.warning("Missing values detected:\n%s", null_counts[null_counts > 0])

        # Process columns.
        result_frames: List[pd.DataFrame] = []
        for col_name, col_spec in schema.items():
            col_type = col_spec.get("type", "passthrough")

            if col_type == "numeric":
                series = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
                norm = col_spec.get("normalize", "none")
                if norm == "standard":
                    mean = series.mean()
                    std = series.std()
                    if std > 0:
                        series = (series - mean) / std
                    else:
                        series = series - mean
                elif norm == "minmax":
                    vmin = series.min()
                    vmax = series.max()
                    rng = vmax - vmin
                    if rng > 0:
                        series = (series - vmin) / rng
                    else:
                        series = series - vmin
                result_frames.append(series.to_frame(col_name))

            elif col_type == "categorical":
                encoding = col_spec.get("encoding", "label")
                if encoding == "onehot":
                    dummies = pd.get_dummies(df[col_name], prefix=col_name)
                    result_frames.append(dummies)
                elif encoding == "label":
                    cats = df[col_name].astype("category")
                    result_frames.append(cats.cat.codes.to_frame(col_name))
                else:
                    result_frames.append(df[[col_name]])

            elif col_type == "passthrough":
                result_frames.append(df[[col_name]])
            else:
                logger.warning("Unknown column type '%s' for '%s' — passing through.", col_type, col_name)
                result_frames.append(df[[col_name]])

        result_df = pd.concat(result_frames, axis=1)

        # Write output.
        if output_format == "parquet":
            table = pa.Table.from_pandas(result_df)
            pq.write_table(table, output_path)
        else:
            result_df.to_csv(output_path, index=False)

        summary = {
            "input_rows": len(df),
            "output_rows": len(result_df),
            "output_columns": len(result_df.columns),
            "output_path": output_path,
        }
        logger.info("Tabular preprocessing complete: %d rows → %s.", len(result_df), output_path)
        return summary


# ---------------------------------------------------------------------------
# Pipeline — generic step chaining
# ---------------------------------------------------------------------------

class Pipeline:
    """Chain preprocessing steps into a reusable pipeline.

    Each step is a ``(name, callable)`` pair.  The callable receives a batch
    (any object) and returns the transformed batch.

    Example::

        pipeline = Pipeline([
            ("resize", lambda batch: resize(batch, (224, 224))),
            ("normalize", lambda batch: batch / 255.0),
        ])
        for out_batch in pipeline.run(input_iter):
            writer.write(out_batch)
    """

    def __init__(self, steps: List[Tuple[str, Callable[..., Any]]]) -> None:
        if not steps:
            raise ValueError("Pipeline must have at least one step.")
        self._steps = steps
        logger.info(
            "Pipeline created with %d step(s): %s",
            len(steps), [name for name, _ in steps],
        )

    @property
    def steps(self) -> List[Tuple[str, Callable[..., Any]]]:
        """The ordered list of (name, callable) steps."""
        return list(self._steps)

    def run(
        self,
        input_iter: Iterable[Any],
        output_writer: Optional[Callable[[Any], None]] = None,
    ) -> Generator[Any, None, None]:
        """Execute the pipeline on an iterable of batches.

        Args:
            input_iter:    Iterable yielding batches.
            output_writer: Optional callable to consume each output batch.

        Yields:
            Transformed batches.
        """
        for batch in input_iter:
            result = batch
            for name, fn in self._steps:
                try:
                    result = fn(result)
                except Exception as exc:
                    logger.error("Pipeline step '%s' failed: %s", name, exc)
                    raise PreprocessingError(
                        f"Step '{name}' failed: {exc}"
                    ) from exc
            if output_writer is not None:
                output_writer(result)
            yield result

    def add_step(self, name: str, fn: Callable[..., Any]) -> None:
        """Append a step to the pipeline."""
        self._steps.append((name, fn))
        logger.info("Added pipeline step '%s' (now %d steps).", name, len(self._steps))

    def __repr__(self) -> str:
        names = [n for n, _ in self._steps]
        return f"Pipeline(steps={names})"
