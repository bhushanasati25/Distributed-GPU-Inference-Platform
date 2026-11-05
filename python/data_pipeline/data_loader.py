#!/usr/bin/env python3
"""
Distributed Data Loader
=======================

High-performance streaming data loader designed for 10TB+ distributed dataset
inference across multi-GPU Kubernetes clusters.

Features:
- Streaming Parquet and binary data without full memory loading
- Asynchronous prefetching via background thread queue
- Smart LRU disk caching for hot shards (e.g. on node-local NVMe)
- Direct conversion to Triton inference requests and raw payloads
- Thread-safe multi-worker iteration

Usage::

    loader = DistributedDataLoader(cache_dir="/mnt/nvme/cache", max_cache_bytes=50 * 1024**3)
    for batch in loader.stream_batches(["shard_000.parquet", "shard_001.parquet"], batch_size=64):
        requests = loader.create_inference_requests(batch, model_name="resnet50")
        # send requests to router or Triton...
"""

from __future__ import annotations

import collections
import concurrent.futures
import io
import logging
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataLoaderError(Exception):
    """Base exception for data loader errors."""
    pass


class StorageFetchError(DataLoaderError):
    """Raised when fetching shard from remote object storage fails."""
    pass


class CacheExhaustionError(DataLoaderError):
    """Raised when cache operations fail due to storage exhaustion."""
    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class BatchData:
    """Represents a unified batch ready for inference.
    
    Attributes:
        batch_id: Unique string identifier for the batch.
        features: Dictionary of input tensor names to numpy arrays.
        metadata: Associated sample IDs, routing keys, or sequence info.
        batch_size: Number of samples in the batch.
    """
    batch_id: str
    features: Dict[str, np.ndarray]
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    batch_size: int = 0

    def __post_init__(self) -> None:
        if not self.batch_size and self.features:
            first_val = next(iter(self.features.values()))
            self.batch_size = len(first_val)


@dataclass
class ShardMetadata:
    """Metadata describing a data shard."""
    shard_id: str
    remote_uri: str
    local_path: Optional[Path] = None
    size_bytes: int = 0
    row_count: int = 0
    last_accessed: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# LRU Shard Disk Cache
# ---------------------------------------------------------------------------

class ShardDiskCache:
    """Thread-safe LRU disk cache for managing local NVMe storage of remote shards."""

    def __init__(self, cache_dir: Union[str, Path], max_cache_bytes: int = 100 * 1024**3):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = max_cache_bytes
        self._current_size_bytes = 0
        self._entries: collections.OrderedDict[str, ShardMetadata] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._scan_existing_cache()

    def _scan_existing_cache(self) -> None:
        """Scan directory to catalog existing files into the LRU index."""
        with self._lock:
            for item in self.cache_dir.glob("*"):
                if item.is_file():
                    size = item.stat().st_size
                    meta = ShardMetadata(
                        shard_id=item.name,
                        remote_uri="",
                        local_path=item,
                        size_bytes=size,
                        last_accessed=item.stat().st_atime
                    )
                    self._entries[item.name] = meta
                    self._current_size_bytes += size
            logger.info("Initialized ShardDiskCache: %d items, %.2f GB used of %.2f GB cap",
                        len(self._entries),
                        self._current_size_bytes / (1024**3),
                        self.max_cache_bytes / (1024**3))

    def get(self, shard_id: str) -> Optional[Path]:
        """Retrieve local path for a cached shard and refresh LRU priority."""
        with self._lock:
            if shard_id in self._entries:
                self._entries.move_to_end(shard_id)
                meta = self._entries[shard_id]
                meta.last_accessed = time.time()
                if meta.local_path and meta.local_path.exists():
                    return meta.local_path
                # File disappeared externally
                del self._entries[shard_id]
        return None

    def reserve_and_put(self, shard_id: str, size_bytes: int, source_path: Path) -> Path:
        """Reserve space in the cache, evicting oldest items if needed, and place shard."""
        with self._lock:
            # Check if already present
            if shard_id in self._entries:
                self._entries.move_to_end(shard_id)
                return self._entries[shard_id].local_path or (self.cache_dir / shard_id)

            # Evict until space is available
            target_free = self.max_cache_bytes - size_bytes
            while self._current_size_bytes > target_free and self._entries:
                oldest_id, oldest_meta = self._entries.popitem(last=False)
                if oldest_meta.local_path and oldest_meta.local_path.exists():
                    try:
                        oldest_meta.local_path.unlink()
                        self._current_size_bytes -= oldest_meta.size_bytes
                        logger.debug("Evicted shard %s (%d bytes)", oldest_id, oldest_meta.size_bytes)
                    except OSError as e:
                        logger.warning("Failed to remove evicted shard %s: %s", oldest_id, e)

            dest_path = self.cache_dir / shard_id
            if source_path != dest_path:
                shutil.copy2(source_path, dest_path)

            meta = ShardMetadata(
                shard_id=shard_id,
                remote_uri="",
                local_path=dest_path,
                size_bytes=size_bytes,
                last_accessed=time.time()
            )
            self._entries[shard_id] = meta
            self._current_size_bytes += size_bytes
            return dest_path


# ---------------------------------------------------------------------------
# Distributed Data Loader
# ---------------------------------------------------------------------------

class DistributedDataLoader:
    """High-throughput distributed dataset loader and batch assembler."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = "/tmp/inference_cache",
        max_cache_bytes: int = 50 * 1024**3,  # 50 GB
        s3_endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.cache = ShardDiskCache(cache_dir, max_cache_bytes)
        self.s3_endpoint_url = s3_endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self._s3_client = None

    def _get_s3_client(self):
        """Lazy initialization of boto3 / MinIO client."""
        if self._s3_client is None:
            try:
                import boto3
                self._s3_client = boto3.client(
                    "s3",
                    endpoint_url=self.s3_endpoint_url,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                )
            except Exception as e:
                logger.warning("Boto3 not initialized or S3 credentials not configured: %s", e)
        return self._s3_client

    def load_from_storage(
        self,
        storage_url: str,
        shard_ids: Sequence[str],
        max_workers: int = 4,
    ) -> List[Path]:
        """Fetch remote shards from S3/MinIO in parallel into local cache.

        Args:
            storage_url: S3 bucket or path (e.g. 's3://datasets/imagenet10k/').
            shard_ids: List of shard filenames to fetch.
            max_workers: Concurrent download threads.

        Returns:
            List of Path objects pointing to verified local cached files.
        """
        resolved_paths: List[Path] = []
        pending_shards: List[str] = []

        # Check cache first
        for shard_id in shard_ids:
            local = self.cache.get(shard_id)
            if local and local.exists():
                resolved_paths.append(local)
            else:
                pending_shards.append(shard_id)

        if not pending_shards:
            return resolved_paths

        # Parse bucket and prefix
        clean_url = storage_url.replace("s3://", "").strip("/")
        parts = clean_url.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        client = self._get_s3_client()

        def _fetch_one(shard: str) -> Path:
            cached_path = self.cache.cache_dir / shard
            key = f"{prefix}/{shard}".lstrip("/")
            if client:
                try:
                    logger.info("Downloading s3://%s/%s -> %s", bucket, key, cached_path)
                    client.download_file(bucket, key, str(cached_path))
                    size = cached_path.stat().st_size
                    return self.cache.reserve_and_put(shard, size, cached_path)
                except Exception as exc:
                    raise StorageFetchError(f"Failed to fetch shard {shard}: {exc}") from exc
            else:
                # If local file exists, treat as source
                local_cand = Path(storage_url) / shard
                if local_cand.exists():
                    size = local_cand.stat().st_size
                    return self.cache.reserve_and_put(shard, size, local_cand)
                raise StorageFetchError(f"Cannot resolve shard {shard} without active S3 client or valid path")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_shard = {executor.submit(_fetch_one, sid): sid for sid in pending_shards}
            for future in concurrent.futures.as_completed(future_to_shard):
                sid = future_to_shard[future]
                try:
                    path = future.result()
                    resolved_paths.append(path)
                except Exception as e:
                    logger.error("Error downloading shard %s: %s", sid, e)
                    raise

        return resolved_paths

    def stream_batches(
        self,
        shard_paths: Sequence[Union[str, Path]],
        batch_size: int = 64,
        columns: Optional[List[str]] = None,
        shuffle: bool = False,
        prefetch: int = 2,
    ) -> Generator[BatchData, None, None]:
        """Stream batches out of Parquet shards with background prefetching.

        Args:
            shard_paths: List of local paths to Parquet files.
            batch_size: Number of records per batch.
            columns: Specific column names to load.
            shuffle: Whether to shuffle row groups or records.
            prefetch: Queue capacity for background batch generation.

        Yields:
            BatchData structures containing loaded numpy arrays.
        """
        batch_queue: queue.Queue[Optional[BatchData]] = queue.Queue(maxsize=max(1, prefetch))
        error_holder: List[Exception] = []

        def _producer() -> None:
            batch_idx = 0
            try:
                for shard_path in shard_paths:
                    path = Path(shard_path)
                    if not path.exists():
                        logger.warning("Shard file %s does not exist, skipping", path)
                        continue

                    # Read row groups in streaming fashion with pyarrow
                    parquet_file = pq.ParquetFile(path)
                    num_groups = parquet_file.num_row_groups
                    group_indices = list(range(num_groups))
                    if shuffle:
                        import random
                        random.shuffle(group_indices)

                    curr_arrays: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
                    accumulated_rows = 0

                    for rg_idx in group_indices:
                        table = parquet_file.read_row_group(rg_idx, columns=columns)
                        pydict = table.to_pydict()
                        rg_rows = table.num_rows

                        for k, v in pydict.items():
                            curr_arrays[k].append(np.array(v))
                        accumulated_rows += rg_rows

                        # Slice out batches of batch_size
                        while accumulated_rows >= batch_size:
                            # Concatenate current columns
                            merged_features: Dict[str, np.ndarray] = {}
                            leftover_features: Dict[str, List[np.ndarray]] = collections.defaultdict(list)

                            for col_name, arr_list in curr_arrays.items():
                                col_full = np.concatenate(arr_list, axis=0)
                                batch_slice = col_full[:batch_size]
                                remainder = col_full[batch_size:]

                                merged_features[col_name] = batch_slice
                                if len(remainder) > 0:
                                    leftover_features[col_name].append(remainder)

                            b = BatchData(
                                batch_id=f"batch_{batch_idx:08d}",
                                features=merged_features,
                                batch_size=batch_size
                            )
                            batch_queue.put(b)
                            batch_idx += 1

                            curr_arrays = leftover_features
                            accumulated_rows -= batch_size

                    # Flush remainder of this shard if any rows left
                    if accumulated_rows > 0:
                        merged_features = {}
                        for col_name, arr_list in curr_arrays.items():
                            merged_features[col_name] = np.concatenate(arr_list, axis=0)

                        b = BatchData(
                            batch_id=f"batch_{batch_idx:08d}",
                            features=merged_features,
                            batch_size=accumulated_rows
                        )
                        batch_queue.put(b)
                        batch_idx += 1

            except Exception as e:
                logger.error("Exception in prefetch producer: %s", e, exc_info=True)
                error_holder.append(e)
            finally:
                batch_queue.put(None)  # Sentinel to terminate consumer

        # Start producer thread
        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        # Consumer loop
        while True:
            item = batch_queue.get()
            if item is None:
                break
            if error_holder:
                raise error_holder[0]
            yield item

        producer_thread.join()
        if error_holder:
            raise error_holder[0]

    def create_inference_requests(
        self,
        batch: BatchData,
        model_name: str,
        model_version: str = "1",
        input_mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Convert a BatchData instance into standardized Triton request payload.

        Args:
            batch: BatchData holding inputs.
            model_name: Target model identifier.
            model_version: Model version string.
            input_mappings: Optional mapping from column name to model tensor input name.

        Returns:
            Dictionary formatted with Triton inputs specification and raw binary payloads.
        """
        inputs_spec = []
        raw_buffers = []

        mappings = input_mappings or {k: k for k in batch.features.keys()}

        for feat_name, arr in batch.features.items():
            tensor_name = mappings.get(feat_name, feat_name)
            dtype_str = self._numpy_to_triton_dtype(arr.dtype)
            
            # Ensure C-contiguous
            arr_contiguous = np.ascontiguousarray(arr)
            raw_bytes = arr_contiguous.tobytes()

            inputs_spec.append({
                "name": tensor_name,
                "shape": list(arr.shape),
                "datatype": dtype_str,
                "parameters": {"binary_data_size": len(raw_bytes)},
            })
            raw_buffers.append(raw_bytes)

        return {
            "model_name": model_name,
            "model_version": model_version,
            "batch_id": batch.batch_id,
            "batch_size": batch.batch_size,
            "inputs": inputs_spec,
            "raw_data": raw_buffers,
        }

    @staticmethod
    def _numpy_to_triton_dtype(dtype: np.dtype) -> str:
        """Map numpy dtype to Triton string enum."""
        mapping = {
            np.float32: "FP32",
            np.float16: "FP16",
            np.float64: "FP64",
            np.int32: "INT32",
            np.int64: "INT64",
            np.int16: "INT16",
            np.int8: "INT8",
            np.uint8: "UINT8",
            np.bool_: "BOOL",
        }
        for np_t, triton_t in mapping.items():
            if np.issubdtype(dtype, np_t):
                return triton_t
        return "BYTES"

