#!/usr/bin/env python3
"""
Distributed Dataset Sharder
===========================

Partitions massive (10TB+) datasets into balanced, manageable shards and assigns
them across multi-cluster GPU node pools with data locality optimization.

Features:
- Streaming Parquet sharding by row groups without full-table in-memory loading
- Image dataset archive sharding (Tar/directory)
- CSV dataset streaming sharding
- Node pool distribution strategies: Balanced, Hash-partitioned, and Locality-aware
- Shard verification and data integrity audits
- Output manifest generation for the Go request router and Kubernetes job schedulers

Usage (CLI)::

    python -m python.data_pipeline.sharding \\
        --input /data/huge_dataset.parquet \\
        --output /data/shards \\
        --num-shards 64 \\
        --node-pools gpu-pool-1,gpu-pool-2,gpu-pool-3,gpu-pool-4

Usage (library)::

    sharder = DatasetSharder()
    shards = sharder.shard_parquet("dataset.parquet", "shards/", num_shards=16)
    manifest = sharder.distribute_shards("shards/", ["gpu-pool-1", "gpu-pool-2"])
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sharding")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ShardingError(Exception):
    """Base exception for sharding operations."""
    pass


class IntegrityError(ShardingError):
    """Raised when record count or checksum verification fails."""
    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ShardInfo:
    """Metadata describing an individual dataset shard."""
    shard_id: str
    filename: str
    file_path: str
    size_bytes: int
    row_count: int
    checksum_sha256: str
    assigned_pool: Optional[str] = None


@dataclass
class ShardingManifest:
    """Complete manifest describing a partitioned dataset and cluster assignments."""
    dataset_name: str
    total_shards: int
    total_rows: int
    total_bytes: int
    distribution_strategy: str
    node_pools: List[str]
    shards: List[ShardInfo] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved sharding manifest to %s", path)


# ---------------------------------------------------------------------------
# Dataset Sharder
# ---------------------------------------------------------------------------

class DatasetSharder:
    """High-throughput dataset sharding engine for 10TB+ inference workloads."""

    def __init__(self, buffer_size_mb: int = 128):
        self.buffer_size_bytes = buffer_size_mb * 1024 * 1024

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA256 checksum in chunks."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(4 * 1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    def shard_parquet(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_shards: Optional[int] = None,
        target_shard_size_mb: int = 256,
    ) -> List[ShardInfo]:
        """Shard a large Parquet file or directory of Parquet files into balanced shards.

        Reads row groups iteratively without loading whole tables into RAM.

        Args:
            input_path: Source Parquet file or directory.
            output_dir: Target destination for generated shards.
            num_shards: Explicit count of shards. If None, calculated from target size.
            target_shard_size_mb: Target size in megabytes per shard if num_shards is None.

        Returns:
            List of generated ShardInfo entries.
        """
        src = Path(input_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        files = [src] if src.is_file() else sorted(list(src.glob("*.parquet")))
        if not files:
            raise ShardingError(f"No parquet files found at {input_path}")

        # Gather dataset statistics
        total_rows = 0
        total_bytes = 0
        schema = None

        logger.info("Inspecting source parquet files...")
        for p in files:
            pf = pq.ParquetFile(p)
            total_rows += pf.metadata.num_rows
            total_bytes += p.stat().st_size
            if schema is None:
                schema = pf.schema_arrow

        if num_shards is None:
            num_shards = max(1, math.ceil(total_bytes / (target_shard_size_mb * 1024 * 1024)))

        logger.info(
            "Source dataset: %d rows, %.2f GB across %d files. Target shards: %d",
            total_rows, total_bytes / (1024**3), len(files), num_shards
        )

        rows_per_shard = math.ceil(total_rows / num_shards)
        shard_writers: List[Tuple[pq.ParquetWriter, Path, int, int]] = []
        
        # Prepare writers for each shard
        created_shards: List[ShardInfo] = []
        
        current_shard_idx = 0
        current_shard_rows = 0
        current_writer: Optional[pq.ParquetWriter] = None
        current_shard_path: Optional[Path] = None

        def _open_shard(idx: int) -> Tuple[pq.ParquetWriter, Path]:
            fname = f"shard_{idx:05d}.parquet"
            path = out / fname
            w = pq.ParquetWriter(path, schema, compression="snappy")
            return w, path

        current_writer, current_shard_path = _open_shard(current_shard_idx)

        # Stream row groups
        processed_rows = 0
        for p in files:
            pf = pq.ParquetFile(p)
            for rg_idx in range(pf.num_row_groups):
                rg_table = pf.read_row_group(rg_idx)
                while rg_table.num_rows > 0:
                    space_in_current = rows_per_shard - current_shard_rows
                    if space_in_current <= 0 and current_shard_idx < num_shards - 1:
                        current_writer.close()
                        size = current_shard_path.stat().st_size
                        checksum = self._compute_sha256(current_shard_path)
                        created_shards.append(ShardInfo(
                            shard_id=f"shard_{current_shard_idx:05d}",
                            filename=current_shard_path.name,
                            file_path=str(current_shard_path.resolve()),
                            size_bytes=size,
                            row_count=current_shard_rows,
                            checksum_sha256=checksum
                        ))
                        logger.info("Completed %s: %d rows, %.2f MB",
                                    current_shard_path.name, current_shard_rows, size / (1024**2))

                        current_shard_idx += 1
                        current_shard_rows = 0
                        current_writer, current_shard_path = _open_shard(current_shard_idx)
                        space_in_current = rows_per_shard

                    if current_shard_idx == num_shards - 1:
                        chunk = rg_table
                        rg_table = rg_table.slice(0, 0)
                    else:
                        chunk_len = min(space_in_current, rg_table.num_rows)
                        chunk = rg_table.slice(0, chunk_len)
                        rg_table = rg_table.slice(chunk_len)

                    current_writer.write_table(chunk)
                    current_shard_rows += chunk.num_rows
                    processed_rows += chunk.num_rows

        # Close final shard
        if current_writer:
            current_writer.close()
            size = current_shard_path.stat().st_size
            checksum = self._compute_sha256(current_shard_path)
            created_shards.append(ShardInfo(
                shard_id=f"shard_{current_shard_idx:05d}",
                filename=current_shard_path.name,
                file_path=str(current_shard_path.resolve()),
                size_bytes=size,
                row_count=current_shard_rows,
                checksum_sha256=checksum
            ))
            logger.info("Completed %s: %d rows, %.2f MB",
                        current_shard_path.name, current_shard_rows, size / (1024**2))

        logger.info("Successfully produced %d shards containing %d rows", len(created_shards), processed_rows)
        return created_shards

    def shard_csv(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_shards: int,
    ) -> List[ShardInfo]:
        """Streamingly shard a large CSV file preserving the header on every shard."""
        src = Path(input_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            raise ShardingError(f"CSV file not found: {src}")

        # Count total rows
        logger.info("Counting rows in %s...", src.name)
        total_rows = 0
        with open(src, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline()
            for _ in f:
                total_rows += 1

        rows_per_shard = math.ceil(total_rows / num_shards)
        logger.info("Total data rows: %d, targeting %d rows/shard", total_rows, rows_per_shard)

        shards: List[ShardInfo] = []
        with open(src, "r", encoding="utf-8", errors="replace") as f:
            _ = f.readline()  # skip header
            for s_idx in range(num_shards):
                shard_filename = f"shard_{s_idx:05d}.csv"
                shard_path = out / shard_filename
                s_rows = 0

                with open(shard_path, "w", encoding="utf-8") as out_f:
                    out_f.write(header_line)
                    while s_rows < rows_per_shard:
                        line = f.readline()
                        if not line:
                            break
                        out_f.write(line)
                        s_rows += 1

                if s_rows > 0:
                    shards.append(ShardInfo(
                        shard_id=f"shard_{s_idx:05d}",
                        filename=shard_filename,
                        file_path=str(shard_path.resolve()),
                        size_bytes=shard_path.stat().st_size,
                        row_count=s_rows,
                        checksum_sha256=self._compute_sha256(shard_path)
                    ))
                if s_rows < rows_per_shard:
                    break

        return shards

    def shard_images(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        num_shards: int,
    ) -> List[ShardInfo]:
        """Shard a directory of images into uncompressed or compressed Tar shards."""
        src = Path(input_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [p for p in src.rglob("*") if p.suffix.lower() in valid_exts]
        if not image_files:
            raise ShardingError(f"No valid images found in {input_dir}")

        total_images = len(image_files)
        per_shard = math.ceil(total_images / num_shards)
        logger.info("Partitioning %d images into %d tar shards (~%d images/shard)",
                    total_images, num_shards, per_shard)

        shards: List[ShardInfo] = []
        for s_idx in range(num_shards):
            batch = image_files[s_idx * per_shard : (s_idx + 1) * per_shard]
            if not batch:
                break
            tar_name = f"shard_{s_idx:05d}.tar"
            tar_path = out / tar_name

            with tarfile.open(tar_path, "w") as tar:
                for img_path in batch:
                    tar.add(img_path, arcname=img_path.name)

            shards.append(ShardInfo(
                shard_id=f"shard_{s_idx:05d}",
                filename=tar_name,
                file_path=str(tar_path.resolve()),
                size_bytes=tar_path.stat().st_size,
                row_count=len(batch),
                checksum_sha256=self._compute_sha256(tar_path)
            ))

        return shards

    def distribute_shards(
        self,
        shard_infos: List[ShardInfo],
        node_pools: Sequence[str],
        strategy: str = "balanced",
    ) -> ShardingManifest:
        """Assign shards to GPU node pools according to the chosen distribution strategy.

        Strategies:
        - `balanced`: Round-robin distribution balancing total byte size per pool
        - `hash`: Shard ID modulo node pools
        - `locality`: Group consecutive shards into contiguous node pools

        Args:
            shard_infos: List of ShardInfo objects to assign.
            node_pools: Names of GPU clusters / node pools (e.g. ['gpu-pool-1', ...]).
            strategy: Distribution algorithm.

        Returns:
            ShardingManifest with pool assignments finalized.
        """
        if not node_pools:
            raise ValueError("At least one node pool must be provided")

        num_pools = len(node_pools)
        pool_bytes = {p: 0 for p in node_pools}
        pool_shards: Dict[str, List[ShardInfo]] = {p: [] for p in node_pools}

        if strategy == "balanced":
            # Greedy largest-first bin packing
            sorted_shards = sorted(shard_infos, key=lambda s: s.size_bytes, reverse=True)
            for s in sorted_shards:
                target_pool = min(pool_bytes.keys(), key=lambda p: pool_bytes[p])
                s.assigned_pool = target_pool
                pool_bytes[target_pool] += s.size_bytes
                pool_shards[target_pool].append(s)
        elif strategy == "hash":
            for s in shard_infos:
                idx = int(hashlib.md5(s.shard_id.encode()).hexdigest(), 16) % num_pools
                pool = node_pools[idx]
                s.assigned_pool = pool
                pool_bytes[pool] += s.size_bytes
                pool_shards[pool].append(s)
        elif strategy == "locality":
            # Contiguous chunks
            chunk_size = math.ceil(len(shard_infos) / num_pools)
            for idx, s in enumerate(shard_infos):
                pool_idx = min(idx // chunk_size, num_pools - 1)
                pool = node_pools[pool_idx]
                s.assigned_pool = pool
                pool_bytes[pool] += s.size_bytes
                pool_shards[pool].append(s)
        else:
            raise ValueError(f"Unknown distribution strategy: {strategy}")

        total_rows = sum(s.row_count for s in shard_infos)
        total_bytes = sum(s.size_bytes for s in shard_infos)

        import datetime
        manifest = ShardingManifest(
            dataset_name="dataset",
            total_shards=len(shard_infos),
            total_rows=total_rows,
            total_bytes=total_bytes,
            distribution_strategy=strategy,
            node_pools=list(node_pools),
            shards=shard_infos,
            created_at=datetime.datetime.utcnow().isoformat() + "Z"
        )

        for pool, b_count in pool_bytes.items():
            s_count = len(pool_shards[pool])
            logger.info("Assigned to %s: %d shards, %.2f GB", pool, s_count, b_count / (1024**3))

        return manifest

    def verify_shards(self, shard_infos: List[ShardInfo], expected_total_rows: int) -> bool:
        """Validate integrity and row counts of all shards against original expectations."""
        actual_rows = sum(s.row_count for s in shard_infos)
        if actual_rows != expected_total_rows:
            raise IntegrityError(
                f"Row count mismatch! Expected {expected_total_rows}, found {actual_rows} across {len(shard_infos)} shards"
            )
        for s in shard_infos:
            p = Path(s.file_path)
            if not p.exists():
                raise IntegrityError(f"Shard file missing: {s.file_path}")
            if p.stat().st_size != s.size_bytes:
                raise IntegrityError(f"Shard size changed for {s.filename}")
        logger.info("Verification passed: all %d shards and %d rows intact", len(shard_infos), actual_rows)
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed GPU Dataset Sharder")
    parser.add_argument("--input", "-i", required=True, help="Input parquet, csv, or image directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for shards")
    parser.add_argument("--num-shards", "-n", type=int, default=None, help="Target number of shards")
    parser.add_argument("--type", choices=["parquet", "csv", "images"], default="parquet", help="Data format")
    parser.add_argument("--node-pools", default="gpu-pool-1,gpu-pool-2,gpu-pool-3,gpu-pool-4",
                        help="Comma-separated list of target GPU node pools")
    parser.add_argument("--strategy", choices=["balanced", "hash", "locality"], default="balanced",
                        help="Distribution strategy")
    return parser.parse_args()


def main():
    args = parse_args()
    sharder = DatasetSharder()
    pools = [p.strip() for p in args.node_pools.split(",") if p.strip()]

    logger.info("Starting sharding process for: %s", args.input)
    if args.type == "parquet":
        shards = sharder.shard_parquet(args.input, args.output, num_shards=args.num_shards)
    elif args.type == "csv":
        if args.num_shards is None:
            parser.error("--num-shards required for CSV sharding")
        shards = sharder.shard_csv(args.input, args.output, num_shards=args.num_shards)
    elif args.type == "images":
        if args.num_shards is None:
            parser.error("--num-shards required for image sharding")
        shards = sharder.shard_images(args.input, args.output, num_shards=args.num_shards)

    manifest = sharder.distribute_shards(shards, pools, strategy=args.strategy)
    manifest.save(Path(args.output) / "manifest.json")
    print(f"Sharding complete. Manifest saved at {Path(args.output) / 'manifest.json'}")


if __name__ == "__main__":
    main()

