"""
Unit and integration tests for data processing, sharding, and loading pipelines.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import pyarrow.parquet as pq

from python.data_pipeline.sharding import DatasetSharder, ShardInfo
from python.data_pipeline.data_loader import DistributedDataLoader, BatchData
from python.data_pipeline.preprocessor import DataPreprocessor, TabularSchema


def test_parquet_sharding(synthetic_parquet_dataset: Path, temp_dir: Path):
    """Verify that parquet sharding produces expected shard counts and preserves all records."""
    sharder = DatasetSharder()
    output_dir = temp_dir / "shards"
    
    shards = sharder.shard_parquet(
        input_path=synthetic_parquet_dataset,
        output_dir=output_dir,
        num_shards=5
    )
    
    assert len(shards) == 5
    total_rows = sum(s.row_count for s in shards)
    assert total_rows == 1000
    
    # Verify data integrity
    assert sharder.verify_shards(shards, expected_total_rows=1000)


def test_shard_distribution(synthetic_parquet_dataset: Path, temp_dir: Path):
    """Verify node pool assignment strategies distribute data across GPU pools."""
    sharder = DatasetSharder()
    output_dir = temp_dir / "shards_dist"
    
    shards = sharder.shard_parquet(
        input_path=synthetic_parquet_dataset,
        output_dir=output_dir,
        num_shards=8
    )
    
    pools = ["gpu-pool-1", "gpu-pool-2", "gpu-pool-3", "gpu-pool-4"]
    manifest = sharder.distribute_shards(shards, pools, strategy="balanced")
    
    assert manifest.total_shards == 8
    assert manifest.total_rows == 1000
    
    assigned_pools = {s.assigned_pool for s in manifest.shards}
    assert assigned_pools.issubset(set(pools))
    assert len(assigned_pools) == len(pools)


def test_streaming_data_loader(synthetic_parquet_dataset: Path, temp_dir: Path):
    """Test streaming batches from shards with background prefetching."""
    sharder = DatasetSharder()
    shard_dir = temp_dir / "loader_shards"
    shards = sharder.shard_parquet(synthetic_parquet_dataset, shard_dir, num_shards=4)
    shard_paths = [s.file_path for s in shards]

    loader = DistributedDataLoader(cache_dir=temp_dir / "cache", max_cache_bytes=100 * 1024**2)
    
    batches = list(loader.stream_batches(shard_paths, batch_size=50))
    assert len(batches) == 20  # 1000 total rows / 50 batch size
    
    first_batch = batches[0]
    assert isinstance(first_batch, BatchData)
    assert "feature_a" in first_batch.features
    assert first_batch.features["feature_a"].shape == (50,)
    assert first_batch.batch_size == 50


def test_inference_request_conversion(synthetic_parquet_dataset: Path, temp_dir: Path):
    """Test converting loaded batches into standardized Triton inference request payloads."""
    loader = DistributedDataLoader(cache_dir=temp_dir / "cache")
    batches = list(loader.stream_batches([synthetic_parquet_dataset], batch_size=32))
    
    req = loader.create_inference_requests(batches[0], model_name="resnet50", model_version="1")
    assert req["model_name"] == "resnet50"
    assert req["batch_size"] == 32
    assert len(req["inputs"]) > 0
    assert len(req["raw_data"]) == len(req["inputs"])


def test_image_preprocessing(synthetic_image_dir: Path, temp_dir: Path):
    """Test batch image preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    out_dir = temp_dir / "preprocessed_images"
    
    out_files = preprocessor.preprocess_images(
        input_dir=synthetic_image_dir,
        output_dir=out_dir,
        target_size=(32, 32),
        normalize=True,
        num_workers=2,
        output_format="parquet",
    )
    
    assert len(out_files) > 0
    # Read one output file and verify shape
    table = pq.read_table(out_files[0])
    pydict = table.to_pydict()
    assert "data" in pydict
    sample_flat = np.array(pydict["data"][0])
    assert sample_flat.shape == (32 * 32 * 3,)
