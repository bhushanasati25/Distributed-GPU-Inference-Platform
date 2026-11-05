"""
Pytest configuration and shared fixtures for Distributed GPU Inference Platform.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory cleaned up after test run."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def synthetic_parquet_dataset(temp_dir: Path) -> Path:
    """Generate a synthetic multi-column Parquet file for data pipeline tests."""
    num_rows = 1000
    data = {
        "id": pa.array([f"sample_{i:06d}" for i in range(num_rows)]),
        "feature_a": pa.array(np.random.randn(num_rows).astype(np.float32)),
        "feature_b": pa.array(np.random.randn(num_rows).astype(np.float32)),
        "label": pa.array(np.random.randint(0, 10, size=num_rows).astype(np.int64)),
    }
    table = pa.Table.from_pydict(data)
    out_file = temp_dir / "synthetic_dataset.parquet"
    pq.write_table(table, out_file, row_group_size=200)
    return out_file


@pytest.fixture
def synthetic_image_dir(temp_dir: Path) -> Path:
    """Generate synthetic dummy images for image preprocessor/sharder testing."""
    from PIL import Image
    img_dir = temp_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(25):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(img_dir / f"test_img_{i:03d}.png")
    return img_dir


@pytest.fixture
def triton_grpc_url() -> str:
    """Return Triton gRPC address from environment or local default."""
    return os.environ.get("TRITON_GRPC_URL", "localhost:8001")


@pytest.fixture
def triton_http_url() -> str:
    """Return Triton HTTP address from environment or local default."""
    return os.environ.get("TRITON_HTTP_URL", "http://localhost:8000")

