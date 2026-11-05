from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory Path."""
    return tmp_path


@pytest.fixture
def synthetic_parquet_dataset(tmp_path: Path) -> Path:
    """Generate a synthetic 1000-row Parquet dataset with feature_a, feature_b, and labels."""
    file_path = tmp_path / "synthetic_dataset.parquet"
    num_rows = 1000
    table = pa.Table.from_pydict({
        "feature_a": np.random.randn(num_rows).astype(np.float32),
        "feature_b": np.random.randn(num_rows).astype(np.float32),
        "label": np.random.randint(0, 2, size=num_rows).astype(np.int64),
    })
    # Write with row_group_size=100 for streaming row groups
    pq.write_table(table, file_path, row_group_size=100)
    return file_path


@pytest.fixture
def synthetic_image_dir(tmp_path: Path) -> Path:
    """Generate synthetic image directory with 5 dummy RGB images."""
    img_dir = tmp_path / "raw_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        img_arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(img_dir / f"img_{i:03d}.png")
    return img_dir


@pytest.fixture
def triton_http_url() -> str:
    """Default local Triton HTTP endpoint."""
    return "http://localhost:8000"


@pytest.fixture
def triton_grpc_url() -> str:
    """Default local Triton gRPC endpoint."""
    return "localhost:8001"
