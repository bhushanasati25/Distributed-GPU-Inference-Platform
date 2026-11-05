"""
Unit tests for model registry configuration and Triton directory tree generation.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from python.model_pipeline.model_registry import ModelRegistry, ModelMetadata


def test_create_model_config():
    """Verify generated config.pbtxt content matches Triton specifications."""
    registry = ModelRegistry()
    config_text = registry.create_model_config(
        model_name="resnet50",
        platform="tensorrt_plan",
        inputs=[{"name": "input_0", "data_type": "TYPE_FP32", "dims": [3, 224, 224]}],
        outputs=[{"name": "output_0", "data_type": "TYPE_FP32", "dims": [1000]}],
        max_batch_size=64,
        dynamic_batching=True,
        max_queue_delay_microseconds=1000,
    )
    
    assert 'name: "resnet50"' in config_text
    assert 'platform: "tensorrt_plan"' in config_text
    assert 'max_batch_size: 64' in config_text
    assert 'dynamic_batching {' in config_text
    assert 'max_queue_delay_microseconds: 1000' in config_text
    assert 'name: "input_0"' in config_text
    assert 'dims: [3, 224, 224]' in config_text


def test_register_model_local_directory(temp_dir: Path):
    """Verify local registration creates proper Triton model folder layout."""
    registry = ModelRegistry()
    model_repo = temp_dir / "models"
    
    # Create fake engine file
    fake_engine = temp_dir / "model.plan"
    fake_engine.write_bytes(b"FAKE_TENSORRT_ENGINE_BINARY_DATA")
    
    config_dict = {
        "platform": "tensorrt_plan",
        "inputs": [{"name": "input", "data_type": "TYPE_FP32", "dims": [10]}],
        "outputs": [{"name": "output", "data_type": "TYPE_FP32", "dims": [2]}],
        "max_batch_size": 32,
    }
    
    meta = registry.register_model(
        model_name="test_net",
        version=1,
        model_file_path=fake_engine,
        config=config_dict,
        repo_dir=model_repo
    )
    
    assert (model_repo / "test_net" / "config.pbtxt").exists()
    assert (model_repo / "test_net" / "1" / "model.plan").exists()
    assert (model_repo / "test_net" / "1" / "model.plan").read_bytes() == b"FAKE_TENSORRT_ENGINE_BINARY_DATA"
    assert meta.model_name == "test_net"
    assert meta.version == 1

