"""
Model Pipeline — Distributed GPU Inference Platform
=====================================================

This package provides tools for converting, optimizing, benchmarking, and
registering ML models for deployment on NVIDIA Triton Inference Server.

Modules:
    convert_to_tensorrt: ONNX/PyTorch → TensorRT engine conversion.
    benchmark:           Triton inference benchmarking (gRPC + HTTP).
    model_registry:      Triton model-repository management with MinIO storage.
"""

from python.model_pipeline.convert_to_tensorrt import TensorRTConverter
from python.model_pipeline.benchmark import InferenceBenchmark
from python.model_pipeline.model_registry import ModelRegistry

__all__ = [
    "TensorRTConverter",
    "InferenceBenchmark",
    "ModelRegistry",
]
