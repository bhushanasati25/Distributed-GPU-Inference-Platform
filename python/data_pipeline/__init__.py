"""
Data Pipeline — Distributed GPU Inference Platform
====================================================

This package provides large-scale data preprocessing, sharding, and loading
utilities designed for distributed GPU inference workloads.

Modules:
    preprocessor:  Image / text / tabular preprocessing with multiprocessing.
    data_loader:   Distributed data loading with caching, prefetching, and
                   memory-mapped access.
    sharding:      Dataset sharding across GPU node pools with verification.
"""

from python.data_pipeline.preprocessor import DataPreprocessor, Pipeline
from python.data_pipeline.data_loader import DistributedDataLoader
from python.data_pipeline.sharding import DatasetSharder

__all__ = [
    "DataPreprocessor",
    "Pipeline",
    "DistributedDataLoader",
    "DatasetSharder",
]
