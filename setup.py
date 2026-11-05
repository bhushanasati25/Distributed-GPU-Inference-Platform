#!/usr/bin/env python3
"""
Root setup.py for Distributed GPU Inference Platform.
"""

from setuptools import setup, find_packages

setup(
    name="distributed-gpu-inference",
    version="1.0.0",
    description="Distributed GPU Inference Platform Model & Data Pipeline",
    author="Platform Engineering",
    packages=find_packages(include=["python", "python.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pyarrow>=14.0.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "boto3>=1.28.0",
        "minio>=7.2.0",
    ],
    entry_points={
        "console_scripts": [
            "convert-model=python.model_pipeline.convert_to_tensorrt:main",
            "benchmark-model=python.model_pipeline.benchmark:main",
            "shard-data=python.data_pipeline.sharding:main",
            "preprocess-data=python.data_pipeline.preprocessor:main",
        ],
    },
)
