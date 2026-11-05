#!/usr/bin/env python3
"""
Model Registry — Triton Model Repository Management
=====================================================

Manages the lifecycle of models in a Triton Inference Server model repository,
including config.pbtxt generation, directory layout, and MinIO (S3-compatible)
storage integration.

Usage (library)::

    from python.model_pipeline.model_registry import ModelRegistry

    registry = ModelRegistry(repo_root="/models")
    registry.register_model(
        model_name="resnet50",
        version=1,
        engine_path="/tmp/resnet50.plan",
        config={
            "platform": "tensorrt_plan",
            "max_batch_size": 32,
            "inputs": [{"name": "input", "data_type": "TYPE_FP32", "dims": [3, 224, 224]}],
            "outputs": [{"name": "output", "data_type": "TYPE_FP32", "dims": [1000]}],
        },
    )
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class ModelRegistryError(Exception):
    """Base exception for model registry operations."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when a model or version does not exist in the registry."""


class ModelAlreadyExistsError(ModelRegistryError):
    """Raised when registering a duplicate model version without force."""


class InvalidPlatformError(ModelRegistryError):
    """Raised for unsupported platform strings."""


class StorageError(ModelRegistryError):
    """Raised on MinIO / S3 upload or list failures."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping of platform → canonical model file name used by Triton.
PLATFORM_MODEL_FILE: Dict[str, str] = {
    "tensorrt_plan": "model.plan",
    "onnxruntime_onnx": "model.onnx",
    "pytorch_libtorch": "model.pt",
    "tensorflow_savedmodel": "model.savedmodel",
}

#: Supported Triton data-type strings.
TRITON_DATA_TYPES = frozenset({
    "TYPE_BOOL",
    "TYPE_UINT8", "TYPE_UINT16", "TYPE_UINT32", "TYPE_UINT64",
    "TYPE_INT8", "TYPE_INT16", "TYPE_INT32", "TYPE_INT64",
    "TYPE_FP16", "TYPE_FP32", "TYPE_FP64",
    "TYPE_STRING",
})


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    """Specification for a single model input or output tensor."""
    name: str
    data_type: str
    dims: List[int]
    reshape: Optional[List[int]] = None

    def to_pbtxt(self) -> str:
        """Render as a protobuf-text block for ``config.pbtxt``."""
        lines = [
            "  {",
            f'    name: "{self.name}"',
            f"    data_type: {self.data_type}",
            f"    dims: [{', '.join(str(d) for d in self.dims)}]",
        ]
        if self.reshape is not None:
            lines.append(f"    reshape {{ shape: [{', '.join(str(d) for d in self.reshape)}] }}")
        lines.append("  }")
        return "\n".join(lines)


@dataclass
class DynamicBatchingConfig:
    """Dynamic batching settings for Triton."""
    preferred_batch_size: List[int] = field(default_factory=lambda: [4, 8, 16])
    max_queue_delay_microseconds: int = 100
    preserve_ordering: bool = False

    def to_pbtxt(self) -> str:
        sizes = ", ".join(str(s) for s in self.preferred_batch_size)
        lines = [
            "dynamic_batching {",
            f"  preferred_batch_size: [{sizes}]",
            f"  max_queue_delay_microseconds: {self.max_queue_delay_microseconds}",
        ]
        if self.preserve_ordering:
            lines.append("  preserve_ordering: true")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class ModelInfo:
    """Metadata for a registered model version."""
    model_name: str
    version: int
    platform: str
    model_file: str
    config_path: str
    local_path: str
    storage_path: Optional[str] = None


#: Alias for ModelInfo used across the platform test suites and APIs.
ModelMetadata = ModelInfo


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Manage a Triton model repository on local disk and optionally in MinIO.

    Args:
        repo_root: Root directory of the local model repository.
    """

    def __init__(self, repo_root: str = "/models") -> None:
        self._repo_root = os.path.abspath(repo_root)
        os.makedirs(self._repo_root, exist_ok=True)
        logger.info("Model registry initialised at %s", self._repo_root)

    # ------------------------------------------------------------------ #
    # Register model
    # ------------------------------------------------------------------ #

    def register_model(
        self,
        model_name: str,
        version: int,
        engine_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        force: bool = False,
        model_file_path: Optional[Union[str, Path]] = None,
        repo_dir: Optional[Union[str, Path]] = None,
    ) -> ModelInfo:
        """Register a model version in the Triton model repository.

        Creates the canonical directory layout::

            <repo_root>/<model_name>/
                config.pbtxt
                <version>/
                    model.plan  (or model.onnx etc.)

        Args:
            model_name:       Triton model name.
            version:          Integer version number (≥ 1).
            engine_path:      Path to the compiled engine / model file.
            config:           Dict with keys: ``platform``, ``max_batch_size``,
                              ``inputs`` (list of dicts), ``outputs`` (list of dicts).
                              Optionally ``dynamic_batching`` (dict) and ``instance_group``.
            force:            If *True*, overwrite an existing version.
            model_file_path:  Alias for engine_path.
            repo_dir:         Optional override for local repository root.

        Returns:
            :class:`ModelInfo` describing the registered model.

        Raises:
            FileNotFoundError:        If *engine_path* does not exist.
            ModelAlreadyExistsError:   If the version exists and *force* is False.
            InvalidPlatformError:      If the platform is unsupported.
        """
        raw_path = engine_path if engine_path is not None else model_file_path
        if raw_path is None:
            raise ValueError("Either engine_path or model_file_path must be specified.")
        engine_path_str = os.path.abspath(str(raw_path))
        if not os.path.exists(engine_path_str):
            raise FileNotFoundError(f"Engine file not found: {engine_path_str}")

        config = config or {}
        platform = config.get("platform", "")
        if platform not in PLATFORM_MODEL_FILE:
            raise InvalidPlatformError(
                f"Unsupported platform '{platform}'. "
                f"Supported: {list(PLATFORM_MODEL_FILE.keys())}"
            )

        base_repo = os.path.abspath(str(repo_dir)) if repo_dir is not None else self._repo_root
        os.makedirs(base_repo, exist_ok=True)

        model_dir = os.path.join(base_repo, model_name)
        version_dir = os.path.join(model_dir, str(version))
        model_filename = PLATFORM_MODEL_FILE[platform]
        dest_model = os.path.join(version_dir, model_filename)

        if os.path.exists(version_dir) and not force:
            raise ModelAlreadyExistsError(
                f"Model '{model_name}' version {version} already exists. "
                "Use force=True to overwrite."
            )

        os.makedirs(version_dir, exist_ok=True)

        # Copy engine file.
        if os.path.isdir(engine_path_str):
            # e.g. tensorflow_savedmodel is a directory
            if os.path.exists(dest_model):
                shutil.rmtree(dest_model)
            shutil.copytree(engine_path_str, dest_model)
        else:
            shutil.copy2(engine_path_str, dest_model)
        logger.info("Copied engine to %s", dest_model)

        # Generate config.pbtxt
        config_path = os.path.join(model_dir, "config.pbtxt")
        config_text = self.create_model_config(
            model_name=model_name,
            platform=platform,
            inputs=config.get("inputs", []),
            outputs=config.get("outputs", []),
            max_batch_size=config.get("max_batch_size", 0),
            dynamic_batching=config.get("dynamic_batching"),
            instance_group=config.get("instance_group"),
        )
        with open(config_path, "w") as f:
            f.write(config_text)
        logger.info("Config written to %s", config_path)

        info = ModelInfo(
            model_name=model_name,
            version=version,
            platform=platform,
            model_file=dest_model,
            config_path=config_path,
            local_path=model_dir,
        )
        logger.info("Model '%s' v%d registered successfully.", model_name, version)
        return info

    # ------------------------------------------------------------------ #
    # Create model config
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_model_config(
        model_name: str,
        platform: str,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        max_batch_size: int = 0,
        dynamic_batching: Optional[Union[Dict[str, Any], bool]] = None,
        instance_group: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a Triton ``config.pbtxt`` string.

        Args:
            model_name:       Model name.
            platform:         Triton platform string.
            inputs:           List of input tensor specs (dicts with ``name``,
                              ``data_type``, ``dims`` keys).
            outputs:          List of output tensor specs.
            max_batch_size:   Maximum batch size (0 for explicit-batch models).
            dynamic_batching: Optional dynamic batching config dict or boolean.
            instance_group:   Optional instance group list.
            **kwargs:         Additional options, e.g. max_queue_delay_microseconds.

        Returns:
            The ``config.pbtxt`` content as a string.
        """
        if platform not in PLATFORM_MODEL_FILE:
            raise InvalidPlatformError(f"Unsupported platform: {platform}")

        lines: List[str] = [
            f'name: "{model_name}"',
            f'platform: "{platform}"',
            f"max_batch_size: {max_batch_size}",
            "",
        ]

        # Inputs
        for spec_dict in inputs:
            ts = TensorSpec(
                name=spec_dict["name"],
                data_type=spec_dict["data_type"],
                dims=spec_dict["dims"],
                reshape=spec_dict.get("reshape"),
            )
            lines.append("input [")
            lines.append(ts.to_pbtxt())
            lines.append("]")
            lines.append("")

        # Outputs
        for spec_dict in outputs:
            ts = TensorSpec(
                name=spec_dict["name"],
                data_type=spec_dict["data_type"],
                dims=spec_dict["dims"],
                reshape=spec_dict.get("reshape"),
            )
            lines.append("output [")
            lines.append(ts.to_pbtxt())
            lines.append("]")
            lines.append("")

        # Dynamic batching
        if dynamic_batching:
            if isinstance(dynamic_batching, bool):
                dynamic_dict = {
                    "preferred_batch_size": kwargs.get("preferred_batch_size", [4, 8, 16]),
                    "max_queue_delay_microseconds": kwargs.get("max_queue_delay_microseconds", 100),
                    "preserve_ordering": kwargs.get("preserve_ordering", False),
                }
            else:
                dynamic_dict = dict(dynamic_batching)
                if "max_queue_delay_microseconds" in kwargs:
                    dynamic_dict["max_queue_delay_microseconds"] = kwargs["max_queue_delay_microseconds"]
            dbc = DynamicBatchingConfig(
                preferred_batch_size=dynamic_dict.get("preferred_batch_size", [4, 8, 16]),
                max_queue_delay_microseconds=dynamic_dict.get("max_queue_delay_microseconds", 100),
                preserve_ordering=dynamic_dict.get("preserve_ordering", False),
            )
            lines.append(dbc.to_pbtxt())
            lines.append("")

        # Instance group
        if instance_group:
            for ig in instance_group:
                lines.append("instance_group [")
                lines.append("  {")
                lines.append(f"    count: {ig.get('count', 1)}")
                kind = ig.get("kind", "KIND_GPU")
                lines.append(f"    kind: {kind}")
                if "gpus" in ig:
                    gpus_str = ", ".join(str(g) for g in ig["gpus"])
                    lines.append(f"    gpus: [{gpus_str}]")
                lines.append("  }")
                lines.append("]")
                lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Upload to MinIO / S3
    # ------------------------------------------------------------------ #

    def upload_to_storage(
        self,
        local_path: str,
        storage_path: str,
        storage_client: Any,
        bucket: str = "models",
    ) -> str:
        """Upload a model directory to MinIO (S3-compatible) storage.

        Args:
            local_path:     Local directory to upload (typically ``<repo>/<model>``).
            storage_path:   Prefix / key path in the bucket.
            storage_client: A ``boto3.client('s3', ...)`` or ``minio.Minio`` instance.
            bucket:         Target bucket name.

        Returns:
            The ``s3://<bucket>/<storage_path>`` URI.

        Raises:
            StorageError: On upload failure.
        """
        local_path = os.path.abspath(local_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

        logger.info("Uploading %s → s3://%s/%s …", local_path, bucket, storage_path)

        try:
            # Ensure bucket exists.
            self._ensure_bucket(storage_client, bucket)

            uploaded = 0
            if os.path.isdir(local_path):
                for root, _dirs, files in os.walk(local_path):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        relative = os.path.relpath(file_path, local_path)
                        key = f"{storage_path}/{relative}"
                        self._put_object(storage_client, bucket, key, file_path)
                        uploaded += 1
            else:
                key = f"{storage_path}/{os.path.basename(local_path)}"
                self._put_object(storage_client, bucket, key, local_path)
                uploaded = 1

            uri = f"s3://{bucket}/{storage_path}"
            logger.info("Uploaded %d file(s) to %s", uploaded, uri)
            return uri

        except Exception as exc:
            raise StorageError(f"Upload failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # List models
    # ------------------------------------------------------------------ #

    def list_models(
        self,
        storage_client: Optional[Any] = None,
        bucket: str = "models",
    ) -> List[Dict[str, Any]]:
        """List all registered models.

        When *storage_client* is provided, lists from MinIO; otherwise scans
        the local repository root.

        Args:
            storage_client: Optional boto3 / minio client.
            bucket:         Bucket name (used with *storage_client*).

        Returns:
            List of dicts with ``model_name``, ``versions``, and ``platform``.
        """
        models: List[Dict[str, Any]] = []

        if storage_client is not None:
            try:
                result = self._list_objects(storage_client, bucket, prefix="", delimiter="/")
                for prefix in result:
                    model_name = prefix.strip("/")
                    models.append({
                        "model_name": model_name,
                        "versions": [],
                        "source": "storage",
                    })
            except Exception as exc:
                raise StorageError(f"Failed to list models from storage: {exc}") from exc
        else:
            for entry in sorted(os.listdir(self._repo_root)):
                model_dir = os.path.join(self._repo_root, entry)
                if not os.path.isdir(model_dir):
                    continue
                config_path = os.path.join(model_dir, "config.pbtxt")
                versions = sorted(
                    int(v) for v in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, v)) and v.isdigit()
                )
                platform = ""
                if os.path.isfile(config_path):
                    platform = self._parse_platform_from_config(config_path)
                models.append({
                    "model_name": entry,
                    "versions": versions,
                    "platform": platform,
                    "config_path": config_path if os.path.isfile(config_path) else None,
                    "local_path": model_dir,
                })

        logger.info("Found %d model(s).", len(models))
        return models

    # ------------------------------------------------------------------ #
    # Get model info
    # ------------------------------------------------------------------ #

    def get_model_info(self, model_name: str, version: int) -> ModelInfo:
        """Retrieve metadata for a specific model version.

        Args:
            model_name: Name of the model.
            version:    Integer version.

        Returns:
            :class:`ModelInfo` for the requested version.

        Raises:
            ModelNotFoundError: If the model or version does not exist.
        """
        model_dir = os.path.join(self._repo_root, model_name)
        if not os.path.isdir(model_dir):
            raise ModelNotFoundError(f"Model '{model_name}' not found in registry.")

        version_dir = os.path.join(model_dir, str(version))
        if not os.path.isdir(version_dir):
            raise ModelNotFoundError(
                f"Version {version} of model '{model_name}' not found."
            )

        config_path = os.path.join(model_dir, "config.pbtxt")
        platform = self._parse_platform_from_config(config_path) if os.path.isfile(config_path) else ""

        # Find model file.
        model_file = ""
        for candidate in PLATFORM_MODEL_FILE.values():
            candidate_path = os.path.join(version_dir, candidate)
            if os.path.exists(candidate_path):
                model_file = candidate_path
                break

        return ModelInfo(
            model_name=model_name,
            version=version,
            platform=platform,
            model_file=model_file,
            config_path=config_path,
            local_path=model_dir,
        )

    # ------------------------------------------------------------------ #
    # Delete model version
    # ------------------------------------------------------------------ #

    def delete_model(
        self,
        model_name: str,
        version: int,
        delete_all_versions: bool = False,
    ) -> None:
        """Remove a model version (or the entire model) from the local repository.

        Args:
            model_name:          Name of the model.
            version:             Version to delete (ignored when *delete_all_versions*).
            delete_all_versions: If *True*, remove the entire model directory.

        Raises:
            ModelNotFoundError: If the model or version does not exist.
        """
        model_dir = os.path.join(self._repo_root, model_name)
        if not os.path.isdir(model_dir):
            raise ModelNotFoundError(f"Model '{model_name}' not found.")

        if delete_all_versions:
            shutil.rmtree(model_dir)
            logger.info("Deleted model '%s' (all versions).", model_name)
            return

        version_dir = os.path.join(model_dir, str(version))
        if not os.path.isdir(version_dir):
            raise ModelNotFoundError(
                f"Version {version} of model '{model_name}' not found."
            )

        shutil.rmtree(version_dir)
        logger.info("Deleted model '%s' version %d.", model_name, version)

        # If no versions remain, clean up config too.
        remaining = [
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()
        ]
        if not remaining:
            shutil.rmtree(model_dir)
            logger.info("No versions remain — removed model directory '%s'.", model_name)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_platform_from_config(config_path: str) -> str:
        """Extract the ``platform`` field from a config.pbtxt file."""
        try:
            with open(config_path, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("platform:"):
                        return stripped.split('"')[1]
        except Exception:
            pass
        return ""

    @staticmethod
    def _ensure_bucket(client: Any, bucket: str) -> None:
        """Create a bucket if it does not exist (supports boto3 and minio)."""
        # Try boto3-style first.
        if hasattr(client, "head_bucket"):
            try:
                client.head_bucket(Bucket=bucket)
            except Exception:
                client.create_bucket(Bucket=bucket)
                logger.info("Created bucket '%s'.", bucket)
        elif hasattr(client, "bucket_exists"):
            # minio-py
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info("Created bucket '%s'.", bucket)

    @staticmethod
    def _put_object(client: Any, bucket: str, key: str, file_path: str) -> None:
        """Upload a single file (supports boto3 and minio)."""
        if hasattr(client, "upload_file"):
            # boto3
            client.upload_file(file_path, bucket, key)
        elif hasattr(client, "fput_object"):
            # minio-py
            client.fput_object(bucket, key, file_path)
        else:
            raise StorageError(
                "Unsupported storage client — expected boto3 S3 client or minio.Minio."
            )

    @staticmethod
    def _list_objects(
        client: Any, bucket: str, prefix: str, delimiter: str = "/"
    ) -> List[str]:
        """List object prefixes in a bucket."""
        if hasattr(client, "list_objects_v2"):
            # boto3
            resp = client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, Delimiter=delimiter
            )
            return [
                cp["Prefix"] for cp in resp.get("CommonPrefixes", [])
            ]
        elif hasattr(client, "list_objects"):
            # minio-py
            objects = client.list_objects(bucket, prefix=prefix, recursive=False)
            return [obj.object_name for obj in objects if obj.is_dir]
        return []
