#!/usr/bin/env python3
"""
TensorRT Model Conversion
==========================

Converts ONNX and PyTorch models to optimised NVIDIA TensorRT engines with
support for FP16/INT8 precision, dynamic shapes, INT8 calibration, and
post-conversion validation + benchmarking.

Usage (CLI)::

    python -m python.model_pipeline.convert_to_tensorrt \\
        --onnx model.onnx \\
        --output model.plan \\
        --precision fp16 \\
        --max-batch 32 \\
        --max-workspace 4

Usage (library)::

    from python.model_pipeline.convert_to_tensorrt import TensorRTConverter

    converter = TensorRTConverter()
    converter.convert_onnx("model.onnx", "model.plan", precision="fp16")
    report = converter.benchmark_engine("model.plan", input_shape=(1, 3, 224, 224))
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class TensorRTConversionError(Exception):
    """Raised when a TensorRT conversion step fails."""


class TensorRTValidationError(Exception):
    """Raised when a built / loaded engine fails validation."""


class TensorRTNotAvailableError(ImportError):
    """Raised when TensorRT is not installed on the system."""


# ---------------------------------------------------------------------------
# Lazy TensorRT import helper
# ---------------------------------------------------------------------------

def _import_trt():
    """Lazily import ``tensorrt`` so the module can be imported even when
    TensorRT is not installed (e.g. during testing or on non-GPU machines)."""
    try:
        import tensorrt as trt  # type: ignore[import-untyped]
        return trt
    except ImportError as exc:
        raise TensorRTNotAvailableError(
            "tensorrt Python package is not installed. "
            "Install it with: pip install tensorrt>=8.6"
        ) from exc


# ---------------------------------------------------------------------------
# INT8 Calibrator
# ---------------------------------------------------------------------------

class _Int8Calibrator:
    """Entropy-based INT8 calibrator backed by a NumPy dataset.

    When ``calibration_data`` is ``None`` a set of random tensors is generated
    automatically which is useful for smoke-testing but **not** recommended
    for production quality.

    Args:
        input_name:        Name of the network input tensor.
        input_shape:       Shape **without** batch dimension, e.g. (3, 224, 224).
        calibration_data:  Optional array of calibration samples (N, *input_shape).
        batch_size:        Number of samples per calibration step.
        cache_file:        Path to write / read the calibration cache.
    """

    def __init__(
        self,
        input_name: str,
        input_shape: Tuple[int, ...],
        calibration_data: Optional[np.ndarray] = None,
        batch_size: int = 32,
        cache_file: str = "calibration.cache",
    ) -> None:
        trt = _import_trt()
        # Store calibrator base — we implement the interface manually to avoid
        # C++ multiple-inheritance issues in some TRT builds.
        self._batch_size = batch_size
        self._cache_file = cache_file
        self._input_name = input_name
        self._current_index = 0

        if calibration_data is not None:
            self._data = calibration_data.astype(np.float32)
        else:
            num_samples = max(batch_size * 10, 512)
            self._data = np.random.randn(num_samples, *input_shape).astype(np.float32)

        self._device_input: Optional[Any] = None
        try:
            import pycuda.driver as cuda  # type: ignore[import-untyped]
            import pycuda.autoinit  # noqa: F401  # type: ignore[import-untyped]
            self._device_input = cuda.mem_alloc(
                self._data[0 : self._batch_size].nbytes
            )
        except ImportError:
            logger.warning(
                "pycuda not installed — INT8 calibration will use CPU fallback. "
                "Install pycuda for full GPU calibration support."
            )

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """Return the next calibration batch (device pointers)."""
        if self._current_index >= len(self._data):
            return None
        end = min(self._current_index + self._batch_size, len(self._data))
        batch = np.ascontiguousarray(self._data[self._current_index : end])
        self._current_index = end

        if self._device_input is not None:
            import pycuda.driver as cuda  # type: ignore[import-untyped]
            cuda.memcpy_htod(self._device_input, batch)
            return [int(self._device_input)]
        return None

    def read_calibration_cache(self) -> Optional[bytes]:
        if os.path.isfile(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        with open(self._cache_file, "wb") as f:
            f.write(cache)
        logger.info("Calibration cache written to %s", self._cache_file)


# ---------------------------------------------------------------------------
# Data-classes for results
# ---------------------------------------------------------------------------

@dataclass
class EngineInfo:
    """Metadata for a loaded TensorRT engine."""

    engine_path: str
    num_bindings: int
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    max_batch_size: int = 0
    device_memory_bytes: int = 0


@dataclass
class BenchmarkReport:
    """Latency statistics from an engine benchmark run."""

    engine_path: str
    input_shape: Tuple[int, ...]
    num_iterations: int
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_ips: float  # inferences per second


# ---------------------------------------------------------------------------
# TensorRTConverter
# ---------------------------------------------------------------------------

class TensorRTConverter:
    """High-level API for converting models to TensorRT engines.

    Example::

        converter = TensorRTConverter()
        converter.convert_onnx(
            "resnet50.onnx",
            "resnet50.plan",
            precision="fp16",
            max_batch_size=32,
        )
        info = converter.validate_engine("resnet50.plan")
        report = converter.benchmark_engine("resnet50.plan", (1, 3, 224, 224))
    """

    # ------------------------------------------------------------------ #
    # ONNX → TensorRT
    # ------------------------------------------------------------------ #

    def convert_onnx(
        self,
        onnx_path: str,
        engine_path: str,
        precision: str = "fp16",
        max_batch_size: int = 32,
        max_workspace_gb: float = 4.0,
        calibration_data: Optional[np.ndarray] = None,
        calibration_cache: str = "calibration.cache",
        min_batch_size: int = 1,
        opt_batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """Convert an ONNX model to a serialised TensorRT engine.

        Args:
            onnx_path:          Path to the source ONNX model file.
            engine_path:        Destination path for the serialised ``.plan`` file.
            precision:          ``"fp32"``, ``"fp16"``, or ``"int8"``.
            max_batch_size:     Maximum batch size for the dynamic shape profile.
            max_workspace_gb:   Maximum GPU workspace memory in GiB.
            calibration_data:   NumPy array of calibration samples for INT8
                                (shape ``(N, *input_shape)``).  Generated
                                automatically when ``None``.
            calibration_cache:  File path for the INT8 calibration cache.
            min_batch_size:     Minimum batch size for the dynamic shape profile.
            opt_batch_size:     Optimal batch size (defaults to ``max_batch_size // 2``).
            verbose:            If ``True`` enable TensorRT verbose logging.

        Returns:
            Absolute path of the written engine file.

        Raises:
            FileNotFoundError:       If *onnx_path* does not exist.
            TensorRTConversionError: If the build fails.
            TensorRTValidationError: If post-build validation fails.
        """
        trt = _import_trt()

        onnx_path = os.path.abspath(onnx_path)
        engine_path = os.path.abspath(engine_path)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        precision = precision.lower()
        if precision not in ("fp32", "fp16", "int8"):
            raise ValueError(f"Unsupported precision '{precision}'. Use fp32, fp16, or int8.")

        opt_batch_size = opt_batch_size or max(max_batch_size // 2, 1)
        logger.info(
            "Starting ONNX→TensorRT conversion: %s → %s  (precision=%s, "
            "batch min/opt/max=%d/%d/%d, workspace=%.1f GiB)",
            onnx_path, engine_path, precision,
            min_batch_size, opt_batch_size, max_batch_size,
            max_workspace_gb,
        )

        # --- TRT Logger ---
        severity = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
        trt_logger = trt.Logger(severity)

        # --- Builder + Network + Parser ---
        builder = trt.Builder(trt_logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)

        logger.info("Parsing ONNX model …")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = "\n".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise TensorRTConversionError(
                    f"ONNX parsing failed with {parser.num_errors} error(s):\n{errors}"
                )
        logger.info("ONNX model parsed successfully — %d layers.", network.num_layers)

        # --- Builder config ---
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(max_workspace_gb * (1 << 30)),
        )

        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                logger.warning("Platform does not have fast FP16 — enabling anyway.")
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 precision enabled.")
        elif precision == "int8":
            if not builder.platform_has_fast_int8:
                logger.warning("Platform does not have fast INT8 — enabling anyway.")
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)  # fallback layers
            logger.info("INT8 + FP16-fallback precision enabled.")

        # --- Dynamic shape optimisation profile ---
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            name = inp.name
            shape = list(inp.shape)
            # Replace the batch dimension (assumed first) with min/opt/max.
            min_shape = [min_batch_size] + shape[1:]
            opt_shape = [opt_batch_size] + shape[1:]
            max_shape = [max_batch_size] + shape[1:]
            profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
            logger.info(
                "Dynamic profile for '%s': min=%s  opt=%s  max=%s",
                name, min_shape, opt_shape, max_shape,
            )
        config.add_optimization_profile(profile)

        # --- INT8 Calibration ---
        if precision == "int8":
            first_input = network.get_input(0)
            input_shape_no_batch = tuple(first_input.shape[1:])
            calibrator = _Int8Calibrator(
                input_name=first_input.name,
                input_shape=input_shape_no_batch,
                calibration_data=calibration_data,
                batch_size=max(min_batch_size, 1),
                cache_file=calibration_cache,
            )
            config.int8_calibrator = calibrator
            logger.info("INT8 calibrator configured (cache=%s).", calibration_cache)

        # --- Build ---
        logger.info("Building TensorRT engine — this may take several minutes …")
        t0 = time.monotonic()
        serialized_engine = builder.build_serialized_network(network, config)
        elapsed = time.monotonic() - t0
        if serialized_engine is None:
            raise TensorRTConversionError("TensorRT engine build returned None.")
        logger.info("Engine built in %.1f s (%d bytes).", elapsed, len(serialized_engine))

        # --- Serialise ---
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        logger.info("Engine serialised to %s", engine_path)

        # --- Validate ---
        self.validate_engine(engine_path)
        return engine_path

    # ------------------------------------------------------------------ #
    # PyTorch → ONNX → TensorRT
    # ------------------------------------------------------------------ #

    def convert_pytorch(
        self,
        model: Any,
        example_input: Any,
        engine_path: str,
        precision: str = "fp16",
        max_batch_size: int = 32,
        max_workspace_gb: float = 4.0,
        opset_version: int = 17,
    ) -> str:
        """Convert a PyTorch model to a TensorRT engine via ONNX export.

        Args:
            model:             A ``torch.nn.Module`` in eval mode.
            example_input:     A tensor (or tuple of tensors) representing a
                               single-batch forward-pass input.
            engine_path:       Destination path for the serialised engine.
            precision:         ``"fp32"``, ``"fp16"``, or ``"int8"``.
            max_batch_size:    Maximum batch size for the dynamic profile.
            max_workspace_gb:  Maximum workspace in GiB.
            opset_version:     ONNX opset version (default 17).

        Returns:
            Absolute path to the written engine file.

        Raises:
            ImportError:             If ``torch`` is not installed.
            TensorRTConversionError: On conversion failure.
        """
        try:
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("PyTorch is required for convert_pytorch.") from exc

        logger.info("Exporting PyTorch model to ONNX (opset=%d) …", opset_version)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx_path = tmp.name

        try:
            dynamic_axes: Dict[str, Dict[int, str]] = {}
            input_names: List[str] = []
            output_names: List[str] = []

            # Determine input names
            if isinstance(example_input, (tuple, list)):
                for idx in range(len(example_input)):
                    name = f"input_{idx}"
                    input_names.append(name)
                    dynamic_axes[name] = {0: "batch_size"}
            else:
                input_names = ["input_0"]
                dynamic_axes["input_0"] = {0: "batch_size"}

            # Trace to discover output count
            with torch.no_grad():
                sample_out = model(
                    *example_input if isinstance(example_input, (tuple, list)) else example_input
                )
            if isinstance(sample_out, (tuple, list)):
                for idx in range(len(sample_out)):
                    name = f"output_{idx}"
                    output_names.append(name)
                    dynamic_axes[name] = {0: "batch_size"}
            else:
                output_names = ["output_0"]
                dynamic_axes["output_0"] = {0: "batch_size"}

            torch.onnx.export(
                model,
                example_input if isinstance(example_input, (tuple, list)) else (example_input,),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )
            logger.info("ONNX export complete → %s", onnx_path)

            # Validate ONNX
            import onnx  # type: ignore[import-untyped]
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validated successfully.")

            return self.convert_onnx(
                onnx_path,
                engine_path,
                precision=precision,
                max_batch_size=max_batch_size,
                max_workspace_gb=max_workspace_gb,
            )
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    # ------------------------------------------------------------------ #
    # Validate
    # ------------------------------------------------------------------ #

    def validate_engine(self, engine_path: str) -> EngineInfo:
        """Load a serialised TensorRT engine and inspect its I/O bindings.

        Args:
            engine_path: Path to a ``.plan`` file.

        Returns:
            An :class:`EngineInfo` instance describing the engine.

        Raises:
            FileNotFoundError:       If *engine_path* does not exist.
            TensorRTValidationError: If the engine cannot be deserialised.
        """
        trt = _import_trt()

        engine_path = os.path.abspath(engine_path)
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise TensorRTValidationError(
                f"Failed to deserialise engine from {engine_path}."
            )

        info = EngineInfo(
            engine_path=engine_path,
            num_bindings=engine.num_io_tensors,
            max_batch_size=0,  # explicit-batch engines report 0
            device_memory_bytes=engine.device_memory_size,
        )

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            binding_info: Dict[str, Any] = {
                "name": name,
                "shape": list(shape),
                "dtype": str(dtype),
            }
            if mode == trt.TensorIOMode.INPUT:
                info.inputs.append(binding_info)
            else:
                info.outputs.append(binding_info)

        logger.info(
            "Engine validated: %d input(s), %d output(s), device memory=%d bytes",
            len(info.inputs), len(info.outputs), info.device_memory_bytes,
        )
        for inp in info.inputs:
            logger.info("  INPUT  %-30s shape=%s dtype=%s", inp["name"], inp["shape"], inp["dtype"])
        for out in info.outputs:
            logger.info("  OUTPUT %-30s shape=%s dtype=%s", out["name"], out["shape"], out["dtype"])

        return info

    # ------------------------------------------------------------------ #
    # Benchmark
    # ------------------------------------------------------------------ #

    def benchmark_engine(
        self,
        engine_path: str,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> BenchmarkReport:
        """Measure inference latency of a TensorRT engine.

        Allocates CUDA memory, runs *warmup_iterations* warm-up passes, then
        measures *num_iterations* inference passes and computes statistics.

        Args:
            engine_path:        Path to a ``.plan`` file.
            input_shape:        Full input shape **including** batch dimension.
            num_iterations:     Number of timed iterations.
            warmup_iterations:  Number of warm-up iterations (not measured).

        Returns:
            A :class:`BenchmarkReport` with latency percentiles and throughput.

        Raises:
            TensorRTValidationError: If the engine cannot be loaded.
            ImportError:             If ``pycuda`` is not installed.
        """
        trt = _import_trt()
        try:
            import pycuda.driver as cuda  # type: ignore[import-untyped]
            import pycuda.autoinit  # noqa: F401  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pycuda is required for benchmarking. "
                "Install with: pip install pycuda"
            ) from exc

        engine_path = os.path.abspath(engine_path)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise TensorRTValidationError(f"Cannot load engine: {engine_path}")

        context = engine.create_execution_context()

        # Set input shape on the context for dynamic-shape engines.
        input_name = engine.get_tensor_name(0)
        context.set_input_shape(input_name, input_shape)

        # Allocate host & device buffers.
        host_inputs: List[np.ndarray] = []
        device_inputs: List[Any] = []
        host_outputs: List[np.ndarray] = []
        device_outputs: List[Any] = []

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = context.get_tensor_shape(name)
            dtype_trt = engine.get_tensor_dtype(name)
            np_dtype = trt.nptype(dtype_trt)
            size = int(np.prod(shape))
            host_buf = np.empty(size, dtype=np_dtype)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)

            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                host_inputs.append(host_buf)
                device_inputs.append(dev_buf)
                context.set_tensor_address(name, int(dev_buf))
            else:
                host_outputs.append(host_buf)
                device_outputs.append(dev_buf)
                context.set_tensor_address(name, int(dev_buf))

        stream = cuda.Stream()

        # Fill input with random data.
        for h in host_inputs:
            h[:] = np.random.randn(*h.shape).astype(h.dtype)

        def _run_once() -> None:
            for h, d in zip(host_inputs, device_inputs):
                cuda.memcpy_htod_async(d, h, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            for h, d in zip(host_outputs, device_outputs):
                cuda.memcpy_dtoh_async(h, d, stream)
            stream.synchronize()

        logger.info("Warming up (%d iterations) …", warmup_iterations)
        for _ in range(warmup_iterations):
            _run_once()

        logger.info("Benchmarking (%d iterations) …", num_iterations)
        latencies: List[float] = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _run_once()
            latencies.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(latencies)
        batch_size = input_shape[0]
        report = BenchmarkReport(
            engine_path=engine_path,
            input_shape=input_shape,
            num_iterations=num_iterations,
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            throughput_ips=batch_size * 1000.0 / float(np.mean(arr)),
        )

        logger.info(
            "Benchmark results  —  mean=%.2f ms  p50=%.2f ms  p95=%.2f ms  "
            "p99=%.2f ms  throughput=%.1f inferences/s",
            report.mean_ms, report.p50_ms, report.p95_ms,
            report.p99_ms, report.throughput_ips,
        )
        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert-model",
        description="Convert ONNX models to TensorRT engines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- convert --
    p_conv = sub.add_parser("convert", help="Convert an ONNX model to TensorRT.")
    p_conv.add_argument("--onnx", required=True, help="Path to ONNX model.")
    p_conv.add_argument("--output", required=True, help="Output engine path (.plan).")
    p_conv.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    p_conv.add_argument("--max-batch", type=int, default=32)
    p_conv.add_argument("--max-workspace", type=float, default=4.0, help="Max workspace in GiB.")
    p_conv.add_argument("--verbose", action="store_true")

    # -- validate --
    p_val = sub.add_parser("validate", help="Validate a TensorRT engine file.")
    p_val.add_argument("--engine", required=True, help="Path to .plan file.")

    # -- benchmark --
    p_bench = sub.add_parser("benchmark", help="Benchmark a TensorRT engine.")
    p_bench.add_argument("--engine", required=True, help="Path to .plan file.")
    p_bench.add_argument(
        "--input-shape", required=True,
        help="Comma-separated input shape, e.g. '1,3,224,224'.",
    )
    p_bench.add_argument("--iterations", type=int, default=100)
    p_bench.add_argument("--warmup", type=int, default=10)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry-point for TensorRT conversion / validation / benchmarking."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    converter = TensorRTConverter()

    if args.command == "convert":
        converter.convert_onnx(
            onnx_path=args.onnx,
            engine_path=args.output,
            precision=args.precision,
            max_batch_size=args.max_batch,
            max_workspace_gb=args.max_workspace,
            verbose=args.verbose,
        )
    elif args.command == "validate":
        info = converter.validate_engine(args.engine)
        print(json.dumps(asdict(info), indent=2))
    elif args.command == "benchmark":
        shape = tuple(int(x) for x in args.input_shape.split(","))
        report = converter.benchmark_engine(
            engine_path=args.engine,
            input_shape=shape,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
        )
        print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
