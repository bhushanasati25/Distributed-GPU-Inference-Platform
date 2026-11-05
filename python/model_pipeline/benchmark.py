#!/usr/bin/env python3
"""
Triton Inference Benchmarking
==============================

Benchmark NVIDIA Triton Inference Server via gRPC and HTTP with concurrent
request simulation, warm-up, and detailed latency percentile reporting.

Usage (CLI)::

    python -m python.model_pipeline.benchmark \\
        --server localhost:8001 \\
        --model resnet50 \\
        --input-shape 1,3,224,224 \\
        --requests 1000 \\
        --concurrency 10

Usage (library)::

    from python.model_pipeline.benchmark import InferenceBenchmark

    bench = InferenceBenchmark()
    report = bench.benchmark_triton(
        server_url="localhost:8001",
        model_name="resnet50",
        input_data=np.random.randn(1, 3, 224, 224).astype(np.float32),
        num_requests=1000,
        concurrency=10,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class BenchmarkError(Exception):
    """Generic benchmark failure."""


class TritonConnectionError(BenchmarkError):
    """Raised when the Triton server is unreachable."""


class TritonModelError(BenchmarkError):
    """Raised when the requested model is not available on the server."""


# ---------------------------------------------------------------------------
# Result data-class
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """Comprehensive latency and throughput report for a Triton benchmark run.

    All latency values are in **milliseconds**.
    """

    server_url: str
    model_name: str
    protocol: str  # "grpc" or "http"
    num_requests: int
    concurrency: int
    warmup_requests: int

    # Throughput
    total_time_s: float
    throughput_ips: float  # inferences per second

    # Latency percentiles (ms)
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float

    # First-token latency (same as regular latency for non-streaming models)
    first_token_mean_ms: float = 0.0
    first_token_p95_ms: float = 0.0

    # Error count
    errors: int = 0

    def summary_table(self) -> str:
        """Return a human-readable summary table string."""
        lines = [
            "=" * 60,
            f"  Triton Benchmark Report ({self.protocol.upper()})",
            "=" * 60,
            f"  Server:          {self.server_url}",
            f"  Model:           {self.model_name}",
            f"  Requests:        {self.num_requests}",
            f"  Concurrency:     {self.concurrency}",
            f"  Warm-up:         {self.warmup_requests}",
            "-" * 60,
            f"  Throughput:      {self.throughput_ips:,.1f} inferences/s",
            f"  Total time:      {self.total_time_s:.2f} s",
            "-" * 60,
            f"  Latency (ms)",
            f"    Mean:          {self.mean_ms:.2f}",
            f"    Median:        {self.median_ms:.2f}",
            f"    P50:           {self.p50_ms:.2f}",
            f"    P95:           {self.p95_ms:.2f}",
            f"    P99:           {self.p99_ms:.2f}",
            f"    Min:           {self.min_ms:.2f}",
            f"    Max:           {self.max_ms:.2f}",
            f"    Stddev:        {self.stddev_ms:.2f}",
            "-" * 60,
            f"  First-token (ms)",
            f"    Mean:          {self.first_token_mean_ms:.2f}",
            f"    P95:           {self.first_token_p95_ms:.2f}",
            "-" * 60,
            f"  Errors:          {self.errors}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# InferenceBenchmark
# ---------------------------------------------------------------------------

class InferenceBenchmark:
    """Benchmark Triton Inference Server over gRPC and HTTP.

    Example::

        bench = InferenceBenchmark()
        report = bench.benchmark_triton(
            "localhost:8001", "resnet50",
            np.random.randn(1, 3, 224, 224).astype(np.float32),
        )
        print(report.summary_table())
    """

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def benchmark_triton(
        self,
        server_url: str,
        model_name: str,
        input_data: np.ndarray,
        num_requests: int = 1000,
        concurrency: int = 10,
        protocol: str = "grpc",
        model_version: str = "",
        warmup_requests: int = 50,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> BenchmarkReport:
        """Run a benchmark against a Triton Inference Server.

        Args:
            server_url:      Triton server address (``host:port``).
            model_name:      Name of the model deployed on Triton.
            input_data:      NumPy array for a **single** inference request.
            num_requests:    Total number of inference requests to send.
            concurrency:     Number of concurrent threads issuing requests.
            protocol:        ``"grpc"`` or ``"http"``.
            model_version:   Model version string (empty = latest).
            warmup_requests: Number of warm-up requests (not measured).
            input_name:      Override input tensor name (auto-detected if *None*).
            output_name:     Override output tensor name (auto-detected if *None*).

        Returns:
            A :class:`BenchmarkReport` with latency stats and throughput.

        Raises:
            TritonConnectionError: If the server is unreachable.
            TritonModelError:      If the model is not loaded.
            ValueError:            If *protocol* is invalid.
        """
        protocol = protocol.lower()
        if protocol not in ("grpc", "http"):
            raise ValueError(f"Unsupported protocol '{protocol}'. Use 'grpc' or 'http'.")

        logger.info(
            "Benchmarking model '%s' on %s via %s  (requests=%d, concurrency=%d)",
            model_name, server_url, protocol.upper(), num_requests, concurrency,
        )

        # --- Create client ---
        client, InferInput, InferRequestedOutput = self._create_client(
            server_url, protocol
        )

        # --- Server health ---
        self._check_server_health(client, protocol)

        # --- Model readiness ---
        self._check_model_ready(client, model_name, model_version, protocol)

        # --- Resolve tensor names ---
        if input_name is None or output_name is None:
            resolved_in, resolved_out = self._resolve_tensor_names(
                client, model_name, model_version, protocol
            )
            input_name = input_name or resolved_in
            output_name = output_name or resolved_out

        # --- Build request factory ---
        def _make_request() -> float:
            """Send one inference and return latency in ms."""
            inp = InferInput(input_name, list(input_data.shape), self._numpy_to_triton_dtype(input_data.dtype))
            inp.set_data_from_numpy(input_data)
            out = InferRequestedOutput(output_name)

            t0 = time.perf_counter()
            client.infer(
                model_name=model_name,
                model_version=model_version,
                inputs=[inp],
                outputs=[out],
            )
            return (time.perf_counter() - t0) * 1000.0

        # --- Warm-up ---
        logger.info("Warm-up: %d requests …", warmup_requests)
        for _ in range(warmup_requests):
            try:
                _make_request()
            except Exception:
                pass  # tolerate warm-up errors

        # --- Timed run ---
        latencies: List[float] = []
        errors = 0
        lock = threading.Lock()

        def _worker(n: int) -> None:
            nonlocal errors
            local_client, LInferInput, LInferRequestedOutput = self._create_client(
                server_url, protocol
            )
            for _ in range(n):
                try:
                    inp = LInferInput(input_name, list(input_data.shape), self._numpy_to_triton_dtype(input_data.dtype))
                    inp.set_data_from_numpy(input_data)
                    out = LInferRequestedOutput(output_name)

                    t0 = time.perf_counter()
                    local_client.infer(
                        model_name=model_name,
                        model_version=model_version,
                        inputs=[inp],
                        outputs=[out],
                    )
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    with lock:
                        latencies.append(elapsed)
                except Exception as exc:
                    logger.debug("Request error: %s", exc)
                    with lock:
                        errors += 1

        # Distribute requests across threads.
        per_thread = num_requests // concurrency
        remainder = num_requests % concurrency
        tasks = [per_thread + (1 if i < remainder else 0) for i in range(concurrency)]

        logger.info("Sending %d requests across %d threads …", num_requests, concurrency)
        wall_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_worker, n) for n in tasks]
            for fut in as_completed(futures):
                fut.result()  # propagate exceptions
        wall_elapsed = time.perf_counter() - wall_t0

        if not latencies:
            raise BenchmarkError("All requests failed — no latency data collected.")

        arr = np.array(latencies)
        report = BenchmarkReport(
            server_url=server_url,
            model_name=model_name,
            protocol=protocol,
            num_requests=num_requests,
            concurrency=concurrency,
            warmup_requests=warmup_requests,
            total_time_s=wall_elapsed,
            throughput_ips=len(latencies) / wall_elapsed,
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            stddev_ms=float(np.std(arr)),
            first_token_mean_ms=float(np.mean(arr)),  # same for non-streaming
            first_token_p95_ms=float(np.percentile(arr, 95)),
            errors=errors,
        )

        logger.info("Benchmark complete.\n%s", report.summary_table())
        return report

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_client(server_url: str, protocol: str):
        """Create a Triton client + InferInput/InferRequestedOutput classes.

        Returns:
            Tuple of (client_instance, InferInput_class, InferRequestedOutput_class).
        """
        if protocol == "grpc":
            try:
                import tritonclient.grpc as grpc_client  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "tritonclient[grpc] is required. Install with: "
                    "pip install tritonclient[all]"
                ) from exc
            client = grpc_client.InferenceServerClient(url=server_url)
            return client, grpc_client.InferInput, grpc_client.InferRequestedOutput
        else:
            try:
                import tritonclient.http as http_client  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "tritonclient[http] is required. Install with: "
                    "pip install tritonclient[all]"
                ) from exc
            client = http_client.InferenceServerClient(url=server_url)
            return client, http_client.InferInput, http_client.InferRequestedOutput

    @staticmethod
    def _check_server_health(client: Any, protocol: str) -> None:
        """Verify the Triton server is alive."""
        try:
            if not client.is_server_live():
                raise TritonConnectionError("Triton server is not live.")
            if not client.is_server_ready():
                raise TritonConnectionError("Triton server is not ready.")
        except Exception as exc:
            if isinstance(exc, TritonConnectionError):
                raise
            raise TritonConnectionError(
                f"Cannot reach Triton server: {exc}"
            ) from exc

    @staticmethod
    def _check_model_ready(
        client: Any,
        model_name: str,
        model_version: str,
        protocol: str,
    ) -> None:
        """Verify the model is loaded and ready."""
        try:
            ready = client.is_model_ready(model_name, model_version=model_version)
            if not ready:
                raise TritonModelError(
                    f"Model '{model_name}' (version='{model_version}') is not ready."
                )
        except Exception as exc:
            if isinstance(exc, TritonModelError):
                raise
            raise TritonModelError(
                f"Error checking model readiness: {exc}"
            ) from exc

    @staticmethod
    def _resolve_tensor_names(
        client: Any,
        model_name: str,
        model_version: str,
        protocol: str,
    ) -> Tuple[str, str]:
        """Auto-detect the first input and output tensor names."""
        try:
            metadata = client.get_model_metadata(model_name, model_version=model_version)
            if protocol == "grpc":
                input_name = metadata.inputs[0].name
                output_name = metadata.outputs[0].name
            else:
                input_name = metadata["inputs"][0]["name"]
                output_name = metadata["outputs"][0]["name"]
            logger.info("Resolved tensors: input='%s', output='%s'", input_name, output_name)
            return input_name, output_name
        except Exception as exc:
            raise TritonModelError(
                f"Cannot resolve tensor names for model '{model_name}': {exc}"
            ) from exc

    @staticmethod
    def _numpy_to_triton_dtype(np_dtype: np.dtype) -> str:
        """Map a NumPy dtype to a Triton datatype string."""
        mapping = {
            np.float16: "FP16",
            np.float32: "FP32",
            np.float64: "FP64",
            np.int8: "INT8",
            np.int16: "INT16",
            np.int32: "INT32",
            np.int64: "INT64",
            np.uint8: "UINT8",
            np.uint16: "UINT16",
            np.uint32: "UINT32",
            np.uint64: "UINT64",
            np.bool_: "BOOL",
        }
        dtype_type = np_dtype.type
        for np_type, triton_str in mapping.items():
            if np.issubdtype(dtype_type, np_type):
                return triton_str
        return "FP32"  # fallback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-model",
        description="Benchmark a model on NVIDIA Triton Inference Server.",
    )
    parser.add_argument("--server", required=True, help="Triton server URL (host:port).")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument(
        "--input-shape", required=True,
        help="Comma-separated input shape (e.g. 1,3,224,224).",
    )
    parser.add_argument("--dtype", default="float32", help="Input numpy dtype.")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--protocol", default="grpc", choices=["grpc", "http"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output-json", help="Path to write JSON report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry-point for Triton benchmarking."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    shape = tuple(int(x) for x in args.input_shape.split(","))
    input_data = np.random.randn(*shape).astype(getattr(np, args.dtype))

    bench = InferenceBenchmark()
    report = bench.benchmark_triton(
        server_url=args.server,
        model_name=args.model,
        input_data=input_data,
        num_requests=args.requests,
        concurrency=args.concurrency,
        protocol=args.protocol,
        warmup_requests=args.warmup,
    )

    print(report.summary_table())

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report written to %s", args.output_json)


if __name__ == "__main__":
    main()
