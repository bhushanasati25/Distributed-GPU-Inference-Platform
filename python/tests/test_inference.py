"""
Integration tests for Triton Inference Server and Go router endpoints.
"""

from __future__ import annotations

import os
import socket
import pytest
import numpy as np


def _is_server_reachable(host: str, port: int) -> bool:
    """Helper to check if server port is open."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


@pytest.mark.skipif(
    not _is_server_reachable("localhost", 8000),
    reason="Triton HTTP server not reachable on localhost:8000"
)
def test_triton_health_http(triton_http_url: str):
    """Verify Triton server readiness via HTTP endpoint."""
    import urllib.request
    import json
    
    url = f"{triton_http_url}/v2/health/ready"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200


@pytest.mark.skipif(
    not _is_server_reachable("localhost", 8001),
    reason="Triton gRPC server not reachable on localhost:8001"
)
def test_triton_grpc_client_inference(triton_grpc_url: str):
    """Test standard inference call using tritonclient gRPC library."""
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        pytest.skip("tritonclient not installed")

    client = grpcclient.InferenceServerClient(url=triton_grpc_url)
    assert client.is_server_live()
    assert client.is_server_ready()


def test_payload_generation():
    """Verify synthetic inference payload creation without needing live server."""
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    assert dummy_input.nbytes == 1 * 3 * 224 * 224 * 4
    raw_bytes = dummy_input.tobytes()
    assert len(raw_bytes) == dummy_input.nbytes

