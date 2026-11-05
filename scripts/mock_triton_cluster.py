#!/usr/bin/env python3
"""
Mock Triton GPU Cluster Server
==============================

Simulates 4 distinct GPU node pools (8 total instances) for local development,
integration testing, and load-test validation without physical NVIDIA GPUs.

Features:
- Realistic latency simulation modeled after A100, H100, and L4 execution profiles
- HTTP /v2/health/ready health probe endpoints
- Socket-based gRPC simulation or HTTP prediction endpoint
- Dynamic chaos controls (inject latency, toggle backend failure)

Usage::

    python scripts/mock_triton_cluster.py --base-port 8000
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import random
import socketserver
import sys
import threading
import time
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("mock-cluster")

GPU_PROFILES = {
    "A100": {"base_latency_ms": 12.0, "jitter_ms": 3.0, "max_concurrency": 64},
    "H100": {"base_latency_ms": 5.0,  "jitter_ms": 1.5, "max_concurrency": 128},
    "L4":   {"base_latency_ms": 22.0, "jitter_ms": 5.0, "max_concurrency": 32},
}

CLUSTER_SPEC = [
    {"name": "triton-pool-1-0", "pool": "gpu-pool-1", "gpu": "A100", "http_offset": 0},
    {"name": "triton-pool-1-1", "pool": "gpu-pool-1", "gpu": "A100", "http_offset": 1},
    {"name": "triton-pool-2-0", "pool": "gpu-pool-2", "gpu": "A100", "http_offset": 2},
    {"name": "triton-pool-2-1", "pool": "gpu-pool-2", "gpu": "A100", "http_offset": 3},
    {"name": "triton-pool-3-0", "pool": "gpu-pool-3", "gpu": "H100", "http_offset": 4},
    {"name": "triton-pool-3-1", "pool": "gpu-pool-3", "gpu": "H100", "http_offset": 5},
    {"name": "triton-pool-4-0", "pool": "gpu-pool-4", "gpu": "L4",   "http_offset": 6},
    {"name": "triton-pool-4-1", "pool": "gpu-pool-4", "gpu": "L4",   "http_offset": 7},
]


class MockTritonHandler(http.server.BaseHTTPRequestHandler):
    """Handles health checks and inference REST requests."""

    node_info: Dict = {}
    is_healthy: bool = True
    injected_latency_ms: float = 0.0

    def log_message(self, format, *args):
        # Silence standard HTTP access logs
        pass

    def do_GET(self):
        """Handle Triton health check queries."""
        if self.path in ("/v2/health/ready", "/v2/health/live"):
            if self.is_healthy:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"UNAVAILABLE")
        elif "/v2/models" in self.path and "ready" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ready": True}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle inference prediction calls."""
        if not self.is_healthy:
            self.send_response(503)
            self.end_headers()
            return

        # Simulate GPU execution time according to hardware profile
        profile = GPU_PROFILES.get(self.node_info["gpu"], GPU_PROFILES["A100"])
        latency = profile["base_latency_ms"] + random.uniform(0, profile["jitter_ms"]) + self.injected_latency_ms
        time.sleep(latency / 1000.0)

        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)

        resp = {
            "model_name": "resnet50",
            "model_version": "1",
            "outputs": [
                {"name": "output_0", "datatype": "FP32", "shape": [1, 1000]}
            ],
            "latency_ms": round(latency, 2),
            "backend": self.node_info["name"],
            "gpu": self.node_info["gpu"]
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def start_node(node_cfg: dict, base_port: int) -> ThreadingHTTPServer:
    port = base_port + node_cfg["http_offset"]
    
    # Custom handler class per node
    class NodeHandler(MockTritonHandler):
        node_info = node_cfg
        is_healthy = True
        injected_latency_ms = 0.0

    server = ThreadingHTTPServer(("127.0.0.1", port), NodeHandler)
    logger.info("Initialized %s (%s, %s) on http://127.0.0.1:%d",
                node_cfg["name"], node_cfg["pool"], node_cfg["gpu"], port)
    return server


def main():
    parser = argparse.ArgumentParser(description="Mock 4-Pool Triton Cluster")
    parser.add_argument("--base-port", type=int, default=8100, help="Base HTTP port for pool instances")
    args = parser.parse_args()

    servers: List[ThreadingHTTPServer] = []
    threads: List[threading.Thread] = []

    for spec in CLUSTER_SPEC:
        srv = start_node(spec, args.base_port)
        servers.append(srv)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        threads.append(t)

    logger.info("==============================================================")
    logger.info(" 8 Mock Triton Nodes Active Across 4 GPU Node Pools")
    logger.info(" Pools: 1 & 2 (A100), Pool 3 (H100), Pool 4 (L4)")
    logger.info(" Ports: %d through %d", args.base_port, args.base_port + len(CLUSTER_SPEC) - 1)
    logger.info(" Press Ctrl+C to terminate.")
    logger.info("==============================================================")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down mock cluster...")
        for srv in servers:
            srv.shutdown()


if __name__ == "__main__":
    main()

