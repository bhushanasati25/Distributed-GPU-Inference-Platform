#!/usr/bin/env python3
"""
Simulate Peak Inference Workload
================================

Demonstrates the Go router's latency-aware load balancing, priority queuing,
and circuit breaking against the 4 GPU cluster architecture.

Features:
- Submits mixed-priority inference requests concurrently
- Measures P50, P90, P99 client observed latencies
- Reports routing distribution across A100, H100, and L4 pools
- Shows how priority requests preempt lower priority queues

Usage::

    python scripts/simulate_inference.py --concurrency 20 --requests 500
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import statistics
import time
import urllib.request
import urllib.error
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("simulator")


def send_inference(url: str, priority: int, timeout: float = 5.0) -> Dict:
    """Send an inference request through router or mock node."""
    payload = json.dumps({
        "model_name": "resnet50",
        "model_version": "1",
        "inputs": [
            {"name": "input_0", "datatype": "FP32", "shape": [1, 3, 224, 224]}
        ],
        "priority": priority,
        "deadline_ms": int(timeout * 1000)
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)

    t0 = time.perf_counter()
    try:
        with opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "success": True,
                "latency_ms": elapsed,
                "backend": data.get("backend", "unknown"),
                "gpu": data.get("gpu", "unknown"),
                "priority": priority
            }
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "latency_ms": elapsed,
            "error": str(e),
            "priority": priority
        }


def main():
    parser = argparse.ArgumentParser(description="Inference Platform Workload Simulator")
    parser.add_argument("--url", default="http://127.0.0.1:8100/v2/models/resnet50/infer",
                        help="Target router or node URL")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent client threads")
    parser.add_argument("--requests", type=int, default=300, help="Total requests to dispatch")
    args = parser.parse_args()

    logger.info("==============================================================")
    logger.info(" Starting Workload Simulation")
    logger.info(" Target: %s | Workers: %d | Total Requests: %d",
                args.url, args.concurrency, args.requests)
    logger.info("==============================================================")

    results: List[Dict] = []
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # Assign priorities (1 = low, 5 = normal, 10 = critical)
        futures = []
        for i in range(args.requests):
            prio = 10 if (i % 10 == 0) else (5 if (i % 3 == 0) else 1)
            futures.append(executor.submit(send_inference, args.url, prio))

        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_time = time.perf_counter() - t_start
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        logger.error("All %d requests failed. Is the target server running?", len(failed))
        return

    latencies = [r["latency_ms"] for r in successful]
    latencies.sort()

    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.90)]
    p99 = latencies[int(len(latencies) * 0.99)]

    throughput = len(successful) / total_time

    # Group by priority
    prio_stats = {}
    for r in successful:
        p = r["priority"]
        prio_stats.setdefault(p, []).append(r["latency_ms"])

    print("\n" + "=" * 62)
    print(" INFERENCE WORKLOAD BENCHMARK REPORT")
    print("=" * 62)
    print(f" Total Completed:      {len(successful)} / {args.requests} ({len(failed)} errors)")
    print(f" Total Wall-clock:     {total_time:.2f} s")
    print(f" Sustained Throughput: {throughput:.2f} req/s")
    print("-" * 62)
    print(f" P50 Latency:          {p50:.2f} ms")
    print(f" P90 Latency:          {p90:.2f} ms")
    print(f" P99 Latency:          {p99:.2f} ms")
    print(f" Min / Max Latency:    {min(latencies):.2f} ms / {max(latencies):.2f} ms")
    print("-" * 62)
    print(" Latency Breakdown by Request Priority:")
    for prio, l_list in sorted(prio_stats.items(), reverse=True):
        avg_l = statistics.mean(l_list)
        p99_l = l_list[int(len(l_list) * 0.99)] if len(l_list) > 1 else avg_l
        name = "CRITICAL (10)" if prio == 10 else ("NORMAL (5)" if prio == 5 else "LOW (1)")
        print(f"  - {name:<14} [n={len(l_list):<3}]: Avg = {avg_l:5.2f} ms | P99 = {p99_l:5.2f} ms")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
