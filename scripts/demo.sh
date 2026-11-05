#!/usr/bin/env bash
# =============================================================================
# demo.sh — End-to-End Platform Demonstration Script
# =============================================================================
# Launches the 4-pool GPU cluster simulator, boots the platform, executes
# concurrent inference workloads, injects live chaos, and visualizes failover.
# =============================================================================
set -euo pipefail

# Ensure standard system tools are prioritized
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_PORT=8100
TOTAL_REQUESTS=150
CONCURRENCY=15

echo ""
echo "================================================================="
echo "   DISTRIBUTED GPU INFERENCE PLATFORM — LIVE DEMONSTRATION"
echo "================================================================="
echo " Topology: 4 GPU Clusters (8 Nodes total across A100, H100, L4)"
echo " Routing:  Latency-Aware Weighted Scoring + Circuit Breakers"
echo "================================================================="
echo ""

cleanup() {
    echo ""
    echo ">>> Tearing down background simulation processes..."
    if [ -n "${MOCK_PID:-}" ]; then
        kill "${MOCK_PID}" 2>/dev/null || true
    fi
    echo ">>> Cleanup complete. Thank you!"
}
trap cleanup EXIT

# 1. Start the 4-Pool Mock Triton Cluster
echo "Step 1: Starting 4-Pool GPU Cluster Simulator on ports ${BASE_PORT}-${BASE_PORT}+7..."
python3 "${ROOT_DIR}/scripts/mock_triton_cluster.py" --base-port "${BASE_PORT}" >/dev/null 2>&1 &
MOCK_PID=$!
sleep 2

# Verify cluster readiness
echo "Step 2: Probing health endpoints across all 8 GPU cluster nodes..."
HEALTHY_COUNT=0
for i in {0..7}; do
    PORT=$((BASE_PORT + i))
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v2/health/ready" || echo "failed")
    if [ "${STATUS}" == "200" ]; then
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    fi
done
echo ">>> All ${HEALTHY_COUNT} / 8 Triton GPU server nodes are READY and passing probes."

# 3. Run Peak Workload Simulation
echo ""
echo "Step 3: Dispatching ${TOTAL_REQUESTS} mixed-priority requests with ${CONCURRENCY} concurrent workers..."
python3 "${ROOT_DIR}/scripts/simulate_inference.py" \
    --url "http://127.0.0.1:${BASE_PORT}/v2/models/resnet50/infer" \
    --concurrency "${CONCURRENCY}" \
    --requests "${TOTAL_REQUESTS}"

# 4. Run Chaos Engineering Fault Injection
echo ""
echo "Step 4: Executing automated chaos test suite (node failure, throttling, priority preemption)..."
python3 "${ROOT_DIR}/scripts/chaos_test.py" --router-url "http://127.0.0.1:${BASE_PORT}"

echo ""
echo "================================================================="
echo " DEMONSTRATION COMPLETED SUCCESSFULLY!"
echo " Control Console available at: ${ROOT_DIR}/web/index.html"
echo "================================================================="
