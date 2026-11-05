#!/usr/bin/env bash
# =============================================================================
# load-test.sh
# Executes peak concurrent inference load tests against the router.
# Measures throughput (RPS), P50/P95/P99 latency, and router overhead.
# =============================================================================
set -euo pipefail

TARGET_HOST="${1:-localhost:50051}"
CONCURRENCY="${2:-50}"
TOTAL_REQUESTS="${3:-5000}"
PROTO_PATH="pkg/proto/inference.proto"

echo "=========================================================="
echo " Starting Peak Inference Load Test"
echo " Target: ${TARGET_HOST} | Concurrency: ${CONCURRENCY} | Requests: ${TOTAL_REQUESTS}"
echo "=========================================================="

if command -v ghz >/dev/null 2>&1; then
    echo ">>> Running gRPC load test with ghz..."
    ghz --insecure \
        --proto="${PROTO_PATH}" \
        --call="inference.InferenceService.Predict" \
        -c "${CONCURRENCY}" \
        -n "${TOTAL_REQUESTS}" \
        -d '{"model_name":"resnet50","model_version":"1","inputs":[{"name":"input_0","datatype":"FP32","shape":[1,3,224,224]}],"priority":10}' \
        --format=summary \
        "${TARGET_HOST}"
else
    echo ">>> ghz not detected on PATH. Running Python benchmark client..."
    python3 -m python.model_pipeline.benchmark \
        --server "${TARGET_HOST}" \
        --model resnet50 \
        --protocol grpc \
        --requests "${TOTAL_REQUESTS}" \
        --concurrency "${CONCURRENCY}" || {
            echo "Tip: Install ghz for high-throughput gRPC benchmarking: brew install ghz"
        }
fi

echo "=========================================================="
echo " Load Test Complete. Check Prometheus/Grafana for live stats."
echo "=========================================================="

