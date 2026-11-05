#!/usr/bin/env bash
# =============================================================================
# deploy.sh
# Builds container images and deploys manifests via Kustomize.
# Usage: ./scripts/deploy.sh [dev|prod]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV="${1:-dev}"
REGISTRY="${REGISTRY:-distributed-gpu-inference}"
TAG="${TAG:-latest}"

echo "=========================================================="
echo " Deploying Platform to Environment: ${ENV}"
echo " Image Registry: ${REGISTRY} | Tag: ${TAG}"
echo "=========================================================="

command -v kubectl >/dev/null 2>&1 || { echo >&2 "kubectl required. Aborting."; exit 1; }

# Build Docker images if local docker daemon is active
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo ">>> Building Router Docker image..."
    docker build -t "${REGISTRY}/router:${TAG}" -f "${ROOT_DIR}/docker/Dockerfile.router" "${ROOT_DIR}"

    echo ">>> Building Pipeline Docker image..."
    docker build -t "${REGISTRY}/pipeline:${TAG}" -f "${ROOT_DIR}/docker/Dockerfile.pipeline" "${ROOT_DIR}"

    echo ">>> Building Triton Docker image..."
    docker build -t "${REGISTRY}/triton-custom:${TAG}" -f "${ROOT_DIR}/docker/Dockerfile.triton" "${ROOT_DIR}"

    # If kind cluster exists, load images directly into nodes
    if kind get clusters 2>/dev/null | grep -q "gpu-inference-cluster"; then
        echo ">>> Loading images into Kind cluster..."
        kind load docker-image "${REGISTRY}/router:${TAG}" --name gpu-inference-cluster || true
        kind load docker-image "${REGISTRY}/pipeline:${TAG}" --name gpu-inference-cluster || true
    fi
fi

# Apply Kustomize manifests
OVERLAY_DIR="${ROOT_DIR}/k8s/overlays/${ENV}"
if [ ! -d "${OVERLAY_DIR}" ]; then
    echo "Overlay ${ENV} not found at ${OVERLAY_DIR}. Aborting."
    exit 1
fi

echo ">>> Applying Kubernetes manifests for overlay: ${ENV}..."
kubectl apply -k "${OVERLAY_DIR}"

echo ">>> Waiting for deployments to stabilize..."
kubectl rollout status deployment/inference-router -n inference --timeout=3m || true
kubectl rollout status statefulset/triton-inference-server -n inference --timeout=5m || true

echo "=========================================================="
echo " Deployment Complete!"
echo " Status of services in 'inference' namespace:"
echo "=========================================================="
kubectl get pods,svc,hpa -n inference

echo ""
echo "Access Endpoints:"
echo "  - Router gRPC:    localhost:50051 (kubectl port-forward svc/inference-router 50051:50051 -n inference)"
echo "  - Router HTTP:    localhost:8080  (kubectl port-forward svc/inference-router 8080:8080 -n inference)"
echo "  - Metrics:        localhost:9090  (kubectl port-forward svc/inference-router 9090:9090 -n inference)"
echo "  - Grafana:        localhost:3000  (kubectl port-forward svc/grafana 3000:3000 -n monitoring)"

