#!/usr/bin/env bash
# =============================================================================
# setup-cluster.sh
# Bootstraps local or cloud Kubernetes cluster with 4 GPU node pools.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLUSTER_NAME="${CLUSTER_NAME:-gpu-inference-cluster}"
ENV_TYPE="${1:-kind}" # 'kind' or 'cloud'

echo "=========================================================="
echo " Setting up Kubernetes Cluster for GPU Inference Platform"
echo " Environment: ${ENV_TYPE} | Cluster: ${CLUSTER_NAME}"
echo "=========================================================="

# 1. Dependency checks
command -v kubectl >/dev/null 2>&1 || { echo >&2 "kubectl is required but not installed. Aborting."; exit 1; }
command -v helm >/dev/null 2>&1 || { echo >&2 "helm is required but not installed. Aborting."; exit 1; }

if [ "${ENV_TYPE}" == "kind" ]; then
    command -v kind >/dev/null 2>&1 || { echo >&2 "kind is required for local setup. Aborting."; exit 1; }

    echo ">>> Creating kind cluster with multi-worker topology..."
    cat <<EOF | kind create cluster --name "${CLUSTER_NAME}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
  labels:
    node-pool: gpu-pool-1
    gpu-node: "true"
    accelerator: nvidia-a100
- role: worker
  labels:
    node-pool: gpu-pool-2
    gpu-node: "true"
    accelerator: nvidia-a100
- role: worker
  labels:
    node-pool: gpu-pool-3
    gpu-node: "true"
    accelerator: nvidia-h100
- role: worker
  labels:
    node-pool: gpu-pool-4
    gpu-node: "true"
    accelerator: nvidia-l4
EOF
    kubectl cluster-info --context "kind-${CLUSTER_NAME}"
else
    echo ">>> Validating existing cloud/bare-metal cluster..."
    kubectl cluster-info
fi

# 2. Create core namespaces
echo ">>> Creating namespaces: inference, monitoring, data..."
kubectl apply -f "${ROOT_DIR}/k8s/base/namespace.yaml"

# 3. Label nodes if running in existing cluster
if [ "${ENV_TYPE}" != "kind" ]; then
    echo ">>> Verifying node pool labels..."
    kubectl get nodes --show-labels
fi

echo "=========================================================="
echo " Cluster foundation ready!"
echo " Next step: Run ./scripts/install-gpu-operator.sh"
echo "=========================================================="

