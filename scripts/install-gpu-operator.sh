#!/usr/bin/env bash
# =============================================================================
# install-gpu-operator.sh
# Deploys NVIDIA GPU Operator via Helm to manage drivers and container toolkit.
# =============================================================================
set -euo pipefail

echo "=========================================================="
echo " Deploying NVIDIA GPU Operator"
echo "=========================================================="

command -v helm >/dev/null 2>&1 || { echo >&2 "helm is required. Aborting."; exit 1; }

echo ">>> Adding NVIDIA helm repository..."
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update

echo ">>> Installing/Upgrading NVIDIA GPU Operator..."
helm upgrade --install gpu-operator nvidia/gpu-operator \
    --namespace gpu-operator \
    --create-namespace \
    --set driver.enabled=true \
    --set toolkit.enabled=true \
    --set dcgmExporter.enabled=true \
    --set gfd.enabled=true \
    --set operator.defaultRuntime=containerd \
    --wait --timeout 10m

echo ">>> Waiting for GPU Operator DaemonSets to roll out..."
kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n gpu-operator --timeout=5m || true

echo ">>> Verifying GPU resources on cluster nodes..."
kubectl describe nodes | grep -A5 "Allocatable" | grep -i "nvidia.com/gpu" || {
    echo "Warning: No nvidia.com/gpu allocatable resources reported yet. Pods may be initializing."
}

echo "=========================================================="
echo " NVIDIA GPU Operator installed successfully!"
echo "=========================================================="

