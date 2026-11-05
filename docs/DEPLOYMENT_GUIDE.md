# Production Deployment & Operations Runbook
==================================================

This guide provides end-to-end instructions for deploying, operating, and scaling the **Distributed GPU Inference Platform** in production enterprise environments across Amazon Web Services (AWS EKS), Google Cloud Platform (GCP GKE), or on-premise bare-metal NVIDIA DGX SuperPOD clusters.

---

## Table of Contents
1. [Architecture Prerequisites & Quotas](#1-architecture-prerequisites--quotas)
2. [Infrastructure Provisioning via Terraform](#2-infrastructure-provisioning-via-terraform)
3. [NVIDIA GPU Operator Installation & DCGM](#3-nvidia-gpu-operator-installation--dcgm)
4. [Distributed Storage Setup (S3 / MinIO / NVMe Cache)](#4-distributed-storage-setup)
5. [Deploying the Inference Platform Stack](#5-deploying-the-inference-platform-stack)
6. [KEDA Dynamic GPU Autoscaling](#6-keda-dynamic-gpu-autoscaling)
7. [Zero-Downtime Model Rollouts & Canary Routing](#7-zero-downtime-model-rollouts--canary-routing)
8. [Production Troubleshooting & Chaos Recovery](#8-production-troubleshooting--chaos-recovery)
9. [Cost Optimization & Compute Sizing Guide](#9-cost-optimization--compute-sizing-guide)

---

## 1. Architecture Prerequisites & Quotas

### Cloud GPU Quota Verification

Before initiating cluster deployment, ensure your cloud project has sufficient vCPU and GPU accelerator quotas:

#### AWS Service Quotas
* `Running On-Demand G and VT instances`: ≥ 32 vCPUs (for L4 `g6` / A10G `g5`)
* `Running On-Demand P instances`: ≥ 64 vCPUs (for A100 `p4d` / H100 `p5`)

#### GCP Compute Engine Quotas
* `NVIDIA_A100_GPUS`: ≥ 4
* `NVIDIA_H100_GPUS`: ≥ 2
* `NVIDIA_L4_GPUS`: ≥ 2

---

## 2. Infrastructure Provisioning via Terraform

### AWS EKS Multi-Pool Cluster Deployment

```bash
cd terraform/aws

# Initialize Terraform providers
terraform init

# Review execution plan
terraform plan \
  -var="cluster_name=gpu-inference-prod" \
  -var="region=us-west-2"

# Apply infrastructure changes
terraform apply -auto-approve
```

#### Export Kubernetes Configuration:
```bash
aws eks update-kubeconfig --region us-west-2 --name gpu-inference-prod
kubectl get nodes -L node.kubernetes.io/instance-type,nvidia.com/gpu.product
```

### GCP GKE Multi-Pool Cluster Deployment

```bash
cd terraform/gcp

terraform init
terraform plan \
  -var="project_id=my-gpu-production-project" \
  -var="region=us-central1"
terraform apply -auto-approve

gcloud container clusters get-credentials gpu-inference-prod --region us-central1
```

---

## 3. NVIDIA GPU Operator Installation & DCGM

The NVIDIA GPU Operator automates the management of all NVIDIA software components needed to provision GPU worker nodes.

Run the automated installer script:
```bash
./scripts/install-gpu-operator.sh
```

Or deploy manually via Helm:
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set dcgmExporter.enabled=true \
  --set gfd.enabled=true \
  --set mig.strategy=mixed
```

### Verification:
```bash
# Verify GPU Operator daemonsets are running
kubectl get pods -n gpu-operator

# Verify GPU resources are advertised to Kubernetes
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

---

## 4. Distributed Storage Setup

### Model Repository Shared PVC
For multi-node Triton clusters, models are loaded from an S3-compatible object store or a shared high-performance POSIX filesystem (AWS FSx Lustre / GCP Filestore / CephFS):

```bash
kubectl apply -f k8s/storage/minio-secret.yaml
kubectl apply -f k8s/storage/minio-deployment.yaml
kubectl apply -f k8s/storage/model-repository-pvc.yaml
kubectl apply -f k8s/storage/redis-deployment.yaml
```

---

## 5. Deploying the Inference Platform Stack

### Deploying via ArgoCD (GitOps)
```bash
kubectl apply -f k8s/argocd/application.yaml
```

### Deploying via Kustomize Overlays
```bash
# Deploy Production Overlay (Production resources, affinities, HPA, and DCGM monitoring)
kubectl apply -k k8s/overlays/prod

# Verify deployment rollout status
kubectl rollout status deployment/inference-router -n inference --timeout=180s
kubectl rollout status deployment/triton-pool-3-h100 -n inference --timeout=300s
```

---

## 6. KEDA Dynamic GPU Autoscaling

The platform utilizes **KEDA (Kubernetes Event-driven Autoscaling)** to scale GPU pods based on incoming Triton queue depth and Go router request rates rather than crude CPU metrics:

```bash
kubectl apply -f k8s/base/triton-keda-scaledobject.yaml
kubectl apply -f k8s/router/keda-scaledobject.yaml
```

### ScaledObject Metric Trigger
```yaml
triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring.svc.cluster.local:9090
      metricName: triton_queue_duration_us
      query: sum(rate(nv_inference_queue_duration_us[1m])) / sum(rate(nv_inference_request_count[1m]))
      threshold: '5000' # Scale out if avg queue wait exceeds 5ms
```

---

## 7. Zero-Downtime Model Rollouts & Canary Routing

When deploying a new model version (e.g., upgrading ResNet-50 v1 to v2 or updating a TensorRT engine):

1. **Upload Engine to S3 / MinIO Repository**:
   ```bash
   python -m python.model_pipeline.model_registry \
     --model-name resnet50 \
     --version 2 \
     --engine-path /tmp/resnet50_v2.plan \
     --upload-s3 s3://model-repository/models/
   ```

2. **Trigger Triton Dynamic Model Poll**:
   ```bash
   curl -X POST http://router.inference.svc.cluster.local:8080/v2/repository/models/resnet50/load
   ```

3. **Canary Weight Shifting via Go Router**:
   Configure traffic splitting in `config.yaml`:
   ```yaml
   models:
     - name: resnet50
       canary:
         enabled: true
         version_weights:
           "1": 90
           "2": 10
   ```

4. **Monitor Error Rates on Grafana Dashboard**:
   Watch `inference-latency.json` and `router-performance.json` for any deviation in error rate or P99 tail latency. Once verified, promote version 2 to 100%.

---

## 8. Production Troubleshooting & Chaos Recovery

| Symptom | Diagnostic Command | Remediation Action |
| :--- | :--- | :--- |
| **Circuit Breaker Tripped** | `dist-inf health` | Pod consecutively failed health probes. Check Triton logs `kubectl logs -l app=triton-pool-3 -n inference`. |
| **GPU Out of Memory (OOM)** | `nvidia-smi` / DCGM | Reduce `max_batch_size` in `config.pbtxt` or enable Dynamic Memory Allocation. |
| **P99 Latency Spikes** | Check Grafana `inference-latency` | Verify if slow L4 node pool is receiving oversized batches. Adjust router weighting $\beta$ in `config.yaml`. |
| **K8s Scheduling Pending** | `kubectl describe pod <pod-name>` | Insufficient GPU capacity in node pool or missing tolerations for `nvidia.com/gpu:NoSchedule`. |

---

## 9. Cost Optimization & Compute Sizing Guide

To minimize cloud spend without violating latency SLAs:

1. **Spot Instances for Worker Pools**:
   - Run **Pool 1 & 2 (A100)** and **Pool 4 (L4)** on AWS Spot / GCP Preemptible VMs (up to **70% cost reduction**).
   - Keep **Pool 3 (H100)** on reserved instances for mission-critical low-latency LLM workloads.

2. **Tensor Core Precision Optimization**:
   - Always convert FP32 models to **FP16** or **INT8** using `python/model_pipeline/convert_to_tensorrt.py`.
   - INT8 calibration typically halves VRAM requirements and doubles throughput per dollar.

3. **Scale-to-Zero During Off-Peak Hours**:
   - KEDA automatically scales idle worker pools down to 0 replicas during low-traffic windows, waking them within 45 seconds upon the arrival of new inference batches.
