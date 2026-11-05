# Troubleshooting & Operations Runbook

A guide for SREs, MLOps, and Platform Engineers operating the Distributed GPU Inference Platform.

---

## 1. GPU Out of Memory (CUDA OOM)

### Symptoms
- Triton pod crashes with `CUDA out of memory` or `cudaErrorMemoryAllocation`.
- Prometheus alert `GPUMemoryExhausted` fires (>95% VRAM).

### Root Causes & Remediation
1. **Dynamic Batch Size Too Large**:
   - In `models/<model_name>/config.pbtxt`, reduce `max_batch_size` (e.g. from 64 down to 32) or adjust `preferred_batch_size`.
2. **Concurrent Model Instances**:
   - Check `instance_group [ { count: 2, kind: KIND_GPU } ]`. If each instance consumes 42GB on an 80GB GPU, 2 instances will OOM. Reduce `count` to `1`.
3. **TensorRT Execution Workspace**:
   - Reduce `max_workspace_size_bytes` in TensorRT conversion arguments to leave headroom for dynamic tensor allocations.

---

## 2. Go Router Circuit Breaker Tripped (OPEN)

### Symptoms
- Router logs: `Backend unhealthy! Circuit tripped to OPEN`.
- Metric `router_backend_healthy{backend="..."}` drops to `0`.
- Client receives `status: Unavailable` or increased queueing delays.

### Remediation Steps
1. **Inspect Triton Pod Logs**:
   ```bash
   kubectl logs -n inference triton-inference-server-0 -c triton --tail=100
   ```
2. **Test Triton Health Manually**:
   ```bash
   kubectl exec -it -n inference deploy/inference-router -- curl -i http://triton-0.triton-service:8000/v2/health/ready
   ```
3. **Verify Model Readiness**:
   ```bash
   curl http://<triton-ip>:8000/v2/models/resnet50/ready
   ```
4. Once the backend responds with HTTP 200 for 2 consecutive checks, the router will transition to `HALF-OPEN` and then automatically `CLOSED`.

---

## 3. High Cross-Node Request Latency

### Symptoms
- Metric `router_cross_node_latency_seconds` P99 > 15ms.
- Dashboard panel *Inference Latency Breakdown* shows large gap between Client E2E and Triton Compute time.

### Remediation Steps
1. **Enable MTU 9000 (Jumbo Frames)**:
   - On bare-metal or cloud VPC, verify network interfaces support 9000 MTU to eliminate TCP packet fragmentation for large tensor payloads.
2. **Review Node Anti-Affinity & Locality**:
   - Verify router pods are scheduled in the same availability zone or top-of-rack switch as their respective GPU worker pools.
3. **Check Connection Pool Saturation**:
   - Check `router_grpc_pool_active` vs `router_grpc_pool_size`. If connections are constantly exhausted, increase gRPC pool size in `internal/router/router.go`.

