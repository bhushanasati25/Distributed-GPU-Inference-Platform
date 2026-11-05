# Distributed GPU Inference Platform — Architecture Deep Dive

## 1. System Topology & Hardware Architecture

The platform coordinates **four heterogeneous GPU node pools** interconnected across a high-speed Kubernetes software-defined overlay network with hardware SR-IOV / RoCE (RDMA over Converged Ethernet) acceleration where available.

```
+-----------------------------------------------------------------------------------+
|                                  API GATEWAY                                      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        GO INFERENCE ROUTER CLUSTER (3-15 PODS)                    |
|  - Latency-Aware Scoring     - Exponential Moving Average Latency                 |
|  - Priority Queue (Heap)     - Active Circuit Breakers                            |
+----+-------------------+-------------------+-------------------+------------------+
     |                   |                   |                   |
     v                   v                   v                   v
+--------------+   +--------------+   +--------------+   +--------------+
|  GPU POOL 1  |   |  GPU POOL 2  |   |  GPU POOL 3  |   |  GPU POOL 4  |
| 2x A100-80GB |   | 2x A100-80GB |   | 2x H100-80GB |   |  2x L4-24GB  |
| PCIe Gen 4   |   | PCIe Gen 4   |   | SXM5 / NVLink|   | PCIe Gen 4   |
| 312 TFLOPS   |   | 312 TFLOPS   |   | 1,979 TFLOPS |   | 242 TFLOPS   |
+--------------+   +--------------+   +--------------+   +--------------+
```

### Hardware Specifications per Node Pool

| Parameter | GPU Pool 1 & 2 | GPU Pool 3 | GPU Pool 4 |
| :--- | :--- | :--- | :--- |
| **GPU Architecture** | NVIDIA Ampere (A100) | NVIDIA Hopper (H100) | NVIDIA Ada Lovelace (L4) |
| **Interconnect** | PCIe Gen4 (64 GB/s) | SXM5 / NVLink 4 (900 GB/s) | PCIe Gen4 (64 GB/s) |
| **VRAM Capacity** | 80 GB HBM2e | 80 GB HBM3 | 24 GB GDDR6 |
| **VRAM Bandwidth**| 2.0 TB/s | 3.35 TB/s | 300 GB/s |
| **Tensor Core FP16**| 312 TFLOPS | 1,979 TFLOPS | 242 TFLOPS |
| **Target Workload** | Heavy Batch / Vision | LLM / High-Throughput Transformer | Low-Latency / Lightweight |

---

## 2. Latency-Aware Routing Algorithm Specification

Standard round-robin load balancers introduce severe head-of-line blocking and straggler latency when distributing tasks across heterogeneous GPU clusters.

### Composite Node Scoring Formula

For each candidate backend node $i$ that passes health probes:

$$\text{Score}_i = \frac{1}{W_i} \left( \alpha \cdot \frac{A_i}{C_i} + \beta \cdot \frac{\bar{L}_i}{L_{\text{baseline}}} + \gamma \cdot \frac{Q_i}{Q_{\text{max}}} \right)$$

Where:
- $W_i$: Configured hardware weight (e.g. $W_{\text{H100}} = 20, W_{\text{A100}} = 10, W_{\text{L4}} = 5$).
- $A_i$: Instantaneous count of active in-flight requests.
- $C_i$: Maximum concurrent request threshold.
- $\bar{L}_i$: Exponential Moving Average (EMA) of inference latency.
- $Q_i$: Current request queue depth on node $i$.
- $\alpha, \beta, \gamma$: Tuning coefficients ($\alpha = 0.40, \beta = 0.40, \gamma = 0.20$).

### Exponential Moving Average Latency Smoothing

$$\bar{L}_{i}^{(t)} = (1 - \lambda) \cdot \bar{L}_{i}^{(t-1)} + \lambda \cdot L_{\text{observed}}$$

With smoothing factor $\lambda = 0.20$. This guarantees quick responsiveness to sudden GPU thermal throttling or PCIe bus contention while filtering out transient outlier packets.

---

## 3. Triton Dynamic Batching & Concurrent Model Execution

In deep learning inference, batching multiple requests together dramatically increases GPU arithmetic intensity (FLOPS per memory byte transferred).

```
Single Request (Batch=1):      [Compute] ------> Idle VRAM waiting for transfer
Dynamic Batching (Batch=32):   [=== Compute ===] Core Utilization > 85%
```

In Triton's `config.pbtxt`:
```protobuf
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 2000
}
```
If incoming requests arrive within 2 milliseconds, Triton dynamically batches them into an optimal tensor dimension before invoking the TensorRT execution context.

