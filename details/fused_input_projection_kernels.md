# 🎛️ Fused Input Projection Kernels

Fused kernels collapse model memory lookups by packing weights onto identical hardware registers.

## 🚀 Concept & Architecture
Instead of firing 4 separate kernels for $Q, K, V,$ and FFN projections, they are stacked as a single weight matrix.

```mermaid
flowchart TD
    W[Fused Weight Matrix: Q, K, V, FFN Gate] --> MatMul[Single Fused GEMM Kernel]
    X[Input Tokens] --> MatMul
    MatMul --> Split[Unpack to Q, K, V, and FFN Hidden Channels]
```

## 📈 Significance
- Reduced memory latency on modern GPU tensor cores.
- Fuses multiple elementwise operations, preventing HBM read/write stalls.

[↩️ Back to README](../README.md)
