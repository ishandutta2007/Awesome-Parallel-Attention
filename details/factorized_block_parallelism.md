# ✂️ Factorized Block Parallelism

Factorized block parallelism splices the input hidden states along the channel dimension.

## 🚀 Concept & Architecture
Given a model width (e.g., $d_{model} = 4096$), channels are split (e.g., first 2048 to attention, and the rest to FFN/MLP).

```mermaid
flowchart TD
    X[Input Hidden State] --> Split[Channel Splitter]
    Split -->|First 2048 Channels| Attn[Attention Branch]
    Split -->|Remaining 2048 Channels| FFN[FFN Branch]
    Attn --> Concat[Channel Concatenator]
    FFN --> Concat
    Concat --> Add[Residual Add]
    X --> Add
    Add --> Out[Output]
```

## 📈 Significance
- Reduced FLOPs per layer.
- Retains representative capability across distinct parallel branches.

[↩️ Back to README](../README.md)
