# ViTill-GCT V2 Evaluation Results on MVTec LOCO AD

> Image AUROC: computed via sklearn.metrics.roc_auc_score.
> sPRO: computed via official MVTec LOCO AD evaluation kit (Bergmann et al., WACV 2022).

## ViTill-GCT V2 (Proposed)

| Category | Logical AUROC (%) | Structural AUROC (%) | Mean AUROC (%) | sPRO (%) | Latency (ms) | FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BREAKFAST_BOX | 91.94% | 90.70% | 91.32% | 61.63% | 70.89 ms | 14.1 |
| JUICE_BOTTLE | 94.10% | 97.94% | 96.02% | 82.44% | 46.39 ms | 21.6 |
| PUSHPINS | 56.65% | 82.99% | 69.82% | 65.87% | 56.60 ms | 17.7 |
| SCREW_BAG | 68.63% | 94.26% | 81.44% | 65.14% | 57.04 ms | 17.5 |
| SPLICING_CONNECTORS | 90.32% | 99.31% | 94.81% | 79.03% | 54.28 ms | 18.4 |
| **MEAN** | 80.33% | 93.04% | 86.68% | **70.82**% | 57.04 ms | 17.9 |

## Comparative Baseline (Single-Stream)

| Category | Logical AUROC (%) | Structural AUROC (%) | Mean AUROC (%) | sPRO (%) | Latency (ms) | FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BREAKFAST_BOX | 88.97% | 92.51% | 90.74% | 61.68% | 70.95 ms | 14.1 |
| JUICE_BOTTLE | 90.74% | 98.20% | 94.47% | 83.04% | 46.12 ms | 21.7 |
| PUSHPINS | 54.90% | 81.24% | 68.07% | 66.89% | 56.87 ms | 17.6 |
| SCREW_BAG | 59.19% | 93.23% | 76.21% | 65.09% | 57.15 ms | 17.5 |
| SPLICING_CONNECTORS | 88.19% | 99.52% | 93.85% | 79.70% | 54.37 ms | 18.4 |
| **MEAN** | 76.40% | 92.94% | 84.67% | **71.28**% | 57.09 ms | 17.9 |

## Performance Delta (ViTill-GCT V2 vs. Baseline)

| Metric | Baseline | ViTill-GCT V2 | Δ (Delta) |
|:---|:---:|:---:|:---:|
| Logical AUROC | 76.40% | **80.33%** | **+3.93%** |
| Structural AUROC | 92.94% | **93.04%** | **+0.10%** |
| Mean AUROC | 84.67% | **86.68%** | **+2.01%** |
| Latency (batch=1) | 57.09 ms | **57.04 ms** | 17.9 FPS |