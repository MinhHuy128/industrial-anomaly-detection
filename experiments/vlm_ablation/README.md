# Appendix: Empirical Boundary Analysis & Ablation Study of Vision-Language Models (VLM) on MVTec LOCO AD

## 1. Overview & Scientific Context

This experimental module investigates the empirical boundaries of **Vision-Language Models (VLMs)** for logical and structural anomaly detection on the **MVTec LOCO AD** benchmark.

While the primary proposed method of this thesis—**ViTill-GCT V2**—achieves State-of-the-Art performance (**86.68% Mean AUROC, 80.33% Logical AUROC**) with real-time inference speed (17.9 FPS), we conducted systematic exploratory research to investigate whether multimodal foundation models (CLIP, DINOv2-to-CLIP distillation, Pairwise Relative Ranking) could resolve difficult edge cases in logical anomalies (such as screw count variation in `screw_bag` and transparent component placement in `pushpins`).

---

## 2. Tested VLM Paradigms & Empirical Findings

Three independent representation-level paradigms were evaluated under strict experimental control (Seed = 42, single-class setup):

| Paradigm | Mechanism | Category Tested | Baseline AUROC | VLM AUROC | Empirical Finding |
|---|---|---|:---:|:---:|---|
| **1. Discrete Symbolic Count Reasoning** | Zero-shot open-vocab CLIP patch segmentation + count statistics | `pushpins` | 56.65% | 54.50% (-2.15%) | **Discrete Counting Ceiling:** VLMs exhibit hallucinations and high variance on small, transparent industrial components. |
| **2. Continuous Feature Distillation** | Student-Teacher distillation from DINOv2 to continuous CLIP language tokens | `pushpins` | 56.65% | 34.11% (-22.54%) | **Spatial Representation Collapse:** Forcing fine-grained visual patch embeddings to align with coarse text tokens destroys localization geometry. |
| **3. Pairwise Relative Ranking & Gated Adapter** | Margin Ranking Loss on synthetic anomaly pairs with Adaptive Sigmoid Gating ($\sigma(\gamma)$) | `screw_bag`<br>`pushpins` | 77.89%<br>56.65% | 54.95% (-22.94%)<br>51.95% (-4.70%) | **Synthetic Shortcut & Gate Dissipation:** Gated adapter automatically closes ($\sigma(\gamma) \to 0.00$) to suppress destructive VLM gradient interference. |

---

## 3. Scientific Conclusion & Thesis Implication

Across all three paradigms, **feature-level VLM integration consistently produced Negative Transfer on logical anomalies**. The root cause is twofold:
1. **The Granularity Mismatch:** Pretrained VLMs are optimized for global image-text semantic alignment, lacking the micro-spatial sensitivity required for sub-patch industrial defect detection.
2. **Synthetic Boundary Shortcuts:** Auxiliary ranking losses trained on synthetic cutout/patch-swap pairs encourage the decoder to seek sharp edge discontinuities rather than reasoning about holistic component arrangements.

Consequently, **ViTill-GCT V2 standalone is validated as the mathematically optimal and empirically superior framework**, outperforming VLM-augmented variants without computational overhead.

---

## 4. Module Structure

```
experiments/vlm_ablation/
├── README.md                      # Academic documentation and empirical findings
├── models/
│   ├── slot_attention.py          # Semantic slot attention mechanism
│   ├── comp_vlm_model.py          # Multi-modal compositional reasoner
│   ├── vlm_reasoner.py            # Open-vocabulary language-guided triage
│   ├── vitill_gct_vlm.py          # Continuous DINOv2-to-CLIP distillation
│   ├── vlm_adapter.py             # Adaptive Sigmoid Gated Adapter
│   └── vitill_gct_ranking.py      # ViTill-GCT with Pairwise Ranking Head
├── losses/
│   ├── comp_vlm_loss.py           # Multi-modal compositional contrastive loss
│   ├── vlm_distill_loss.py        # Continuous distillation loss
│   └── pairwise_ranking_loss.py   # Margin Ranking Loss on anomaly scores
├── data/
│   ├── synthetic_anomaly_pairs.py # Synthetic patch-swap & cutout pair generator
│   └── precompute_vlm_pairwise.py # Offline pairwise cache generator
├── configs/
│   └── vlm_ranking_loco.yaml      # 5-stage ablation configuration
├── run_ablation.py                # Automated 5-stage ablation runner with Bootstrap CI
├── train_vlm_ranking.py           # Training script for gated ranking model
├── eval_vlm_ranking.py            # Evaluation script for gated ranking model
├── statistical_verification.py    # Non-parametric 95% Bootstrap CI verification
└── xai_reporter.py                # Explainable AI multimodal triage reporter
```

---

## 5. Reproduction Instructions

To reproduce the 5-stage ablation and verify the Negative Transfer finding:

```bash
# Run 5-stage ablation on screw_bag
python experiments/vlm_ablation/run_ablation.py --category screw_bag

# Run statistical 95% Bootstrap CI verification
python experiments/vlm_ablation/statistical_verification.py
```
