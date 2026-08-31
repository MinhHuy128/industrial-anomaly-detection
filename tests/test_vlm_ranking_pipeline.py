import unittest
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from experiments.vlm_ablation.data.synthetic_anomaly_pairs import (
    apply_patch_swap,
    apply_cutout,
    generate_synthetic_pair
)
from experiments.vlm_ablation.losses.pairwise_ranking_loss import (
    AnomalyPairwiseRankingLoss,
    compute_combined_ranking_loss
)
from experiments.vlm_ablation.models.vlm_adapter import (
    AdaptiveSigmoidGate,
    VLMSelectiveAdapter,
    VLMFeatureModulator
)
from experiments.vlm_ablation.models.vitill_gct_ranking import ViTillGCTRanking
from experiments.vlm_ablation.statistical_verification import compute_bootstrap_ci


class TestVLMRankingPipeline(unittest.TestCase):
    def test_synthetic_pair_generator(self):
        img_a = Image.new("RGB", (392, 392), color=(100, 150, 200))
        img_b = Image.new("RGB", (392, 392), color=(200, 50, 50))

        # Test patch swap
        p_img, p_meta = apply_patch_swap(img_a, img_b, patch_size_range=(32, 64))
        self.assertEqual(p_img.size, (392, 392))
        self.assertEqual(p_meta["type"], "patch_swap")
        self.assertEqual(len(p_meta["bbox"]), 4)

        # Test cutout
        c_img, c_meta = apply_cutout(img_a, patch_size_range=(24, 48), fill_mode="mean")
        self.assertEqual(c_img.size, (392, 392))
        self.assertEqual(c_meta["type"], "cutout")

        # Test pair generator
        im1, im2, rank, meta = generate_synthetic_pair(img_a, img_b)
        self.assertEqual(im1.size, (392, 392))
        self.assertEqual(im2.size, (392, 392))
        self.assertIn(rank, [1, -1])

    def test_adaptive_gate_and_adapter(self):
        gate = AdaptiveSigmoidGate(init_gamma=-4.0)
        w = gate.get_gate_weight()
        self.assertTrue(0.01 < w < 0.03, f"Expected gate weight ~0.018, got {w}")

        adapter = VLMSelectiveAdapter(vlm_dim=512, embed_dim=768)
        dummy_vlm = torch.randn(4, 512)
        proj = adapter(dummy_vlm)
        self.assertEqual(proj.shape, (4, 768))

        modulator = VLMFeatureModulator(vlm_dim=512, embed_dim=768, init_gamma=-4.0)
        dummy_x = torch.randn(4, 784, 768)
        mod_x, g_val = modulator(dummy_x, dummy_vlm)
        self.assertEqual(mod_x.shape, (4, 784, 768))
        self.assertTrue(0.01 < g_val < 0.03)

    def test_pairwise_ranking_loss(self):
        loss_fn = AnomalyPairwiseRankingLoss(margin=0.1)

        s1 = torch.tensor([0.2, 0.8], requires_grad=True)
        s2 = torch.tensor([0.9, 0.1], requires_grad=True)
        y = torch.tensor([1.0, -1.0])

        loss = loss_fn(s1, s2, y)
        self.assertTrue(loss.item() >= 0.0)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        self.assertIsNotNone(s1.grad)
        self.assertIsNotNone(s2.grad)

    def test_vitill_gct_ranking_forward_all_modes(self):
        B, N, C = 2, 784, 768
        dummy_feat_list = [torch.randn(B, N, C) for _ in range(8)]
        dummy_cls = torch.randn(B, C)
        dummy_vlm = torch.randn(B, 512)

        modes = [
            ("gct_baseline", dict(use_gct=True, use_adapter=False, use_modulation=False)),
            ("gct_ranking", dict(use_gct=True, use_adapter=False, use_modulation=False)),
            ("gct_adapter", dict(use_gct=True, use_adapter=True, use_modulation=False)),
            ("gct_adapter_modulation", dict(use_gct=True, use_adapter=True, use_modulation=True)),
            ("gct_adapter_modulation_ranking", dict(use_gct=True, use_adapter=True, use_modulation=True)),
        ]

        for mode_name, kwargs in modes:
            model = ViTillGCTRanking(
                embed_dim=768,
                vlm_dim=512,
                num_decoder_layers=2,
                **kwargs
            )
            en, de, gct_loss, gate_val = model(
                dummy_feat_list,
                cls_token=dummy_cls,
                vlm_feat=dummy_vlm if kwargs.get("use_modulation") or kwargs.get("use_adapter") else None
            )
            self.assertEqual(len(en), 2)
            self.assertEqual(len(de), 2)
            self.assertEqual(en[0].shape, (B, 768, 28, 28))
            self.assertEqual(de[0].shape, (B, 768, 28, 28))
            self.assertTrue(torch.isfinite(gct_loss))

            score = model.compute_anomaly_score_from_maps(en, de, gct_loss, gamma=1.0)
            self.assertEqual(score.shape, (B,))
            self.assertTrue(torch.all(torch.isfinite(score)))

    def test_bootstrap_ci_computation(self):
        np.random.seed(42)
        n = 100
        labels = np.array([0] * 50 + [1] * 50)
        scores_base = np.array([0.1] * 50 + [0.8] * 50) + np.random.normal(0, 0.1, n)
        scores_new = np.array([0.05] * 50 + [0.9] * 50) + np.random.normal(0, 0.1, n)

        ci_res = compute_bootstrap_ci(labels, scores_base, scores_new, n_bootstraps=200, seed=42)
        self.assertIn("baseline_auroc", ci_res)
        self.assertIn("new_model_auroc", ci_res)
        self.assertIn("ci_95_lower", ci_res)
        self.assertIn("ci_95_upper", ci_res)
        self.assertTrue(ci_res["ci_95_lower"] <= ci_res["ci_95_upper"])
        self.assertIn("gate_verdict", ci_res)


if __name__ == "__main__":
    unittest.main()
