import unittest
from pathlib import Path
import sys
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.vitill_gct import ViTillGCT, ViTillBaseline
from src.losses.cosine_loss import combined_loss, global_cosine_hm_percent
from src.eval import compute_auroc, compute_f1_max


class TestViTillCore(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 768
        self.num_decoder_layers = 8
        self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        self.batch_size = 2
        self.num_patches = 784  # 28x28 for 392x392 input with patch 14

    def test_vitill_baseline_forward(self):
        model = ViTillBaseline(
            embed_dim=self.embed_dim,
            num_decoder_layers=self.num_decoder_layers,
            target_layers=self.target_layers
        )
        dummy_feats = [
            torch.randn(self.batch_size, self.num_patches, self.embed_dim)
            for _ in self.target_layers
        ]
        en_list, de_list = model(dummy_feats)
        self.assertEqual(len(en_list), 2)
        self.assertEqual(len(de_list), 2)
        for en, de in zip(en_list, de_list):
            self.assertEqual(en.shape, (self.batch_size, self.embed_dim, 28, 28))
            self.assertEqual(de.shape, (self.batch_size, self.embed_dim, 28, 28))

    def test_vitill_gct_forward(self):
        model = ViTillGCT(
            embed_dim=self.embed_dim,
            num_decoder_layers=self.num_decoder_layers,
            target_layers=self.target_layers
        )
        dummy_feats = [
            torch.randn(self.batch_size, self.num_patches, self.embed_dim)
            for _ in self.target_layers
        ]
        dummy_cls = torch.randn(self.batch_size, self.embed_dim)

        en_list, de_list, gct_loss = model(dummy_feats, dummy_cls)
        self.assertEqual(len(en_list), 2)
        self.assertEqual(len(de_list), 2)
        self.assertTrue(torch.is_tensor(gct_loss))
        self.assertTrue(gct_loss.item() >= 0.0)

    def test_loss_functions(self):
        de_list = [torch.randn(self.batch_size, self.embed_dim, 28, 28, requires_grad=True) for _ in range(2)]
        en_list = [torch.randn(self.batch_size, self.embed_dim, 28, 28) for _ in range(2)]
        gct_loss = torch.tensor(0.15)

        total_loss = combined_loss(en_list, de_list, gct_loss, p=0.9, factor=0.1, gct_lambda=0.1)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(total_loss.item() >= 0.0)

        # verify backpropagation
        total_loss.backward()
        for de in de_list:
            self.assertIsNotNone(de.grad)

    def test_eval_metrics(self):
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores = [0.1, 0.2, 0.15, 0.05, 0.8, 0.85, 0.9, 0.95]
        auroc = compute_auroc(labels, scores)
        self.assertAlmostEqual(auroc, 100.0, places=2)

        f1 = compute_f1_max(labels, scores)
        self.assertAlmostEqual(f1, 100.0, places=2)


if __name__ == "__main__":
    unittest.main()
