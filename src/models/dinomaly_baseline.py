"""
Dinomaly Baseline Wrapper Module (Standalone Prototype).
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DINOMALY_DIR = ROOT / "Dinomaly"
if str(DINOMALY_DIR) not in sys.path:
    sys.path.append(str(DINOMALY_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
class DinomalyBaseline(nn.Module):
    """Baseline Dinomaly architecture wrapper using standard PyTorch TransformerDecoder."""
    def __init__(self, embed_dim=768, num_decoder_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_decoder_layers = num_decoder_layers
        
        # Bottleneck MLP: [B, N, 768] -> [B, N, 768]
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Standard PyTorch Transformer Decoder: 8 layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, x):
        # x: [B, N, 768] -> returns reconstructed tokens [B, N, 768]
        bottleneck_out = self.bottleneck(x)
        reconstructed = self.decoder(bottleneck_out, bottleneck_out)
        return reconstructed


if __name__ == "__main__":
    model = DinomalyBaseline(embed_dim=768, num_decoder_layers=8)
    dummy_input = torch.randn(2, 1024, 768)
    out = model(dummy_input)
    print(f"[SUCCESS] DinomalyBaseline forward test passed! Output shape: {out.shape}")
