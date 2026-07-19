import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add project root and Dinomaly directory to path
ROOT = Path(__file__).resolve().parent.parent.parent
DINOMALY_DIR = ROOT / "Dinomaly"
if str(DINOMALY_DIR) not in sys.path:
    sys.path.append(str(DINOMALY_DIR))

class DinomalyBaseline(nn.Module):
    """
    Dinomaly Baseline Wrapper for Industrial Anomaly Detection on MVTec LOCO AD.
    Decoupled from original codebase for clean extensibility.
    """
    def __init__(self, embed_dim=768, num_decoder_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_decoder_layers = num_decoder_layers
        
        # Linear Projection / Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 8-layer Decoder Block
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8, # 8 attention heads
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, x):
        """
        Forward pass for patch token reconstruction.
        Args:
            x (torch.Tensor): Feature tokens [B, N, D]
        Returns:
            torch.Tensor: Reconstructed tokens [B, N, D]
        """
        bottleneck_out = self.bottleneck(x)
        reconstructed = self.decoder(bottleneck_out, bottleneck_out)
        return reconstructed

if __name__ == "__main__":
    # Sanity Check Dry Test
    model = DinomalyBaseline(embed_dim=768, num_decoder_layers=8)
    dummy_input = torch.randn(2, 1024, 768)
    out = model(dummy_input)
    print(f"[SUCCESS] DinomalyBaseline Forward Test Passed! Output shape: {out.shape}")
