"""
Dinomaly GCT Wrapper Module (Standalone Prototype).
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from src.losses.gct_loss import GlobalConsistencyLoss


# ─────────────────────────────────────────────────────────────────────────────
# GCT WRAPPER (Standalone Prototype)
# ─────────────────────────────────────────────────────────────────────────────
class DinomalyGCT(nn.Module):
    """Dinomaly + GCT prototype model using PyTorch TransformerDecoder."""
    def __init__(self, embed_dim=768, num_decoder_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_decoder_layers = num_decoder_layers
        
        # GCT token parameter: [1, 1, 768]
        self.gct_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
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
        
        self.gct_loss_fn = GlobalConsistencyLoss(embed_dim=embed_dim)

    def forward(self, patch_tokens, dinov2_cls_token=None):
        # patch_tokens: [B, N, 768], dinov2_cls_token: [B, 768]
        B, N, D = patch_tokens.shape
        
        # Prepend GCT token -> [B, 1 + N, 768]
        gct_expanded = self.gct_token.expand(B, -1, -1)
        input_sequence = torch.cat([gct_expanded, patch_tokens], dim=1)
        
        bottleneck_out = self.bottleneck(input_sequence)
        decoder_out = self.decoder(bottleneck_out, bottleneck_out)
        
        # Separate GCT token and patch tokens
        gct_output = decoder_out[:, 0:1, :]           # [B, 1, 768]
        reconstructed_patches = decoder_out[:, 1:, :]  # [B, N, 768]
        
        rec_loss = F.mse_loss(reconstructed_patches, patch_tokens)
        
        loss_gct = torch.tensor(0.0, device=patch_tokens.device)
        if dinov2_cls_token is not None:
            loss_gct = self.gct_loss_fn(gct_output, dinov2_cls_token)
            
        total_loss = rec_loss + 0.5 * loss_gct
            
        return {
            "reconstructed_patches": reconstructed_patches,
            "gct_output": gct_output,
            "rec_loss": rec_loss,
            "loss_gct": loss_gct,
            "total_loss": total_loss
        }


if __name__ == "__main__":
    model = DinomalyGCT(embed_dim=768, num_decoder_layers=8)
    dummy_patches = torch.randn(2, 1024, 768)
    dummy_cls = torch.randn(2, 768)
    
    output = model(dummy_patches, dinov2_cls_token=dummy_cls)
    print(f"[SUCCESS] DinomalyGCT forward test passed! Total loss: {output['total_loss'].item():.4f}")
