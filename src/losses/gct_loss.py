"""
Global Consistency Loss module (standalone prototype version).
Calculates Cosine Distance between projected GCT token and DINOv2 CLS token.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSISTENCY LOSS (Standalone Prototype)
# ─────────────────────────────────────────────────────────────────────────────
class GlobalConsistencyLoss(nn.Module):
    """Standalone GCT loss function with 2-layer MLP projection head."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, gct_output_token, dinov2_cls_token):
        # gct_output_token: [B, 1, 768] or [B, 768], dinov2_cls_token: [B, 768]
        if gct_output_token.dim() == 3:
            gct_output_token = gct_output_token.squeeze(1)  # [B, 768]
            
        target_cls = dinov2_cls_token.detach()  # Freeze DINOv2 CLS gradient
        projected_gct = self.projection_head(gct_output_token)  # [B, 768]
        
        cosine_sim = F.cosine_similarity(projected_gct, target_cls, dim=-1)
        loss = torch.mean(1.0 - cosine_sim)
        return loss


if __name__ == "__main__":
    loss_fn = GlobalConsistencyLoss(embed_dim=768)
    dummy_gct = torch.randn(2, 1, 768)
    dummy_cls = torch.randn(2, 768)
    l = loss_fn(dummy_gct, dummy_cls)
    print(f"[SUCCESS] GlobalConsistencyLoss forward test passed: {l.item():.4f}")
