import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalConsistencyLoss(nn.Module):
    """
    Global Consistency Loss for GCT (Global Consistency Token).
    Calculates Cosine Distance between Projected GCT Token output and DINOv2 CLS token.
    Enforces DINOv2 CLS token detachment to prevent gradient drift.
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        # 2-layer MLP Projection Head for vector space alignment
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, gct_output_token, dinov2_cls_token):
        """
        Args:
            gct_output_token (torch.Tensor): Output GCT token from Decoder [B, 1, D] or [B, D]
            dinov2_cls_token (torch.Tensor): CLS token from frozen DINOv2 encoder [B, D]
        Returns:
            torch.Tensor: Cosine distance loss scalar
        """
        if gct_output_token.dim() == 3:
            gct_output_token = gct_output_token.squeeze(1)
            
        # Crucial rule: Freeze DINOv2 CLS token to prevent feature corruption
        target_cls = dinov2_cls_token.detach()
        
        # Project GCT token through 2-layer MLP Projection Head
        projected_gct = self.projection_head(gct_output_token)
        
        # Calculate Cosine Similarity Loss (1 - cosine_similarity)
        cosine_sim = F.cosine_similarity(projected_gct, target_cls, dim=-1)
        loss = torch.mean(1.0 - cosine_sim) # Minimize cosine distance
        
        return loss

if __name__ == "__main__":
    # Sanity Check Dry Test
    loss_fn = GlobalConsistencyLoss(embed_dim=768)
    dummy_gct = torch.randn(2, 1, 768)
    dummy_cls = torch.randn(2, 768)
    l = loss_fn(dummy_gct, dummy_cls)
    print(f"[SUCCESS] GlobalConsistencyLoss Forward Test Passed! Loss: {l.item():.4f}")
  
  
  
  
