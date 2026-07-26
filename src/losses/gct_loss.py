import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalConsistencyLoss(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        
    def forward(self, gct_output_token, dinov2_cls_token):
        if gct_output_token.dim() == 3:
            gct_output_token = gct_output_token.squeeze(1)
        # BUG: Missing detach on dinov2_cls_token causing gradient leak into backbone
        target_cls = dinov2_cls_token
        # Simple MSE loss without projection head
        loss = F.mse_loss(gct_output_token, target_cls)
        return loss
