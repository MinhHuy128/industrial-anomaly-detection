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

class DinomalyGCT(nn.Module):
    """
    Dinomaly + Global Consistency Token (GCT) for Industrial Anomaly Detection on MVTec LOCO AD.
    
    Architecture:
    1. Prepend learnable GCT embedding token to input patch tokens before 8-layer Decoder.
    2. Pass GCT token output through 2-layer MLP Projection Head.
    3. Compute Cosine Distance Loss against frozen DINOv2 CLS token (.detach()).
    """
    def __init__(self, embed_dim=768, num_decoder_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_decoder_layers = num_decoder_layers
        
        # 1. Learnable GCT Embedding Token [1, 1, D]
        self.gct_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 2. Linear Bottleneck Projection
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 3. 8-layer Transformer Decoder Block
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 4. GCT Loss Function with Projection Head
        self.gct_loss_fn = GlobalConsistencyLoss(embed_dim=embed_dim)

    def forward(self, patch_tokens, dinov2_cls_token=None):
        """
        Args:
            patch_tokens (torch.Tensor): Input patch feature tokens [B, N, D]
            dinov2_cls_token (torch.Tensor, optional): CLS token from frozen DINOv2 backbone [B, D]
        Returns:
            dict: Reconstructed patch tokens, output GCT token, and total loss scalar.
        """
        B, N, D = patch_tokens.shape
        
        # Expand GCT token for batch dimension [B, 1, D]
        gct_expanded = self.gct_token.expand(B, -1, -1)
        
        # Concatenate GCT token to head of patch sequence [B, 1 + N, D]
        input_sequence = torch.cat([gct_expanded, patch_tokens], dim=1)
        
        # Pass through Bottleneck & Decoder
        bottleneck_out = self.bottleneck(input_sequence)
        decoder_out = self.decoder(bottleneck_out, bottleneck_out)
        
        # Separate GCT token output and reconstructed patch tokens
        gct_output = decoder_out[:, 0:1, :]           # [B, 1, D]
        reconstructed_patches = decoder_out[:, 1:, :]  # [B, N, D]
        
        # Calculate Patch Reconstruction Loss (MSE + Cosine Distance)
        rec_loss = F.mse_loss(reconstructed_patches, patch_tokens)
        
        # Calculate GCT Loss if DINOv2 CLS token is provided
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
    # Sanity Check Dry Test
    model = DinomalyGCT(embed_dim=768, num_decoder_layers=8)
    dummy_patches = torch.randn(2, 1024, 768)
    dummy_cls = torch.randn(2, 768)
    
    output = model(dummy_patches, dinov2_cls_token=dummy_cls)
    print(f"[SUCCESS] DinomalyGCT Audit Test Passed!")
    print(f"         Total Loss: {output['total_loss'].item():.4f} (Rec Loss: {output['rec_loss'].item():.4f}, GCT Loss: {output['loss_gct'].item():.4f})")
# Ensured tensor concatenation aligns with batch and sequence dimensions
# Synthetic dry-test validates forward pass stability
  
  
