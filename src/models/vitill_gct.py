"""
ViTill-GCT Model Architecture.
Integrates frozen DINOv2-Register backbone, Bottleneck MLP, GCT Module, and 8-layer Decoder.
"""
import math
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder_blocks import bMlp, DecoderBlock, init_weights


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE ENCODER (DINOv2-Register)
# ─────────────────────────────────────────────────────────────────────────────
def load_dinov2_register(device: torch.device) -> nn.Module:
    """Load and freeze DINOv2-Register ViT-B/14 backbone."""
    from pathlib import Path
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"

    if hub_dir.exists():
        print(f"[BACKBONE] Loading DINOv2-Register from local cache ({hub_dir.name})...")
        backbone = torch.hub.load(str(hub_dir), 'dinov2_vitb14_reg', source='local').to(device)
    else:
        print("[BACKBONE] Downloading DINOv2-Register ViT-B/14 from Meta PyTorch Hub...")
        import time
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', source='github').to(device)
                break
            except Exception as e:
                if attempt < max_retries:
                    print(f"[WARN] Backbone download attempt {attempt}/{max_retries} failed ({e}). Retrying...")
                    time.sleep(2)
                else:
                    raise RuntimeError(f"[ERROR] Failed to load DINOv2 backbone: {e}") from e

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print("[BACKBONE] DINOv2-Register loaded and frozen.")
    return backbone


def extract_intermediate_features(
    backbone: nn.Module,
    x: torch.Tensor,        # [B, 3, 392, 392]
    target_layers: list,   # [2, 3, 4, 5, 6, 7, 8, 9]
    return_cls: bool = True
):
    """
    Extract intermediate patch features and CLS token from DINOv2.
    Returns:
        feat_list: list of 8 x [B, 784, 768] patch feature tensors
        cls_token:  [B, 768] global CLS token from the last target layer
    """
    outputs = backbone.get_intermediate_layers(
        x, n=target_layers, return_class_token=return_cls
    )
    if return_cls:
        feat_list = [o[0] for o in outputs]  # each: [B, 784, 768]
        cls_token = outputs[-1][1]           # [B, 768]
    else:
        feat_list = outputs
        cls_token = None
    return feat_list, cls_token


# ─────────────────────────────────────────────────────────────────────────────
# GCT MODULE (Global Consistency Token)
# ─────────────────────────────────────────────────────────────────────────────
class GCTModule(nn.Module):
    """
    Global Consistency Token module:
    - Prepends a learnable GCT token [1, 1, C] to patch tokens before Decoder.
    - Passes projected GCT token output through 1-layer Linear + LayerNorm head.
    - Computes Cosine Distance loss against frozen DINOv2 CLS token.
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        # Learnable GCT token: [1, 1, 768]
        self.gct_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.gct_token, std=0.01)

        # Projection head: 1-layer Linear + LayerNorm
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def prepend(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 784, 768] -> returns [B, 785, 768]
        B = x.shape[0]
        gct = self.gct_token.expand(B, -1, -1)  # [B, 1, 768]
        return torch.cat([gct, x], dim=1)        # [B, 785, 768]

    def compute_loss(self, gct_final: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        # gct_final: [B, 768], cls_token: [B, 768]
        proj_gct = self.projection_head(gct_final)         # [B, 768]
        cls_detached = cls_token.detach()                  # Freeze DINOv2 CLS gradient
        return (1.0 - F.cosine_similarity(proj_gct, cls_detached, dim=-1)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# VITILL-GCT FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────
class ViTillGCT(nn.Module):
    """ViTill-GCT model: DINOv2 + Bottleneck + GCT + 8-layer Decoder."""
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_decoder_layers: int = 8,
        target_layers: list = None,
        fuse_layer_encoder: list = None,
        fuse_layer_decoder: list = None,
        gct_lambda: float = 0.1,
        bottleneck_drop: float = 0.2,
    ):
        super().__init__()
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        self.target_layers  = target_layers
        self.fuse_layer_enc = fuse_layer_encoder
        self.fuse_layer_dec = fuse_layer_decoder
        self.gct_lambda     = gct_lambda
        self.embed_dim      = embed_dim

        self.bottleneck = bMlp(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            drop=bottleneck_drop
        )

        self.gct = GCTModule(embed_dim=embed_dim)

        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8)
            )
            for _ in range(num_decoder_layers)
        ])

        self.bottleneck.apply(init_weights)
        for blk in self.decoder:
            blk.apply(init_weights)

    def fuse_features(self, feat_list: list, idxs: list) -> torch.Tensor:
        # Fuse specified encoder/decoder feature layers: [B, N, C]
        selected = [feat_list[i] for i in idxs]
        return torch.stack(selected, dim=0).mean(dim=0)

    def forward(self, feat_list: list, cls_token: torch.Tensor):
        # feat_list: 8 x [B, 784, 768], cls_token: [B, 768]
        # 1. Average fuse 8 encoder layers
        x = self.fuse_features(feat_list, list(range(len(feat_list))))  # [B, 784, 768]

        # 2. Bottleneck MLP
        x = self.bottleneck(x)  # [B, 784, 768]

        # 3. Prepend GCT token at index 0
        x = self.gct.prepend(x)  # [B, 785, 768]

        # 4. Decode through 8 blocks, collect patch tokens
        de_list = []
        for blk in self.decoder:
            x = blk(x)                   # [B, 785, 768]
            de_list.append(x[:, 1:, :])  # [B, 784, 768] patch tokens only

        # 5. GCT loss on final decoded GCT token (index 0)
        gct_final = x[:, 0, :]                            # [B, 768]
        gct_loss  = self.gct.compute_loss(gct_final, cls_token)

        de_list = de_list[::-1]  # Reverse layer order

        # 6. Spatial reshape: [B, N, C] -> [B, C, 28, 28]
        N = feat_list[0].shape[1]
        side = int(math.sqrt(N))
        B, _, C = feat_list[0].shape

        def to_spatial(t):
            return t.permute(0, 2, 1).reshape(B, C, side, side).contiguous()

        en = [to_spatial(self.fuse_features(feat_list, idxs)) for idxs in self.fuse_layer_enc]
        de = [to_spatial(self.fuse_features(de_list, idxs))   for idxs in self.fuse_layer_dec]

        return en, de, gct_loss


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE MODEL (No GCT)
# ─────────────────────────────────────────────────────────────────────────────
class ViTillBaseline(nn.Module):
    """Baseline model without GCT module for comparative benchmark."""
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_decoder_layers: int = 8,
        target_layers: list = None,
        fuse_layer_encoder: list = None,
        fuse_layer_decoder: list = None,
        bottleneck_drop: float = 0.2,
    ):
        super().__init__()
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        self.target_layers  = target_layers
        self.fuse_layer_enc = fuse_layer_encoder
        self.fuse_layer_dec = fuse_layer_decoder

        self.bottleneck = bMlp(embed_dim, embed_dim * 4, embed_dim, drop=bottleneck_drop)
        self.decoder = nn.ModuleList([
            DecoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(num_decoder_layers)
        ])
        self.bottleneck.apply(init_weights)
        for blk in self.decoder:
            blk.apply(init_weights)

    def fuse_features(self, feat_list, idxs):
        selected = [feat_list[i] for i in idxs]
        return torch.stack(selected, dim=0).mean(dim=0)

    def forward(self, feat_list: list):
        # feat_list: 8 x [B, 784, 768]
        x = self.fuse_features(feat_list, list(range(len(feat_list))))  # [B, 784, 768]
        x = self.bottleneck(x)                                          # [B, 784, 768]
        de_list = []
        for blk in self.decoder:
            x = blk(x)          # [B, 784, 768]
            de_list.append(x)
        de_list = de_list[::-1]

        N = feat_list[0].shape[1]
        side = int(math.sqrt(N))
        B, _, C = feat_list[0].shape

        def to_spatial(t):
            return t.permute(0, 2, 1).reshape(B, C, side, side).contiguous()

        en = [to_spatial(self.fuse_features(feat_list, idxs)) for idxs in self.fuse_layer_enc]
        de = [to_spatial(self.fuse_features(de_list, idxs))   for idxs in self.fuse_layer_dec]
        return en, de
