"""
ViTill-GCT with VLM distillation head.
"""

import math
import sys
from pathlib import Path
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder_blocks import bMlp, DecoderBlock, init_weights


class GCTModuleVLM(nn.Module):
    def __init__(self, embed_dim: int = 768, vlm_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self.vlm_dim = vlm_dim

        # learnable token
        self.gct_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.gct_token, std=0.01)

        # cls projection
        self.projection_cls = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # vlm teacher projection
        self.projection_vlm = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vlm_dim),
            nn.LayerNorm(vlm_dim)
        )

    def prepend(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        gct = self.gct_token.expand(B, -1, -1)
        return torch.cat([gct, x], dim=1)

    def compute_cls_loss(self, gct_final: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        proj_cls = self.projection_cls(gct_final)
        cls_detached = cls_token.detach()
        return (1.0 - F.cosine_similarity(proj_cls, cls_detached, dim=-1)).mean()

    def compute_vlm_loss(self, gct_final: torch.Tensor, vlm_target: torch.Tensor) -> torch.Tensor:
        proj_vlm = self.projection_vlm(gct_final)
        vlm_detached = F.normalize(vlm_target.detach(), dim=-1)
        proj_normalized = F.normalize(proj_vlm, dim=-1)
        return (1.0 - torch.sum(proj_normalized * vlm_detached, dim=-1)).mean()


class ViTillGCTVLM(nn.Module):
    """
    ViTill-GCT with VLM Knowledge Distillation.
    Decoder decodes patch tokens for micro-structural reconstruction,
    while GCT token simultaneously absorbs DINOv2 CLS and VLM Teacher semantic representations.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        vlm_dim: int = 768,
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
        self.embed_dim      = embed_dim

        # Bottleneck MLP
        self.bottleneck = bMlp(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            drop=bottleneck_drop
        )

        # Dual-Head GCT Module
        self.gct = GCTModuleVLM(embed_dim=embed_dim, vlm_dim=vlm_dim)

        # 8-Layer Transformer Decoder
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
        selected = [feat_list[i] for i in idxs]
        return torch.stack(selected, dim=0).mean(dim=0)

    def forward(
        self,
        feat_list: list,
        cls_token: torch.Tensor,
        vlm_target: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            feat_list: 8 x [B, 784, 768] patch features from DINOv2 encoder layers
            cls_token: [B, 768] DINOv2 CLS token
            vlm_target: Optional [B, vlm_dim] continuous embedding from VLM Teacher (Train only)
        Returns:
            en: fused encoder features in spatial shape [[B, C, 28, 28], [B, C, 28, 28]]
            de: fused decoder features in spatial shape [[B, C, 28, 28], [B, C, 28, 28]]
            gct_cls_loss: scalar loss against DINOv2 CLS
            gct_vlm_loss: scalar loss against VLM Teacher (0 if vlm_target is None)
            gct_final: final decoded GCT token [B, 768]
        """
        # average fuse encoder layers
        x = self.fuse_features(feat_list, list(range(len(feat_list))))

        # bottleneck projection
        x = self.bottleneck(x)

        # prepend token
        x = self.gct.prepend(x)

        # decode through blocks
        de_list = []
        for blk in self.decoder:
            x = blk(x)
            de_list.append(x[:, 1:, :])

        # final gct token
        gct_final = x[:, 0, :]
        gct_cls_loss = self.gct.compute_cls_loss(gct_final, cls_token)

        # teacher loss if provided
        if vlm_target is not None:
            gct_vlm_loss = self.gct.compute_vlm_loss(gct_final, vlm_target)
        else:
            gct_vlm_loss = torch.tensor(0.0, device=x.device)

        de_list = de_list[::-1]

        # spatial reshape
        N = feat_list[0].shape[1]
        side = int(math.sqrt(N))
        B, _, C = feat_list[0].shape

        def to_spatial(t):
            return t.permute(0, 2, 1).reshape(B, C, side, side).contiguous()

        en = [to_spatial(self.fuse_features(feat_list, idxs)) for idxs in self.fuse_layer_enc]
        de = [to_spatial(self.fuse_features(de_list, idxs))   for idxs in self.fuse_layer_dec]

        return en, de, gct_cls_loss, gct_vlm_loss, gct_final
