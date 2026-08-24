"""
Compositional visual-language model for anomaly detection.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder_blocks import bMlp, DecoderBlock, init_weights
from experiments.vlm_ablation.models.slot_attention import SemanticSlotAttention
from experiments.vlm_ablation.models.vlm_reasoner import VLMCompositionalReasoner, CATEGORY_SPECIFICATIONS


class GCTModule(nn.Module):
    """Global Consistency Token module."""
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.gct_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.gct_token, std=0.01)

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def prepend(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        gct = self.gct_token.expand(B, -1, -1)
        return torch.cat([gct, x], dim=1)

    def compute_loss(self, gct_final: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        proj_gct = self.projection_head(gct_final)
        cls_detached = cls_token.detach()
        return (1.0 - F.cosine_similarity(proj_gct, cls_detached, dim=-1)).mean()


class CompVLMAD(nn.Module):
    """
    Comp-VLM-AD: Compositional Visual-Language Reasoning with Semantic Slot Attention.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_slots: int = 8,
        vlm_dim: int = 512,
        num_heads: int = 12,
        num_decoder_layers: int = 8,
        target_layers: list = None,
        fuse_layer_encoder: list = None,
        fuse_layer_decoder: list = None,
        bottleneck_drop: float = 0.2
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
        self.num_slots      = num_slots

        self.slot_attention = SemanticSlotAttention(
            num_slots=num_slots,
            dim=embed_dim,
            slot_dim=embed_dim,
            iters=3
        )

        self.vlm_reasoner = VLMCompositionalReasoner(
            slot_dim=embed_dim,
            vlm_dim=vlm_dim
        )

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
                norm_layer=nn.LayerNorm
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
        text_embeddings: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            feat_list: 8 x [B, 784, 768] patch tokens from DINOv2
            cls_token: [B, 768] DINOv2 CLS token
            text_embeddings: [M, vlm_dim] multi-modal text query embeddings
        """
        x_dense = self.fuse_features(feat_list, list(range(len(feat_list))))

        slots, attn_maps = self.slot_attention(x_dense)
        slot_ortho_loss = self.slot_attention.compute_slot_orthogonality_loss(slots)

        vlm_logical_score, sim_matrix, vlm_coverage_loss = self.vlm_reasoner(slots, text_embeddings)

        x = self.bottleneck(x_dense)
        x = self.gct.prepend(x)

        de_list = []
        for blk in self.decoder:
            x = blk(x)
            de_list.append(x[:, 1:, :])

        gct_final = x[:, 0, :]
        gct_loss = self.gct.compute_loss(gct_final, cls_token)

        de_list = de_list[::-1]
        N = feat_list[0].shape[1]
        side = int(math.sqrt(N))
        B, _, C = feat_list[0].shape

        def to_spatial(t):
            return t.permute(0, 2, 1).reshape(B, C, side, side).contiguous()

        en = [to_spatial(self.fuse_features(feat_list, idxs)) for idxs in self.fuse_layer_enc]
        de = [to_spatial(self.fuse_features(de_list, idxs))   for idxs in self.fuse_layer_dec]

        return en, de, gct_loss, slot_ortho_loss, vlm_coverage_loss, vlm_logical_score, gct_final, attn_maps
