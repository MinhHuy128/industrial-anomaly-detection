"""
ViTill-GCT model with optional ranking and VLM adapter modules.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder_blocks import bMlp, DecoderBlock, init_weights
from experiments.vlm_ablation.models.vlm_adapter import AdaptiveSigmoidGate, VLMFeatureModulator, VLMSelectiveAdapter


class GCTModule(nn.Module):
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


class ViTillGCTRanking(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        vlm_dim: int = 512,
        num_heads: int = 12,
        num_decoder_layers: int = 8,
        target_layers: Optional[List[int]] = None,
        fuse_layer_encoder: Optional[List[List[int]]] = None,

        fuse_layer_decoder: Optional[List[List[int]]] = None,
        bottleneck_drop: float = 0.2,
        use_gct: bool = True,
        use_adapter: bool = False,
        use_modulation: bool = False,
        init_gamma: float = -4.0
    ):
        super().__init__()
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        self.target_layers = target_layers
        self.fuse_layer_enc = fuse_layer_encoder
        self.fuse_layer_dec = fuse_layer_decoder
        self.embed_dim = embed_dim
        self.use_gct = use_gct
        self.use_adapter = use_adapter
        self.use_modulation = use_modulation

        # bottleneck
        self.bottleneck = bMlp(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            drop=bottleneck_drop
        )

        # gct token module
        self.gct = GCTModule(embed_dim=embed_dim) if use_gct else None

        # optional vlm modulator
        if use_adapter or use_modulation:
            self.vlm_modulator = VLMFeatureModulator(
                vlm_dim=vlm_dim,
                embed_dim=embed_dim,
                init_gamma=init_gamma
            )
        else:
            self.vlm_modulator = None

        # decoder blocks
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            )
            for _ in range(num_decoder_layers)
        ])

        # Weight initialization
        self.bottleneck.apply(init_weights)
        for blk in self.decoder:
            blk.apply(init_weights)

    def get_gate_weight(self) -> float:
        """Returns the current scalar gate value for monitoring."""
        if self.vlm_modulator is not None:
            return self.vlm_modulator.gate.get_gate_weight()
        return 0.0

    def fuse_features(self, feat_list: List[torch.Tensor], idxs: List[int]) -> torch.Tensor:
        """Averages selected layer feature tensors safely."""
        valid_idxs = [i for i in idxs if i < len(feat_list)]
        if not valid_idxs:
            valid_idxs = [len(feat_list) - 1]
        selected = [feat_list[i] for i in valid_idxs]
        return torch.stack(selected, dim=0).mean(dim=0)

    def forward(
        self,
        feat_list: List[torch.Tensor],
        cls_token: Optional[torch.Tensor] = None,
        vlm_feat: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, float]:
        """Forward pass for training reconstruction & validation inference.

        Args:
            feat_list: 8 x [B, 784, 768] patch features from DINOv2.
            cls_token: [B, 768] CLS token from DINOv2.
            vlm_feat: Optional [B, vlm_dim] multi-modal VLM visual embedding.

        Returns:
            en: List of fused encoder feature spatial maps.
            de: List of fused decoder feature spatial maps.
            gct_loss: Scalar GCT self-consistency loss.
            gate_val: Current adaptive gate value.
        """
        # encoder feature fusion
        x_dense = self.fuse_features(feat_list, list(range(len(feat_list))))

        # bottleneck
        x = self.bottleneck(x_dense)

        # optional vlm modulation
        gate_val = 0.0
        if self.use_modulation and self.vlm_modulator is not None and vlm_feat is not None:
            x, gate_val = self.vlm_modulator(x, vlm_feat)

        # prepend token
        if self.use_gct:
            x = self.gct.prepend(x)

        # decoder pass
        de_list = []
        for blk in self.decoder:
            x = blk(x)
            if self.use_gct:
                de_list.append(x[:, 1:, :])
            else:
                de_list.append(x)

        # gct loss
        if self.use_gct and cls_token is not None:
            gct_final = x[:, 0, :]
            if self.use_adapter and self.vlm_modulator is not None and vlm_feat is not None:
                gct_final, _ = self.vlm_modulator(gct_final, vlm_feat)
            gct_loss = self.gct.compute_loss(gct_final, cls_token)
        else:
            gct_loss = torch.tensor(0.0, device=x.device)

        de_list = de_list[::-1]

        # spatial maps
        N = feat_list[0].shape[1]
        side = int(math.sqrt(N))
        B, _, C = feat_list[0].shape

        def to_spatial(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 2, 1).reshape(B, C, side, side).contiguous()

        en = [to_spatial(self.fuse_features(feat_list, idxs)) for idxs in self.fuse_layer_enc]
        de = [to_spatial(self.fuse_features(de_list, idxs)) for idxs in self.fuse_layer_dec]

        return en, de, gct_loss, gate_val

    def compute_anomaly_score_from_maps(
        self,
        en_list: List[torch.Tensor],
        de_list: List[torch.Tensor],
        gct_loss: torch.Tensor,
        gamma: float = 1.0,
        top_ratio: float = 0.01
    ) -> torch.Tensor:
        """Computes sample-level anomaly score from spatial reconstruction maps and GCT signal.

        Args:
            en_list: Spatial encoder feature maps.
            de_list: Spatial decoder feature maps.
            gct_loss: GCT token distance scalar or batch tensor.
            gamma: Weight factor for GCT anomaly score stream.
            top_ratio: Percentage of top anomalous pixels to average for score.

        Returns:
            [B] anomaly score vector.
        """
        B = en_list[0].shape[0]
        diff_maps = []
        for en, de in zip(en_list, de_list):
            en_norm = F.normalize(en, dim=1)
            de_norm = F.normalize(de, dim=1)
            # Cosine distance: 1 - cosine_similarity
            cos_dist = 1.0 - (en_norm * de_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            diff_maps.append(cos_dist)

        avg_diff = torch.stack(diff_maps, dim=0).mean(dim=0)  # [B, 1, H, W]
        flat_diff = avg_diff.view(B, -1)                      # [B, H*W]

        k = max(1, int(flat_diff.shape[1] * top_ratio))
        topk_scores, _ = torch.topk(flat_diff, k=k, dim=1)
        patch_score = topk_scores.mean(dim=1)                 # [B]

        if self.use_gct and gamma > 0.0:
            if isinstance(gct_loss, torch.Tensor) and gct_loss.dim() > 0:
                gct_comp = gct_loss.view(B)
            else:
                gct_comp = gct_loss.expand(B)
            total_score = patch_score + (gamma * gct_comp)
        else:
            total_score = patch_score

        return total_score
