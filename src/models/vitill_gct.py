"""
ViTill-GCT: Paper-faithful Dinomaly architecture + Global Consistency Token (GCT).

Architecture (faithful to Dinomaly paper, Section 3):
  - Encoder: DINOv2-Register ViT-B/14 (frozen), multi-layer feature extraction
  - Bottleneck: bMlp(768, 3072, 768, drop=0.2)
  - Decoder: 8 × DecoderBlock with LinearAttention2 (O(N) complexity)
  - GCT Token: Learned token injected after Bottleneck, conditioned on DINOv2 CLS
  - Loss: global_cosine_hm_percent(encoder_feats, decoder_feats) + λ × GCT_cosine_loss

Reference:
  Kang et al., "Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised
  Anomaly Detection", arXiv 2405.14325
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from src.models.decoder_blocks import bMlp, DecoderBlock, init_weights

# ─────────────────────────────────────────────────────────────────────────────
# ENCODER: DINOv2-Register via torch.hub
# ─────────────────────────────────────────────────────────────────────────────
def load_dinov2_register(device: torch.device) -> nn.Module:
    """
    Load DINOv2-Register ViT-B/14 (4 register tokens) — paper default encoder.
    Uses torch.hub which downloads from Meta's public release.
    """
    print("[BACKBONE] Loading DINOv2-Register ViT-B/14 (dinov2_vitb14_reg)...")
    backbone = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitb14_reg'
    ).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print("[BACKBONE] DINOv2-Register loaded and frozen.")
    return backbone


def extract_intermediate_features(
    backbone: nn.Module,
    x: torch.Tensor,
    target_layers: list,   # e.g. [2,3,4,5,6,7,8,9]
    return_cls: bool = True
):
    """
    Extract intermediate patch token features from DINOv2 backbone.
    Uses get_intermediate_layers() — official DINOv2 API.

    Returns:
        feat_list: list of [B, N_patches, 768] tensors (one per target layer)
        cls_token:  [B, 768] CLS token from LAST target layer
    """
    outputs = backbone.get_intermediate_layers(
        x, n=target_layers, return_class_token=return_cls
    )
    if return_cls:
        # outputs is list of (patch_tokens, cls_token) tuples
        feat_list = [o[0] for o in outputs]   # [B, N, C] each
        cls_token  = outputs[-1][1]            # [B, C] from last layer
    else:
        feat_list = outputs
        cls_token  = None
    return feat_list, cls_token


# ─────────────────────────────────────────────────────────────────────────────
# GCT MODULE (Global Consistency Token)
# ─────────────────────────────────────────────────────────────────────────────
class GCTModule(nn.Module):
    """
    Global Consistency Token (GCT):
      - A learnable token injected into the token sequence after Bottleneck.
      - Passes through all 8 Decoder Blocks (attends to all patch tokens).
      - AFTER decoder, the final GCT output is supervised via Cosine Distance
        Loss against the frozen DINOv2 CLS token.

    Design rationale:
      Supervision on the FINAL decoder output (not the initial parameter) forces
      the decoder to aggregate global semantic context through the attention
      mechanism — this is the key mechanism enabling logical anomaly detection.

    Loss formula (computed in ViTillGCT.forward after decoder):
      gct_final = decoder_output[:, 0, :]   # GCT token position
      L_gct = 1 - cosine_similarity(proj(gct_final), cls_token.detach())
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        # Learnable GCT token (initialized small, same as Dinomaly weight init)
        self.gct_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.gct_token, std=0.01)

        # 2-layer MLP projection head (align GCT output with DINOv2 CLS space)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def prepend(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend GCT token to token sequence. [B,N,C] → [B,N+1,C]"""
        B = x.shape[0]
        gct = self.gct_token.expand(B, -1, -1)  # [B, 1, C]
        return torch.cat([gct, x], dim=1)        # [B, N+1, C]

    def compute_loss(self, gct_final: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Compute GCT loss on the FINAL decoder output of the GCT position.

        Args:
            gct_final: [B, C] — GCT token output from LAST decoder block
            cls_token: [B, C] — DINOv2 CLS token (detach() is applied here)
        Returns:
            scalar loss
        """
        proj_gct = self.projection_head(gct_final)         # [B, C]
        cls_detached = cls_token.detach()                  # NO gradient to backbone!
        return (1.0 - F.cosine_similarity(proj_gct, cls_detached, dim=-1)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# VITILL-GCT: Full Model
# ─────────────────────────────────────────────────────────────────────────────
class ViTillGCT(nn.Module):
    """
    Paper-faithful Dinomaly + Global Consistency Token.

    Encoder: DINOv2-Register ViT-B/14 (frozen, 8 intermediate layers)
    Bottleneck: bMlp
    GCT: 1 learnable token + projection head
    Decoder: 8 × LinearAttention2 DecoderBlocks
    """
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
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]   # 8 layers, paper default
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

        self.target_layers   = target_layers
        self.fuse_layer_enc  = fuse_layer_encoder
        self.fuse_layer_dec  = fuse_layer_decoder
        self.gct_lambda      = gct_lambda
        self.embed_dim       = embed_dim

        # Bottleneck: single bMlp layer
        self.bottleneck = bMlp(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            drop=bottleneck_drop
        )

        # GCT Module
        self.gct = GCTModule(embed_dim=embed_dim)

        # Decoder: 8 LinearAttention2 blocks
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

        # Initialize weights (paper: trunc_normal std=0.01)
        self.bottleneck.apply(init_weights)
        for blk in self.decoder:
            blk.apply(init_weights)

    def fuse_features(self, feat_list: list, idxs: list) -> torch.Tensor:
        """Average-fuse features from specified layer indices."""
        selected = [feat_list[i] for i in idxs]
        return torch.stack(selected, dim=0).mean(dim=0)  # [B, N, C]

    def forward(self, feat_list: list, cls_token: torch.Tensor):
        """
        Args:
            feat_list: list of [B, N, C] tensors (8 intermediate encoder layers)
            cls_token: [B, C] DINOv2 CLS token (from backbone, will be detached in gct.compute_loss)

        Returns:
            en:       list of [B, C, H, W] fused encoder feature maps
            de:       list of [B, C, H, W] fused decoder feature maps
            gct_loss: scalar GCT cosine distance loss (on FINAL decoder GCT output)
        """
        # Fuse encoder features into single representation
        x = self.fuse_features(feat_list, list(range(len(feat_list))))  # [B, N, C]

        # Bottleneck
        x = self.bottleneck(x)  # [B, N, C]

        # Inject GCT token (prepended at position 0)
        x = self.gct.prepend(x)  # [B, N+1, C]

        # Decode through 8 LinearAttention blocks, collect all outputs
        de_list = []
        for blk in self.decoder:
            x = blk(x)
            # Strip GCT token (position 0) before collecting patch decoder features
            de_list.append(x[:, 1:, :])  # [B, N, C]

        # ✅ GCT loss on FINAL decoder output — x[:, 0, :] is the GCT token
        # after attending to ALL patch tokens through all 8 decoder blocks.
        # This is the correct supervision point: the decoder has aggregated
        # global context into the GCT token, and we supervise it against CLS.
        gct_final = x[:, 0, :]                            # [B, C]
        gct_loss  = self.gct.compute_loss(gct_final, cls_token)

        de_list = de_list[::-1]  # Reverse for layer ordering (paper convention)

        # Spatial reshape helper
        N = feat_list[0].shape[1]
        side = int(math.sqrt(N))
        B, _, C = feat_list[0].shape

        def to_spatial(t):  # [B, N, C] → [B, C, H, W]
            return t.permute(0, 2, 1).reshape(B, C, side, side).contiguous()

        # Fuse encoder and decoder feature groups
        en = [to_spatial(self.fuse_features(feat_list, idxs))
              for idxs in self.fuse_layer_enc]
        de = [to_spatial(self.fuse_features(de_list, idxs))
              for idxs in self.fuse_layer_dec]

        return en, de, gct_loss


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE (no GCT) — for fair comparison
# ─────────────────────────────────────────────────────────────────────────────
class ViTillBaseline(nn.Module):
    """
    Paper-faithful Dinomaly Baseline (no GCT).
    Identical architecture to ViTillGCT but without GCT token.
    """
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
        x = self.fuse_features(feat_list, list(range(len(feat_list))))
        x = self.bottleneck(x)
        de_list = []
        for blk in self.decoder:
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        N = feat_list[0].shape[1]
        side = int(math.sqrt(N))
        B, _, C = feat_list[0].shape

        def to_spatial(t):
            return t.permute(0, 2, 1).reshape(B, C, side, side).contiguous()

        en = [to_spatial(self.fuse_features(feat_list, idxs))
              for idxs in self.fuse_layer_enc]
        de = [to_spatial(self.fuse_features(de_list, idxs))
              for idxs in self.fuse_layer_dec]
        return en, de
