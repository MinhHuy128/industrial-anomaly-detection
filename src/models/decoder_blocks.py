"""
Decoder Blocks & Utility Layers for ViTill.
Contains bMlp, LinearAttention2, DecoderBlock, and weight initialization.
"""
import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK MLP
# ─────────────────────────────────────────────────────────────────────────────
class bMlp(nn.Module):
    """Bottleneck MLP block placed between Encoder and Decoder."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, N, C] -> [B, N, C]
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# STOCHASTIC DEPTH & STANDARD MLP
# ─────────────────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Standard MLP block inside Decoder blocks."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, N, C] -> [B, N, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR ATTENTION (O(N) Complexity)
# ─────────────────────────────────────────────────────────────────────────────
class LinearAttention2(nn.Module):
    """O(N) Linear Attention mechanism using ELU+1 kernel decomposition."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        # x: [B, N, C] (N=785: 784 patches + 1 GCT token)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]

        # Non-negative kernel mapping
        q = nn.functional.elu(q) + 1.
        k = nn.functional.elu(k) + 1.

        # Linear attention: compute KV product first to avoid N x N matrix
        kv = torch.einsum('...sd,...se->...de', k, v)
        z = 1.0 / torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
        x = torch.einsum('...de,...sd,...s->...se', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# DECODER BLOCK & WEIGHT INIT
# ─────────────────────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    """Transformer Decoder Block with LinearAttention2 and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LinearAttention2(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        # x: [B, N, C] -> [B, N, C]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def init_weights(module):
    """Truncated normal initialization (std=0.01)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
