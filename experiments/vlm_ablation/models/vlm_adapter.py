"""
VLM adapter and gated modulation modules.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSigmoidGate(nn.Module):
    def __init__(self, init_gamma: float = -4.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor([init_gamma], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma)

    def get_gate_weight(self) -> float:
        return float(torch.sigmoid(self.gamma).item())


class VLMSelectiveAdapter(nn.Module):
    def __init__(
        self,
        vlm_dim: int = 512,
        embed_dim: int = 768,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.net = nn.Sequential(
            nn.Linear(vlm_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, vlm_feat: torch.Tensor) -> torch.Tensor:
        dtype = next(self.net.parameters()).dtype
        return self.net(vlm_feat.to(dtype=dtype))


class VLMFeatureModulator(nn.Module):
    def __init__(
        self,
        vlm_dim: int = 512,
        embed_dim: int = 768,
        init_gamma: float = -4.0
    ):
        super().__init__()
        self.adapter = VLMSelectiveAdapter(vlm_dim=vlm_dim, embed_dim=embed_dim)
        self.gate = AdaptiveSigmoidGate(init_gamma=init_gamma)

    def forward(
        self,
        x: torch.Tensor,
        vlm_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        gate_w = self.gate()
        gate_val = float(gate_w.item())

        if vlm_feat is None:
            return x, gate_val

        dtype = x.dtype
        proj_vlm = self.adapter(vlm_feat).to(dtype=dtype)
        if x.dim() == 3 and proj_vlm.dim() == 2:
            proj_vlm = proj_vlm.unsqueeze(1)

        modulated_x = x + (gate_w.to(dtype=dtype) * proj_vlm)
        return modulated_x, gate_val

