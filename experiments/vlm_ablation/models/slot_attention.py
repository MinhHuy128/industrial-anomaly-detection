"""
Iterative slot attention module for component decomposition.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSlotAttention(nn.Module):
    """
    Iterative Slot Attention module for fine-grained industrial component decomposition.
    """
    def __init__(
        self,
        num_slots: int = 8,
        dim: int = 768,
        slot_dim: int = 768,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 1536
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        # Slot initialization parameters (learned Gaussian distribution)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.slots_mu, std=0.02)

        # Projections for Q, K, V
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(dim, slot_dim, bias=False)
        self.to_v = nn.Linear(dim, slot_dim, bias=False)

        # Recurrent update GRU
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Slot feed-forward refinement network
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Patch tokens from backbone [B, N, D] (e.g. [B, 784, 768])
        Returns:
            slots: Physical component slot representations [B, K, slot_dim]
            attn_maps: Normalized spatial slot attention maps [B, K, H, W]
        """
        B, N, D = inputs.shape
        H = W = int(math.sqrt(N))

        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)

            dots = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn = F.softmax(dots, dim=1) + self.eps
            attn_weights = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bkn,bnd->bkd', attn_weights, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            )
            slots = slots.reshape(B, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        attn_maps = attn.reshape(B, self.num_slots, H, W)
        return slots, attn_maps

    def compute_slot_orthogonality_loss(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Slot Orthogonality Loss: Enforces slots to be diverse and distinct from each other.
        L_ortho = || S S^T - I ||_F^2
        """
        B, K, D = slots.shape
        slots_norm = F.normalize(slots, dim=-1)
        sim_matrix = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]
        identity = torch.eye(K, device=slots.device).unsqueeze(0).expand(B, -1, -1)
        loss = F.mse_loss(sim_matrix, identity)
        return loss


# Convenience alias
SlotAttention = SemanticSlotAttention
