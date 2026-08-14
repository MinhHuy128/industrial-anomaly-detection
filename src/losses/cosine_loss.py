"""
Loss functions for ViTill-GCT.
Implements patch reconstruction loss with hard patch mining and combined training loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT MASKING & HARD PATCH MINING
# ─────────────────────────────────────────────────────────────────────────────
def modify_grad(x, inds, factor=0.):
    """Zero-out or scale gradients for easy background patches."""
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine_hm_percent(
    en_list: list,
    de_list: list,
    p: float = 0.9,
    factor: float = 0.
) -> torch.Tensor:
    """
    Patch reconstruction loss with hard patch mining:
    1. Compute per-patch Cosine Distance d_i = 1 - cos_sim(en_i, de_i).
    2. Find p-th percentile threshold t across all spatial patches.
    3. Retain gradient for top (1-p)% hardest patches; scale down easy patches.
    """
    cos_loss = nn.CosineSimilarity(dim=1)
    loss = torch.tensor(0., device=en_list[0].device)
    hook_handles = []

    for en, de in zip(en_list, de_list):
        # en, de: [B, C, 28, 28]
        en_ = en.detach()  # Freeze encoder gradient
        de_ = de

        with torch.no_grad():
            point_dist = (1 - cos_loss(en_, de_)).unsqueeze(1)  # [B, 1, 28, 28]
            point_dist_flat = point_dist.reshape(-1)

            # Top (1-p)% threshold
            k = max(1, int(point_dist_flat.numel() * (1 - p)))
            thresh = torch.topk(point_dist_flat, k=k)[0][-1]

        en_flat = en_.reshape(en_.shape[0], en_.shape[1], -1)   # [B, C, 784]
        de_flat = de_.reshape(de_.shape[0], de_.shape[1], -1)   # [B, C, 784]
        loss = loss + torch.mean(1 - cos_loss(en_flat, de_flat))

        # Register backward hook to zero gradients on easy patches
        easy_mask = (point_dist < thresh)
        hook_fn = partial(modify_grad, inds=easy_mask, factor=factor)
        handle = de_.register_hook(hook_fn)
        hook_handles.append(handle)

    loss = loss / len(en_list)

    # Cleanup backward hooks after backward pass
    def _cleanup_hooks(grad):
        for h in hook_handles:
            h.remove()
    loss.register_hook(_cleanup_hooks)

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# GCT COSINE LOSS & COMBINED LOSS
# ─────────────────────────────────────────────────────────────────────────────
def gct_cosine_loss(proj_gct: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
    """Cosine Distance between projected GCT token and frozen DINOv2 CLS token."""
    # proj_gct: [B, 768], cls_token: [B, 768]
    return (1.0 - F.cosine_similarity(proj_gct, cls_token.detach(), dim=-1)).mean()


def combined_loss(
    en_list: list,
    de_list: list,
    gct_loss: torch.Tensor,
    p: float = 0.9,
    factor: float = 0.1,
    gct_lambda: float = 0.1,
) -> torch.Tensor:
    """Total loss: L_total = L_reconstruction + lambda * L_gct."""
    l_rec = global_cosine_hm_percent(en_list, de_list, p=p, factor=factor)
    return l_rec + gct_lambda * gct_loss
