"""
Loss functions for ViTill-GCT training.

Implements:
  1. global_cosine_hm_percent — paper's main reconstruction loss (Eq.4-6)
     Hard-mining variant: top-p% hardest patches get full gradient;
     bottom (1-p)% easy patches have gradients zeroed.

  2. GCT Cosine Loss — Cosine Distance between GCT projection and DINOv2 CLS token

Reference:
  Kang et al., "Dinomaly: The Less Is More Philosophy...", arXiv 2405.14325
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def modify_grad(x, inds, factor=0.):
    """Selectively zero-out gradients for easy patches (gradient masking)."""
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
    Paper's main reconstruction loss (Eq.4-6, Dinomaly):

    For each feature group:
      1. Compute per-patch cosine distance: d_i = 1 - cos_sim(en_i, de_i)
      2. Find threshold t = p-th percentile of all d_i values
      3. Patches with d_i >= t (hardest top-10%) → full gradient
         Patches with d_i <  t (easy bottom-90%) → gradient × factor (≈0)
      4. Loss = mean cosine distance across all positions

    This "hard patch mining" forces the decoder to focus on the most
    anomalous/uncertain regions during reconstruction.

    Args:
        en_list: list of [B, C, H, W] encoder feature maps
        de_list: list of [B, C, H, W] decoder feature maps
        p:       fraction of EASY patches to suppress (default: 0.9)
        factor:  gradient scale for easy patches (default: 0.0 → fully suppressed)

    Returns:
        loss: scalar tensor
    """
    cos_loss = nn.CosineSimilarity(dim=1)
    loss = torch.tensor(0., device=en_list[0].device)
    hook_handles = []  # Track handles to remove hooks after backward

    for en, de in zip(en_list, de_list):
        # Detach encoder — no gradient flows back to DINOv2 (frozen backbone)
        en_ = en.detach()
        de_ = de

        with torch.no_grad():
            # Per-patch cosine distance: [B, H, W] → [B, H*W, 1]
            point_dist = (1 - cos_loss(en_, de_)).unsqueeze(1)  # [B, 1, H, W]
            point_dist_flat = point_dist.reshape(-1)

            # p-th percentile threshold (top (1-p)% = hardest patches)
            k = max(1, int(point_dist_flat.numel() * (1 - p)))
            thresh = torch.topk(point_dist_flat, k=k)[0][-1]

        # Reconstruction loss (global cosine distance, flattened)
        en_flat = en_.reshape(en_.shape[0], en_.shape[1], -1)   # [B, C, N]
        de_flat = de_.reshape(de_.shape[0], de_.shape[1], -1)   # [B, C, N]
        loss = loss + torch.mean(1 - cos_loss(en_flat, de_flat))

        # Register gradient hook: suppress gradients for easy patches.
        # Store handle so we can remove it after backward (prevents hook accumulation / memory leak).
        easy_mask = (point_dist < thresh)  # captured in closure
        hook_fn = partial(modify_grad, inds=easy_mask, factor=factor)
        handle = de_.register_hook(hook_fn)
        hook_handles.append(handle)

    loss = loss / len(en_list)

    # Auto-remove all hooks after backward to prevent memory leak across iterations.
    def _cleanup_hooks(grad):
        for h in hook_handles:
            h.remove()
    loss.register_hook(_cleanup_hooks)

    return loss


def gct_cosine_loss(proj_gct: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
    """
    GCT Loss: Cosine Distance between projected GCT output and DINOv2 CLS token.

    L_gct = 1 - cosine_similarity(proj(gct_out), cls.detach())

    The .detach() on cls_token is CRITICAL: prevents gradients flowing
    into the frozen DINOv2 backbone.

    Args:
        proj_gct:  [B, C] — GCT token after projection head
        cls_token: [B, C] — DINOv2 CLS token (should already be detached)
    """
    return (1.0 - F.cosine_similarity(proj_gct, cls_token.detach(), dim=-1)).mean()


def combined_loss(
    en_list: list,
    de_list: list,
    gct_loss: torch.Tensor,
    p: float = 0.9,
    factor: float = 0.1,
    gct_lambda: float = 0.1,
) -> torch.Tensor:
    """
    Total training loss:
      L = L_reconstruction + λ × L_gct

    Args:
        en_list:    encoder feature maps
        de_list:    decoder feature maps
        gct_loss:   precomputed GCT cosine loss
        p:          hard mining percentile
        factor:     gradient suppression factor for easy patches
        gct_lambda: weight for GCT loss term
    """
    l_rec = global_cosine_hm_percent(en_list, de_list, p=p, factor=factor)
    return l_rec + gct_lambda * gct_loss
