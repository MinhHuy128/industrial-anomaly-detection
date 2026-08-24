"""
Loss functions for compositional VLM ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.cosine_loss import global_cosine_hm_percent


def combined_comp_vlm_loss(
    en_list: list,
    de_list: list,
    gct_loss: torch.Tensor,
    slot_ortho_loss: torch.Tensor,
    vlm_coverage_loss: torch.Tensor,
    p: float = 0.9,
    factor: float = 0.1,
    lambda_gct: float = 0.1,
    lambda_ortho: float = 0.05,
    lambda_vlm: float = 0.05
) -> torch.Tensor:
    """
    Multi-task objective:
    L_total = L_recon + lambda_gct * L_gct + lambda_ortho * L_ortho + lambda_vlm * L_vlm
    """
    l_rec = global_cosine_hm_percent(en_list, de_list, p=p, factor=factor)
    total_loss = (
        l_rec +
        (lambda_gct * gct_loss) +
        (lambda_ortho * slot_ortho_loss) +
        (lambda_vlm * vlm_coverage_loss)
    )
    return total_loss
