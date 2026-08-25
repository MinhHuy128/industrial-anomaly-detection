"""
Cross-modal distillation loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from src.losses.cosine_loss import global_cosine_hm_percent


def vlm_distillation_loss(proj_vlm: torch.Tensor, vlm_teacher_emb: torch.Tensor) -> torch.Tensor:
    """
    Computes Cosine Alignment Distillation Loss between Student GCT projection and Frozen VLM Teacher.
    L_distill = 1 - cos_sim(proj_vlm, vlm_teacher_emb)
    """
    proj_norm = F.normalize(proj_vlm, dim=-1)
    teacher_norm = F.normalize(vlm_teacher_emb.detach(), dim=-1)
    return (1.0 - torch.sum(proj_norm * teacher_norm, dim=-1)).mean()


def combined_vlm_distill_loss(
    en_list: list,
    de_list: list,
    gct_cls_loss: torch.Tensor,
    gct_vlm_loss: torch.Tensor,
    p: float = 0.9,
    factor: float = 0.1,
    lambda_cls: float = 0.1,
    lambda_vlm: float = 0.1,
) -> torch.Tensor:
    """
    Total multi-task training objective:
    L_total = L_recon + lambda_cls * L_cls + lambda_vlm * L_vlm
    """
    l_rec = global_cosine_hm_percent(en_list, de_list, p=p, factor=factor)
    return l_rec + (lambda_cls * gct_cls_loss) + (lambda_vlm * gct_vlm_loss)
