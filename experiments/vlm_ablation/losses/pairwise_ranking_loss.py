"""
Pairwise ranking loss module.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.cosine_loss import global_cosine_hm_percent


class AnomalyPairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        self.ranking_loss_fn = nn.MarginRankingLoss(margin=margin, reduction="mean")

    def forward(
        self,
        score_1: torch.Tensor,
        score_2: torch.Tensor,
        target_rank: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = (target_rank != 0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=score_1.device, requires_grad=True)

        s1 = score_1[valid_mask]
        s2 = score_2[valid_mask]
        y = target_rank[valid_mask]

        return self.ranking_loss_fn(s2, s1, y)


def compute_combined_ranking_loss(
    en_list: list,
    de_list: list,
    gct_loss: torch.Tensor,
    score_p1: Optional[torch.Tensor] = None,
    score_p2: Optional[torch.Tensor] = None,
    target_rank: Optional[torch.Tensor] = None,
    gct_lambda: float = 0.5,
    rank_lambda: float = 0.05,
    rank_margin: float = 0.1,
    p: float = 0.9,
    factor: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # reconstruction loss
    recon_loss = global_cosine_hm_percent(en_list, de_list, p=p, factor=factor)

    # pairwise ranking loss
    if (
        score_p1 is not None and
        score_p2 is not None and
        target_rank is not None and
        rank_lambda > 0.0
    ):
        ranking_fn = AnomalyPairwiseRankingLoss(margin=rank_margin)
        rank_loss = ranking_fn(score_p1, score_p2, target_rank)
    else:
        rank_loss = torch.tensor(0.0, device=gct_loss.device)

    total_loss = recon_loss + (gct_lambda * gct_loss) + (rank_lambda * rank_loss)

    loss_dict = {
        "total_loss": float(total_loss.item()),
        "recon_loss": float(recon_loss.item()),
        "gct_loss": float(gct_loss.item()),
        "rank_loss": float(rank_loss.item())
    }

    return total_loss, loss_dict


# Convenience alias
PairwiseMarginRankingLoss = AnomalyPairwiseRankingLoss

