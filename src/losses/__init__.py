"""Loss functions for ViTill-GCT and comparative baseline."""

from src.losses.cosine_loss import (
    combined_loss,
    global_cosine_hm_percent,
    gct_cosine_loss,
)
from src.losses.gct_loss import GlobalConsistencyLoss

__all__ = [
    "combined_loss",
    "global_cosine_hm_percent",
    "gct_cosine_loss",
    "GlobalConsistencyLoss",
]
