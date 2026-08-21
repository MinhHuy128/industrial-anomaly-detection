import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine_hm_percent(
    en_list: list,
    de_list: list,
    p: float = 0.9,
    factor: float = 0.
) -> torch.Tensor:
    # hard patch mining: backprop only on the top (1-p)% hardest patches
    cos_loss = nn.CosineSimilarity(dim=1)
    loss = torch.tensor(0., device=en_list[0].device)
    hook_handles = []

    for en, de in zip(en_list, de_list):
        en_ = en.detach()
        de_ = de

        with torch.no_grad():
            point_dist = (1 - cos_loss(en_, de_)).unsqueeze(1)
            point_dist_flat = point_dist.reshape(-1)

            # find threshold for top (1-p)% hardest patches
            k = max(1, int(point_dist_flat.numel() * (1 - p)))
            thresh = torch.topk(point_dist_flat, k=k)[0][-1]

        en_flat = en_.reshape(en_.shape[0], en_.shape[1], -1)
        de_flat = de_.reshape(de_.shape[0], de_.shape[1], -1)
        loss = loss + torch.mean(1 - cos_loss(en_flat, de_flat))

        # zero out gradient on easy patches below threshold
        if de_.requires_grad:
            easy_mask = (point_dist < thresh)
            hook_fn = partial(modify_grad, inds=easy_mask, factor=factor)
            handle = de_.register_hook(hook_fn)
            hook_handles.append(handle)

    loss = loss / len(en_list)

    # remove backward hooks
    if loss.requires_grad and hook_handles:
        def _cleanup_hooks(grad):
            for h in hook_handles:
                h.remove()
        loss.register_hook(_cleanup_hooks)

    return loss


def gct_cosine_loss(proj_gct: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(proj_gct, cls_token.detach(), dim=-1)).mean()


def combined_loss(
    en_list: list,
    de_list: list,
    gct_loss: torch.Tensor,
    p: float = 0.9,
    factor: float = 0.1,
    gct_lambda: float = 0.1,
) -> torch.Tensor:
    l_rec = global_cosine_hm_percent(en_list, de_list, p=p, factor=factor)
    return l_rec + gct_lambda * gct_loss
