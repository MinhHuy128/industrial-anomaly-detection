"""
Training script for ranking and ablation models.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from src.models.vitill_gct import load_dinov2_register, extract_intermediate_features
from experiments.vlm_ablation.models.vitill_gct_ranking import ViTillGCTRanking
from experiments.vlm_ablation.losses.pairwise_ranking_loss import compute_combined_ranking_loss
from experiments.vlm_ablation.data.synthetic_anomaly_pairs import MVTecLocoPairwiseDataset


def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class WarmCosineScheduler:
    def __init__(self, optimizer, base_lr: float, final_lr: float, total_iters: int, warmup_iters: int = 100):
        self.optimizer = optimizer
        warmup = np.linspace(0.0, base_lr, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        cosine = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * iters / max(1, len(iters))))
        self.schedule = np.concatenate([warmup, cosine])
        self._step = 0

    def step(self) -> float:
        if self._step < len(self.schedule):
            lr = float(self.schedule[self._step])
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            lr = float(self.schedule[-1])
        self._step += 1
        return lr


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        if (ROOT / path).exists():
            path = ROOT / path
        elif (ROOT / "experiments" / "vlm_ablation" / "configs" / "vlm_ranking_loco.yaml").exists():
            path = ROOT / "experiments" / "vlm_ablation" / "configs" / "vlm_ranking_loco.yaml"

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def build_model_for_mode(mode: str, cfg: dict, device: torch.device) -> Tuple[ViTillGCTRanking, float]:
    m_cfg = cfg.get("model", {})
    ab_cfg = cfg.get("ablation", {})
    t_cfg = cfg.get("train", {})


    embed_dim = m_cfg.get("embed_dim", 768)
    vlm_dim = m_cfg.get("vlm_dim", 512)
    num_heads = m_cfg.get("num_heads", 12)
    decoder_layers = m_cfg.get("decoder_layers", 8)
    bottleneck_drop = m_cfg.get("bottleneck_drop", 0.2)
    target_layers = m_cfg.get("target_layers", [2, 3, 4, 5, 6, 7, 8, 9])
    fuse_enc = m_cfg.get("fuse_layer_encoder", [[0, 1, 2, 3], [4, 5, 6, 7]])
    fuse_dec = m_cfg.get("fuse_layer_decoder", [[0, 1, 2, 3], [4, 5, 6, 7]])
    init_gamma = ab_cfg.get("init_gamma", -4.0)

    rank_lambda = t_cfg.get("rank_lambda", 0.05)

    if mode == "gct_baseline":
        use_gct = True
        use_adapter = False
        use_mod = False
        active_rank_lambda = 0.0
    elif mode == "gct_ranking":
        use_gct = True
        use_adapter = False
        use_mod = False
        active_rank_lambda = rank_lambda
    elif mode == "gct_adapter":
        use_gct = True
        use_adapter = True
        use_mod = False
        active_rank_lambda = 0.0
    elif mode == "gct_adapter_modulation":
        use_gct = True
        use_adapter = True
        use_mod = True
        active_rank_lambda = 0.0
    elif mode == "gct_adapter_modulation_ranking":
        use_gct = True
        use_adapter = True
        use_mod = True
        active_rank_lambda = rank_lambda
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    model = ViTillGCTRanking(
        embed_dim=embed_dim,
        vlm_dim=vlm_dim,
        num_heads=num_heads,
        num_decoder_layers=decoder_layers,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_enc,
        fuse_layer_decoder=fuse_dec,
        bottleneck_drop=bottleneck_drop,
        use_gct=use_gct,
        use_adapter=use_adapter,
        use_modulation=use_mod,
        init_gamma=init_gamma
    ).to(device)

    return model, active_rank_lambda


def train_category(
    category: str,
    cfg: dict,
    mode: Optional[str] = None,
    rank_lambda_override: Optional[float] = None,
    total_iters_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    accum_steps_override: Optional[int] = None,
    device_str: Optional[str] = None
) -> Dict:
    """Executes training for a single category under the designated ablation mode."""
    p_cfg = cfg.get("project", {})
    d_cfg = cfg.get("dataset", {})
    t_cfg = cfg.get("train", {})

    seed = p_cfg.get("seed", 42)
    set_deterministic(seed)

    if device_str is None:
        device_str = p_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    target_mode = mode or cfg.get("ablation", {}).get("mode", "gct_ranking")
    data_root = ROOT / d_cfg.get("data_path", "data/mvtec_loco")
    cat_root = data_root / category
    vlm_cache_dir = ROOT / d_cfg.get("vlm_cache_path", "data/vlm_cache")
    vlm_cache_json = vlm_cache_dir / f"{category}_pairwise_vlm_labels.json"
    vlm_cache_pt = vlm_cache_dir / f"{category}_vlm_embeddings.pt"

    total_iters = total_iters_override or t_cfg.get("total_iters", 5000)
    batch_size = batch_size_override or t_cfg.get("batch_size", 4)
    accum_steps = accum_steps_override or t_cfg.get("accum_steps", 4)
    use_amp = t_cfg.get("use_amp", True) and (device.type == "cuda")

    base_lr = t_cfg.get("learning_rate", 0.002)
    final_lr = t_cfg.get("final_lr", 0.0002)
    warmup_iters = min(100, total_iters // 10)
    weight_decay = t_cfg.get("weight_decay", 0.0001)
    gct_lambda = t_cfg.get("gct_lambda", 0.5)
    rank_margin = t_cfg.get("rank_margin", 0.10)
    hm_p = t_cfg.get("hm_p", 0.9)
    hm_factor = t_cfg.get("hm_factor", 0.1)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info()
        vram_str = f"VRAM Free: {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB"
    else:
        vram_str = "CPU Mode"

    print(f"[TRAIN] Category: {category} | Mode: {target_mode} | Iters: {total_iters} | Batch: {batch_size} | Device: {device}")

    # dataset & dataloader
    dataset = MVTecLocoPairwiseDataset(
        category_root=cat_root,
        vlm_cache_path=vlm_cache_json if vlm_cache_json.exists() else None,
        vlm_embed_path=vlm_cache_pt if vlm_cache_pt.exists() else None,
        img_size=d_cfg.get("img_size", 448),
        crop_size=d_cfg.get("crop_size", 392),
        synthetic_on_the_fly=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # backbone and model
    backbone = load_dinov2_register(device)
    model, rank_lambda = build_model_for_mode(target_mode, cfg, device)
    if rank_lambda_override is not None:
        rank_lambda = rank_lambda_override

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = WarmCosineScheduler(optimizer, base_lr, final_lr, total_iters, warmup_iters)


    # checkpoint save dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = ROOT / t_cfg.get("save_dir", "experiments") / f"{category}_{target_mode}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # training loop
    target_layers = cfg.get("model", {}).get("target_layers", [2, 3, 4, 5, 6, 7, 8, 9])
    model.train()
    iter_count = 0
    start_time = time.time()
    loss_history = []
    gate_history = []
    optimizer.zero_grad()

    while iter_count < total_iters:
        for x_normal, x_p1, x_p2, rank_target, vlm_feat in dataloader:
            if iter_count >= total_iters:
                break

            current_lr = scheduler.step()
            x_normal = x_normal.to(device, non_blocking=True)
            vlm_feat = vlm_feat.to(device=device, dtype=torch.float32, non_blocking=True)

            with torch.no_grad():
                feat_list, cls_token = extract_intermediate_features(backbone, x_normal, target_layers, return_cls=True)

            # Forward normal sample for reconstruction
            en, de, gct_loss, current_gate = model(feat_list, cls_token=cls_token, vlm_feat=vlm_feat)

            # Pairwise Ranking Stream (if enabled)
            if rank_lambda > 0.0:
                x_p1 = x_p1.to(device, non_blocking=True)
                x_p2 = x_p2.to(device, non_blocking=True)
                rank_target = rank_target.to(device, non_blocking=True)

                with torch.no_grad():
                    f1_list, c1 = extract_intermediate_features(backbone, x_p1, target_layers, return_cls=True)
                    f2_list, c2 = extract_intermediate_features(backbone, x_p2, target_layers, return_cls=True)

                en_1, de_1, g1, _ = model(f1_list, cls_token=c1)
                en_2, de_2, g2, _ = model(f2_list, cls_token=c2)

                score_p1 = model.compute_anomaly_score_from_maps(en_1, de_1, g1, gamma=cfg.get("eval", {}).get("gct_gamma", 1.0))
                score_p2 = model.compute_anomaly_score_from_maps(en_2, de_2, g2, gamma=cfg.get("eval", {}).get("gct_gamma", 1.0))
            else:
                score_p1 = None
                score_p2 = None
                rank_target = None

            p_curr = min(hm_p * iter_count / 1000.0, hm_p)
            total_loss, loss_dict = compute_combined_ranking_loss(
                en_list=en,
                de_list=de,
                gct_loss=gct_loss,
                score_p1=score_p1,
                score_p2=score_p2,
                target_rank=rank_target,
                gct_lambda=gct_lambda,
                rank_lambda=rank_lambda,
                rank_margin=rank_margin,
                p=p_curr,
                factor=hm_factor
            )
            loss_step = total_loss / max(1, accum_steps)
            loss_step.backward()

            iter_count += 1
            if iter_count % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_history.append(loss_dict["total_loss"])
            gate_history.append(current_gate)

            if iter_count % t_cfg.get("log_interval", 100) == 0 or iter_count == total_iters:
                elapsed = time.time() - start_time
                print(
                    f"[{iter_count:04d}/{total_iters:04d}] "
                    f"Loss: {loss_dict['total_loss']:.4f} | "
                    f"Recon: {loss_dict['recon_loss']:.4f} | "
                    f"GCT: {loss_dict['gct_loss']:.4f} | "
                    f"Rank: {loss_dict['rank_loss']:.4f} | "
                    f"Gate: {current_gate:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

    # save checkpoint
    ckpt_path = exp_dir / "best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "mode": target_mode,
        "category": category,
        "rank_lambda": rank_lambda,
        "final_gate_weight": float(model.get_gate_weight()),
        "seed": seed,
        "total_iters": total_iters
    }, ckpt_path)

    final_summary = {
        "category": category,
        "mode": target_mode,
        "seed": seed,
        "total_iters": total_iters,
        "final_loss": round(float(loss_history[-1]), 5),
        "mean_loss_last100": round(float(np.mean(loss_history[-100:])), 5),
        "final_gate_weight": round(float(model.get_gate_weight()), 5),
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "rank_lambda": rank_lambda
    }

    with open(exp_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "loss_history": [round(float(v), 5) for v in loss_history[::10]],
            "gate_history": [round(float(v), 5) for v in gate_history[::10]]
        }, f, indent=2)

    print(f"\n[INFO] Finished {category} ({target_mode}). Checkpoint -> {ckpt_path.name}")
    return final_summary


def main():
    parser = argparse.ArgumentParser(description="Train ViTill-GCT with Pairwise VLM Ranking & Gated Adapter.")
    parser.add_argument("--config", type=str, default="experiments/vlm_ablation/configs/vlm_ranking_loco.yaml", help="Path to config YAML")
    parser.add_argument("--category", type=str, default="pushpins", help="Category to train")
    parser.add_argument("--mode", type=str, choices=[
        "gct_baseline", "gct_ranking", "gct_adapter", "gct_adapter_modulation", "gct_adapter_modulation_ranking"
    ], default=None, help="Ablation mode override")
    parser.add_argument("--rank_lambda", type=float, default=None, help="Override lambda for ranking loss")
    parser.add_argument("--iters", type=int, default=None, help="Override total iterations")
    parser.add_argument("--batch_size", type=int, default=None, help="Micro-batch size (default: 4)")
    parser.add_argument("--accum_steps", type=int, default=None, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_category(
        category=args.category,
        cfg=cfg,
        mode=args.mode,
        rank_lambda_override=args.rank_lambda,
        total_iters_override=args.iters,
        batch_size_override=args.batch_size,
        accum_steps_override=args.accum_steps,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
