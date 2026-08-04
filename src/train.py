"""
Training script for ViTill-GCT (Paper-Strict Branch).

Faithful to Dinomaly paper training protocol:
  - Iteration-based: 5000 iterations (not epoch-based)
  - Batch size: 16
  - Image size: 448, Crop size: 392
  - Optimizer: StableAdamW(lr=2e-3, betas=(0.9,0.999), wd=1e-4, amsgrad=True)
  - Scheduler: WarmCosineScheduler(base=2e-3, final=2e-4, warmup=100)
  - Loss: global_cosine_hm_percent(p=0.9, factor=0.1) + λ×GCT_cosine_loss
  - Grad clipping: max_norm=0.1

Reference: Kang et al., "Dinomaly: The Less Is More...", arXiv 2405.14325
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.vitill_gct import ViTillGCT, ViTillBaseline, load_dinov2_register, extract_intermediate_features
from src.losses.cosine_loss import combined_loss, global_cosine_hm_percent

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_deterministic(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class MVTecLocoTrainDataset(Dataset):
    """
    Loads 'good' training images from MVTec LOCO AD.
    Uses same transforms as Dinomaly paper:
      - Resize to 448
      - CenterCrop 392
      - Normalize with ImageNet stats
    """
    def __init__(self, category_root: Path, img_size: int = 448, crop_size: int = 392):
        train_dir = category_root / "train" / "good"
        if not train_dir.exists():
            raise FileNotFoundError(f"Training dir not found: {train_dir}")

        self.img_paths = sorted(
            list(train_dir.glob("*.png")) +
            list(train_dir.glob("*.jpg")) +
            list(train_dir.glob("*.bmp"))
        )
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No training images found in: {train_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise RuntimeError("PyYAML not installed. Use .json config.")


def save_loss_curve(save_path: Path, iter_history, loss_history, gct_history=None):
    if not iter_history or not loss_history:
        print("[WARN] No training history available; skipping loss curve export.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 800
    margin_left, margin_right = 90, 30
    margin_top, margin_bottom = 70, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    values = list(loss_history)
    if gct_history is not None and any(value != 0.0 for value in gct_history):
        values.extend(gct_history)

    min_loss = min(values)
    max_loss = max(values)
    if max_loss <= min_loss:
        max_loss = min_loss + 1.0

    def x_pos(index: int) -> float:
        if len(iter_history) == 1:
            return margin_left + plot_w / 2.0
        return margin_left + (index / (len(iter_history) - 1)) * plot_w

    def y_pos(value: float) -> float:
        return margin_top + (max_loss - value) / (max_loss - min_loss) * plot_h

    # Axes and grid
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="black", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="black", width=2)

    for frac in [0.25, 0.5, 0.75]:
        y = margin_top + plot_h * frac
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(220, 220, 220), width=1)

    def draw_polyline(series, color):
        if series is None:
            return
        points = [(x_pos(i), y_pos(float(value))) for i, value in enumerate(series)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for x, y in points[::max(1, len(points) // 50)]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    draw_polyline(loss_history, (31, 119, 180))
    if gct_history is not None and any(value != 0.0 for value in gct_history):
        draw_polyline(gct_history, (214, 39, 40))

    title = "Training Loss Curve"
    draw.text((margin_left, 20), title, fill="black")
    draw.text((margin_left, height - 45), "Iteration", fill="black")
    draw.text((15, margin_top), "Loss", fill="black")
    draw.text((margin_left + 20, 40), "Blue: Total Loss", fill=(31, 119, 180))
    if gct_history is not None and any(value != 0.0 for value in gct_history):
        draw.text((margin_left + 220, 40), "Red: GCT Loss", fill=(214, 39, 40))

    image.save(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER (WarmCosineScheduler from paper)
# ─────────────────────────────────────────────────────────────────────────────
class WarmCosineScheduler:
    """
    Linear warmup + Cosine decay LR schedule (matches paper exactly).
    Operates per-iteration (not per-epoch).
    """
    def __init__(self, optimizer, base_lr, final_lr, total_iters, warmup_iters=100):
        self.optimizer = optimizer
        warmup  = np.linspace(0., base_lr, warmup_iters)
        iters   = np.arange(total_iters - warmup_iters)
        cosine  = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate([warmup, cosine])
        self._step = 0

    def step(self):
        if self._step < len(self.schedule):
            lr = float(self.schedule[self._step])
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        self._step += 1


# ─────────────────────────────────────────────────────────────────────────────
# STABLEADAMW (from paper's optimizers/)
# ─────────────────────────────────────────────────────────────────────────────
class StableAdamW(torch.optim.Optimizer):
    """
    AdamW with gradient clipping via RMS norm (Stable AdamW variant).
    Taken verbatim from Dinomaly/optimizers/StableAdamW.py.
    """
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4, amsgrad=True, clip_threshold=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        clip_threshold=clip_threshold)
        super().__init__(params, defaults)

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad    = p.grad
                amsgrad = group['amsgrad']
                state   = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg']    = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_sq = state['max_exp_avg_sq']
                    torch.max(max_sq, exp_avg_sq, out=max_sq)
                    denom = (max_sq.sqrt() / math.sqrt(bc2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(group['eps'])

                lr_scale = grad / denom
                lr_scale = max(1.0, self._rms(lr_scale) / group['clip_threshold'])
                step_size = group['lr'] / bc1 / lr_scale
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    cfg    = load_config(args.config)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else cfg["project"]["device"])
    set_deterministic(SEED)

    use_gct    = args.use_gct
    model_name = "DINOMALY + GCT (Paper-Strict)" if use_gct else "DINOMALY BASELINE (Paper-Strict)"
    category   = args.category

    print(f"[INFO] GPU : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"[INFO] Model: {model_name}  |  Category: {category}")
    print(f"[SEED] {SEED}")

    # ── Hyper-parameters (paper defaults) ─────────────────────────────────
    TOTAL_ITERS  = cfg.get("train", {}).get("total_iters", 5000)
    BATCH_SIZE   = cfg.get("train", {}).get("batch_size", 16)
    IMG_SIZE     = cfg.get("dataset", {}).get("img_size", 448)
    CROP_SIZE    = cfg.get("dataset", {}).get("crop_size", 392)
    BASE_LR      = cfg.get("train", {}).get("learning_rate", 2e-3)
    FINAL_LR     = cfg.get("train", {}).get("final_lr", 2e-4)
    WARMUP_ITERS = cfg.get("train", {}).get("warmup_iters", 100)
    WEIGHT_DECAY = cfg.get("train", {}).get("weight_decay", 1e-4)
    GCT_LAMBDA   = cfg.get("train", {}).get("gct_lambda", 0.1)
    HM_P         = cfg.get("train", {}).get("hm_p", 0.9)
    HM_FACTOR    = cfg.get("train", {}).get("hm_factor", 0.1)
    SAVE_DIR     = ROOT / cfg["train"]["save_dir"]
    TARGET_LAYERS = cfg.get("model", {}).get("target_layers", [2, 3, 4, 5, 6, 7, 8, 9])
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # ── Dataset ────────────────────────────────────────────────────────────
    data_root = ROOT / cfg["dataset"]["data_path"] / category
    dataset   = MVTecLocoTrainDataset(data_root, img_size=IMG_SIZE, crop_size=CROP_SIZE)
    loader    = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[DATASET] Loaded {len(dataset)} training images from: {data_root / 'train/good'}")

    # ── Load DINOv2-Register Backbone (frozen) ─────────────────────────────
    backbone = load_dinov2_register(device)

    # ── Build Model ────────────────────────────────────────────────────────
    embed_dim     = cfg["model"]["embed_dim"]       # 768
    num_decoder   = cfg["model"]["decoder_layers"]  # 8

    if use_gct:
        model = ViTillGCT(
            embed_dim=embed_dim,
            num_decoder_layers=num_decoder,
            target_layers=TARGET_LAYERS,
            gct_lambda=GCT_LAMBDA,
        ).to(device)
        trainable_params = list(model.bottleneck.parameters()) + \
                           list(model.decoder.parameters())    + \
                           list(model.gct.parameters())
    else:
        model = ViTillBaseline(
            embed_dim=embed_dim,
            num_decoder_layers=num_decoder,
            target_layers=TARGET_LAYERS,
        ).to(device)
        trainable_params = list(model.bottleneck.parameters()) + \
                           list(model.decoder.parameters())

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer  = StableAdamW(
        [{'params': trainable_params}],
        lr=BASE_LR, betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY, amsgrad=True, eps=1e-8
    )
    scheduler  = WarmCosineScheduler(optimizer, BASE_LR, FINAL_LR, TOTAL_ITERS, WARMUP_ITERS)

    # ── Artifacts / Logging ────────────────────────────────────────────────
    save_path = SAVE_DIR / ("gct" if use_gct else "baseline")
    save_path.mkdir(parents=True, exist_ok=True)
    log_path = save_path / "train.log"
    tb_writer = None
    if SummaryWriter is not None:
        tb_dir = save_path / "tensorboard" / f"{category}_{timestamp}"
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[TENSORBOARD] Writing events to: {tb_dir.relative_to(ROOT)}")
    else:
        print("[WARN] torch.utils.tensorboard unavailable; continuing without TensorBoard.")

    def log_message(message: str):
        print(message)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    iter_history = []
    loss_history = []
    gct_history = []
    lr_history = []

    log_message(f"[RUN] {timestamp} | model={model_name} | category={category} | config={args.config}")
    log_message(f"[RUN] train.log => {log_path.relative_to(ROOT)}")

    # ── Training Loop (iteration-based, not epoch-based) ──────────────────
    model.train()
    data_iter  = iter(loader)
    start_time = time.time()
    loss_accum = []

    log_message(f"[TRAIN] Starting training loop for {TOTAL_ITERS} iterations...")

    for it in range(TOTAL_ITERS):
        # Infinite data iterator
        try:
            imgs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs = next(data_iter)

        imgs = imgs.to(device, non_blocking=True)

        # Extract features from frozen DINOv2-Register backbone
        with torch.no_grad():
            feat_list, cls_token = extract_intermediate_features(
                backbone, imgs, TARGET_LAYERS, return_cls=True
            )

        # Forward through Bottleneck + GCT + Decoder
        optimizer.zero_grad()

        if use_gct:
            en, de, gct_loss = model(feat_list, cls_token)
            # Progressive p warmup: paper uses p = min(0.9 * it/1000, 0.9)
            p_curr = min(HM_P * it / 1000.0, HM_P)
            loss = combined_loss(en, de, gct_loss, p=p_curr, factor=HM_FACTOR, gct_lambda=GCT_LAMBDA)
            gct_loss_value = float(gct_loss.item())
        else:
            en, de = model(feat_list)
            # Progressive p warmup (same curriculum as paper)
            p_curr = min(HM_P * it / 1000.0, HM_P)
            loss = global_cosine_hm_percent(en, de, p=p_curr, factor=HM_FACTOR)
            gct_loss_value = 0.0

        loss.backward()
        nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
        optimizer.step()
        scheduler.step()

        iter_idx = it + 1
        loss_value = float(loss.item())
        loss_accum.append(loss.item())
        iter_history.append(iter_idx)
        loss_history.append(loss_value)
        gct_history.append(gct_loss_value)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", loss_value, iter_idx)
            tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iter_idx)
            tb_writer.add_scalar("train/p_curr", p_curr, iter_idx)
            if use_gct:
                tb_writer.add_scalar("train/gct_loss", gct_loss_value, iter_idx)

        # Logging every 100 iters
        if iter_idx % 100 == 0:
            elapsed = time.time() - start_time
            avg_loss = np.mean(loss_accum)
            speed = elapsed / iter_idx
            remaining = speed * (TOTAL_ITERS - it - 1)
            log_message(f"  [Iter {iter_idx:05d}/{TOTAL_ITERS}] "
                        f"Loss: {avg_loss:.6f}  "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}  "
                        f"Elapsed: {elapsed:.0f}s  "
                        f"ETA: {remaining:.0f}s")
            loss_accum = []

    total_time = time.time() - start_time
    log_message(f"[SUCCESS] Training completed in {total_time:.1f} seconds.")

    curve_path = save_path / f"loss_curve_{category}_{timestamp}.png"
    save_loss_curve(curve_path, iter_history, loss_history, gct_history)
    log_message(f"[SAVED] Loss curve saved to: {curve_path.relative_to(ROOT)}")

    manifest = {
        "timestamp": timestamp,
        "config": str(Path(args.config).as_posix()),
        "category": category,
        "model_name": model_name,
        "use_gct": use_gct,
        "seed": SEED,
        "total_iters": TOTAL_ITERS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "crop_size": CROP_SIZE,
        "checkpoint": str((save_path / f"{'gct' if use_gct else 'baseline'}_{category}_strict.pth").relative_to(ROOT)),
        "train_log": str(log_path.relative_to(ROOT)),
        "loss_curve": str(curve_path.relative_to(ROOT)),
        "tensorboard": str((save_path / "tensorboard" / f"{category}_{timestamp}").relative_to(ROOT)) if tb_writer is not None else None,
    }
    manifest_path = save_path / f"run_manifest_{category}_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_message(f"[SAVED] Run manifest saved to: {manifest_path.relative_to(ROOT)}")

    # ── Save Checkpoint ────────────────────────────────────────────────────
    prefix    = "gct" if use_gct else "baseline"
    subfolder = "gct" if use_gct else "baseline"
    ckpt_path = save_path / f"{prefix}_{category}_strict.pth"

    # Only save trainable parameters (backbone is frozen, not included)
    save_dict = {
        "model_state": model.state_dict(),
        "category":    category,
        "model_type":  "gct" if use_gct else "baseline",
        "total_iters": TOTAL_ITERS,
    }
    torch.save(save_dict, ckpt_path)
    log_message(f"[SAVED] Checkpoint saved to: {ckpt_path.relative_to(ROOT)}")

    if tb_writer is not None:
        tb_writer.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViTill-GCT on MVTec LOCO AD (Paper-Strict)")
    parser.add_argument("--config",   type=str, default="src/configs/loco_strict.json")
    parser.add_argument("--category", type=str, default="breakfast_box")
    parser.add_argument("--use_gct",  action="store_true", help="Train GCT model (default: Baseline)")
    parser.add_argument("--cpu",      action="store_true")
    args = parser.parse_args()
    train(args)
