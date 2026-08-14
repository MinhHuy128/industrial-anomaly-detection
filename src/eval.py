"""
Evaluation script for ViTill-GCT on MVTec LOCO AD.
Computes image AUROC (Logical / Structural / Mean), sPRO pixel metric, and inference latency.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.vitill_gct import ViTillGCT, ViTillBaseline, load_dinov2_register, extract_intermediate_features


# ─────────────────────────────────────────────────────────────────────────────
# AUROC CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_auroc(labels, scores):
    """Compute ROC-AUC percentage (0 - 100%)."""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores) * 100.0
    except ImportError:
        labels = np.array(labels)
        scores = np.array(scores)
        pos    = scores[labels == 1]
        neg    = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 50.0
        rank_sum = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
        return (rank_sum / (len(pos) * len(neg))) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY MAP GENERATION & SPATIAL ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def compute_anomaly_map(en_list, de_list, crop_size: int = 392, out_size: int = 448) -> np.ndarray:
    """
    Computes per-patch cosine distance anomaly map with exact spatial alignment.
    Returns:
        smooth_amap: [448, 448] smoothed anomaly heatmap
    """
    anomaly_map = torch.zeros(1, 1, 1, 1, device=en_list[0].device)
    for en, de in zip(en_list, de_list):
        # en, de: [B, C, 28, 28] -> cosine distance per spatial location
        a_map = 1.0 - F.cosine_similarity(en, de, dim=1, eps=1e-8)  # [B, 28, 28]
        a_map = a_map.unsqueeze(1)                                    # [B, 1, 28, 28]
        # Upsample to crop region: [B, 1, 392, 392]
        a_map = F.interpolate(a_map, size=(crop_size, crop_size),
                               mode='bilinear', align_corners=False)
        anomaly_map = anomaly_map + a_map

    anomaly_map = anomaly_map / len(en_list)
    crop_amap   = anomaly_map.squeeze().cpu().numpy()  # [392, 392]

    # Paste into out_size canvas (448x448) at offset 28px to align with GT mask
    if crop_size < out_size:
        canvas = np.zeros((out_size, out_size), dtype=crop_amap.dtype)
        top  = (out_size - crop_size) // 2   # 28
        left = (out_size - crop_size) // 2   # 28
        canvas[top:top + crop_size, left:left + crop_size] = crop_amap
    else:
        canvas = crop_amap

    # Gaussian smoothing (sigma=4)
    smooth_amap = gaussian_filter(canvas, sigma=4)
    return smooth_amap


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SCORE POOLING
# ─────────────────────────────────────────────────────────────────────────────
def image_score(anomaly_map: np.ndarray, max_ratio: float = 0.01) -> float:
    """Compute image-level score via Top-1% mean percentile pooling."""
    flat = anomaly_map.flatten()
    k    = max(1, int(len(flat) * max_ratio))
    return float(np.sort(flat)[-k:].mean())


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE IMAGE INFERENCE (Dual-Stream Scoring)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_one(backbone, model, img_path: Path, transform, device, use_gct: bool,
              target_layers: list, out_size: int, crop_size: int = 392, gamma: float = 1.0) -> tuple:
    """
    Run inference on a single image.
    Score_final = Score_patch + gamma * Score_gct
    Returns (score_final, anomaly_map)
    """
    img   = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)  # [1, 3, 392, 392]

    feat_list, cls_token = extract_intermediate_features(backbone, img_t, target_layers)

    if use_gct:
        en, de, gct_loss = model(feat_list, cls_token)
        score_gct = float(gct_loss.item())
    else:
        en, de = model(feat_list)
        score_gct = 0.0

    amap  = compute_anomaly_map(en, de, crop_size=crop_size, out_size=out_size)
    score_patch = image_score(amap)

    # Combine local patch error and global GCT alignment score
    score_final = score_patch + (gamma * score_gct if use_gct else 0.0)
    return score_final, amap


# ─────────────────────────────────────────────────────────────────────────────
# SPRO PIXEL METRIC
# ─────────────────────────────────────────────────────────────────────────────
def compute_spro(backbone, model, test_path: Path, transform, device,
                 use_gct: bool, target_layers: list, img_size: int, crop_size: int = 392, gamma: float = 1.0) -> float:
    """Calculate sPRO (Structural Pseudo-ROC) metric over FPR range [0, 0.30]."""
    gt_root    = test_path.parent / "ground_truth"
    anom_types = ["logical_anomalies", "structural_anomalies"]
    all_gt, all_pred = [], []

    for anom_type in anom_types:
        anom_dir = test_path / anom_type
        gt_dir   = gt_root / anom_type
        if not anom_dir.exists() or not gt_dir.exists():
            continue
        img_paths = sorted(list(anom_dir.glob("*.png")) + list(anom_dir.glob("*.jpg")))
        for p in img_paths:
            stem      = p.stem
            mask_path = gt_dir / stem / (stem + ".png")
            if not mask_path.exists():
                mask_path = gt_dir / (stem + ".png")
            if not mask_path.exists():
                continue
            gt_mask = np.array(Image.open(mask_path).convert("L").resize(
                (img_size, img_size))) > 0

            _, amap = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
            all_gt.append(gt_mask.flatten())
            all_pred.append(amap.flatten())

    if len(all_gt) == 0:
        return -1.0

    all_gt   = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    spro_vals = []
    for fpr_limit in np.linspace(0, 0.30, 100):
        thresh   = np.percentile(all_pred, (1 - fpr_limit) * 100)
        pred_bin = all_pred >= thresh
        tp = np.sum(pred_bin & all_gt)
        fn = np.sum(~pred_bin & all_gt)
        spro_vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0.)

    return float(np.mean(spro_vals)) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(args):
    path = Path(args.config)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    device        = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                                else cfg["project"]["device"])
    category      = args.category
    use_gct       = args.use_gct
    img_size      = cfg["dataset"]["img_size"]   # 448
    crop_size     = cfg["dataset"].get("crop_size", 392)
    embed_dim     = cfg["model"]["embed_dim"]
    num_decoder   = cfg["model"]["decoder_layers"]
    target_layers = cfg["model"].get("target_layers", [2, 3, 4, 5, 6, 7, 8, 9])
    gamma         = args.gamma if args.gamma is not None else (cfg["eval"].get("gct_gamma", 1.0) if use_gct else 0.0)
    model_name    = "DINOMALY + GCT V2" if use_gct else "DINOMALY BASELINE"

    print(f"[EVAL] Model: {model_name}  |  Category: {category}  |  Device: {device}  |  Gamma: {gamma if use_gct else 0.0}")

    if use_gct:
        model = ViTillGCT(embed_dim=embed_dim, num_decoder_layers=num_decoder,
                          target_layers=target_layers).to(device)
    else:
        model = ViTillBaseline(embed_dim=embed_dim, num_decoder_layers=num_decoder,
                              target_layers=target_layers).to(device)

    # Load checkpoint
    ckpt_dir  = ROOT / cfg["train"]["save_dir"] / ("gct" if use_gct else "baseline")
    prefix    = "gct" if use_gct else "baseline"
    ckpt_path = ckpt_dir / f"{prefix}_{category}_strict.pth"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / f"{prefix}_{category}.pth"

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        print(f"[CKPT] Loaded checkpoint from: {ckpt_path.relative_to(ROOT)}")
    else:
        print(f"[WARN] Checkpoint not found at: {ckpt_path}. Evaluating untrained model!")

    model.eval()

    backbone = load_dinov2_register(device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    test_path = ROOT / cfg["dataset"]["data_path"] / category / "test"
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    # Evaluate Logical Anomalies
    log_labels, log_scores = [], []
    log_dir = test_path / "logical_anomalies"
    if log_dir.exists():
        for p in sorted(list(log_dir.glob("*.png")) + list(log_dir.glob("*.jpg"))):
            score, _ = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
            log_labels.append(1)
            log_scores.append(score)

    # Evaluate Structural Anomalies
    struct_labels, struct_scores = [], []
    struct_dir = test_path / "structural_anomalies"
    if struct_dir.exists():
        for p in sorted(list(struct_dir.glob("*.png")) + list(struct_dir.glob("*.jpg"))):
            score, _ = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
            struct_labels.append(1)
            struct_scores.append(score)

    # Evaluate Good images
    good_dir = test_path / "good"
    if good_dir.exists():
        for p in sorted(list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))):
            score, _ = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
            log_labels.append(0)
            log_scores.append(score)
            struct_labels.append(0)
            struct_scores.append(score)

    log_auroc    = compute_auroc(log_labels, log_scores)
    struct_auroc = compute_auroc(struct_labels, struct_scores)
    mean_auroc   = (log_auroc + struct_auroc) / 2.0

    print(f"[METRICS] Logical AUROC   : {log_auroc:.2f}%")
    print(f"[METRICS] Structural AUROC: {struct_auroc:.2f}%")
    print(f"[METRICS] Mean AUROC      : {mean_auroc:.2f}%")

    # Latency benchmark (batch_size=1)
    dummy_img = test_path / "good"
    sample_imgs = sorted(list(dummy_img.glob("*.png")) + list(dummy_img.glob("*.jpg")))
    if len(sample_imgs) > 0:
        sample_p = sample_imgs[0]
        # Warmup
        for _ in range(5):
            _ = infer_one(backbone, model, sample_p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        t0 = time.time()
        n_runs = 20
        for _ in range(n_runs):
            _ = infer_one(backbone, model, sample_p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        latency_ms = ((t1 - t0) / n_runs) * 1000.0
        print(f"[LATENCY] Batch=1 Inference Latency: {latency_ms:.2f} ms/image")

    spro = compute_spro(backbone, model, test_path, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
    print(f"[METRICS] sPRO (pixel-level): {spro:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViTill-GCT on MVTec LOCO AD")
    parser.add_argument("--config",   type=str, default="src/configs/loco_strict.json")
    parser.add_argument("--category", type=str, default="breakfast_box")
    parser.add_argument("--use_gct",  action="store_true", help="Evaluate GCT model")
    parser.add_argument("--gamma",    type=float, default=None, help="GCT active score weight gamma (default: 1.0)")
    parser.add_argument("--cpu",      action="store_true")
    args = parser.parse_args()
    evaluate(args)
