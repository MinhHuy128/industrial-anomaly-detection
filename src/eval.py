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

# AUROC CALCULATION
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

def compute_f1_max(labels, scores):
    """Compute optimal Image-level F1-max score across all thresholds (0 - 100%)."""
    try:
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        return float(np.nanmax(f1_scores)) * 100.0
    except Exception:
        return 0.0

# ANOMALY MAP GENERATION & SPATIAL ALIGNMENT
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

# IMAGE SCORE POOLING
def image_score(anomaly_map: np.ndarray, max_ratio: float = 0.01) -> float:
    """Compute image-level score via Top-1% mean percentile pooling."""
    flat = anomaly_map.flatten()
    k    = max(1, int(len(flat) * max_ratio))
    return float(np.sort(flat)[-k:].mean())

# SINGLE IMAGE INFERENCE (Dual-Stream Scoring)
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

# REGION OVERLAP METRIC: Normalized AUPRO (sPRO approximation, max_fpr=0.30)
def calculate_au_pro(masks: list, amaps: list, max_fpr: float = 0.30, num_thresholds: int = 500) -> float:
    """
    Computes Normalized AUPRO (sPRO approximation, max_fpr = 0.30).
    Uses connected-component region labeling via scipy.ndimage.label.
    """
    from scipy.ndimage import label as cc_label
    from sklearn.metrics import auc

    if len(masks) == 0 or len(amaps) == 0:
        return 0.0

    # 1. Identify connected component regions across all GT masks
    labeled_regions = []
    total_region_count = 0
    for mask in masks:
        labeled_mask, num_features = cc_label(mask)
        labeled_regions.append((labeled_mask, num_features))
        total_region_count += num_features

    if total_region_count == 0:
        return 100.0

    # 2. Extract candidate thresholds from predictions
    all_amaps_flat = np.concatenate([a.flatten() for a in amaps])
    thresholds     = np.quantile(all_amaps_flat, np.linspace(0.0, 1.0, num_thresholds))

    # Mask of normal/background pixels across all test images
    all_masks_flat    = np.concatenate([m.flatten() for m in masks])
    num_normal_pixels = np.sum(all_masks_flat == 0)

    if num_normal_pixels == 0:
        return 0.0

    pros, fprs = [], []

    # 3. Scan thresholds from highest to lowest
    for th in reversed(thresholds):
        fp_count = np.sum((all_amaps_flat >= th) & (all_masks_flat == 0))
        fpr      = fp_count / num_normal_pixels

        region_overlaps = []
        for amap, (labeled_mask, num_features) in zip(amaps, labeled_regions):
            if num_features == 0:
                continue
            pred_bin = (amap >= th)
            for region_idx in range(1, num_features + 1):
                region_mask = (labeled_mask == region_idx)
                region_size = np.sum(region_mask)
                if region_size > 0:
                    overlap = np.sum(pred_bin & region_mask) / region_size
                    region_overlaps.append(overlap)

        pro = np.mean(region_overlaps) if len(region_overlaps) > 0 else 0.0
        pros.append(pro)
        fprs.append(fpr)

    # 4. Filter curve up to max_fpr (0.30)
    fprs = np.array(fprs)
    pros = np.array(pros)

    unique_fprs, unique_indices = np.unique(fprs, return_index=True)
    unique_pros = pros[unique_indices]

    valid_mask = unique_fprs <= max_fpr
    valid_fprs = unique_fprs[valid_mask]
    valid_pros = unique_pros[valid_mask]

    if len(valid_fprs) < 2:
        return 0.0

    if valid_fprs[0] > 0.0:
        valid_fprs = np.insert(valid_fprs, 0, 0.0)
        valid_pros = np.insert(valid_pros, 0, 0.0)

    if valid_fprs[-1] < max_fpr:
        valid_pros = np.append(valid_pros, valid_pros[-1])
        valid_fprs = np.append(valid_fprs, max_fpr)

    # 5. Integrate area under curve and normalize by max_fpr
    au_pro = auc(valid_fprs, valid_pros) / max_fpr
    return float(au_pro * 100.0)

def compute_spro(backbone, model, test_path: Path, transform, device,
                 use_gct: bool, target_layers: list, img_size: int, crop_size: int = 392, gamma: float = 1.0) -> float:
    """Calculate Normalized AUPRO (sPRO, max_fpr=0.30) using connected component analysis."""
    gt_root    = test_path.parent / "ground_truth"
    anom_types = ["logical_anomalies", "structural_anomalies"]
    all_masks, all_amaps = [], []

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
            # Use NEAREST interpolation to preserve binary mask boundary integrity
            resample_mode = getattr(Image, "Resampling", Image).NEAREST
            gt_mask = np.array(Image.open(mask_path).convert("L").resize(
                (img_size, img_size), resample=resample_mode)) > 0

            _, amap = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
            all_masks.append(gt_mask)
            all_amaps.append(amap)

    # Also add normal good images (with zero masks) for accurate background FPR calculation
    good_dir = test_path / "good"
    if good_dir.exists():
        for p in sorted(list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))):
            _, amap = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
            all_masks.append(np.zeros((img_size, img_size), dtype=bool))
            all_amaps.append(amap)

    return calculate_au_pro(all_masks, all_amaps, max_fpr=0.30, num_thresholds=500)

# MAIN EVALUATION FUNCTION
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

    log_f1    = compute_f1_max(log_labels, log_scores)
    struct_f1 = compute_f1_max(struct_labels, struct_scores)
    mean_f1   = (log_f1 + struct_f1) / 2.0

    print(f"[METRICS] Logical AUROC   : {log_auroc:.2f}%  (F1-max: {log_f1:.2f}%)")
    print(f"[METRICS] Structural AUROC: {struct_auroc:.2f}%  (F1-max: {struct_f1:.2f}%)")
    print(f"[METRICS] Mean AUROC      : {mean_auroc:.2f}%  (Mean F1: {mean_f1:.2f}%)")

    # Latency benchmark (batch_size=1)
    latency_ms = 0.0
    fps = 0.0
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
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        print(f"[LATENCY] Batch=1 Inference Latency: {latency_ms:.2f} ms/image ({fps:.1f} FPS)")

    spro = compute_spro(backbone, model, test_path, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
    print(f"[METRICS] sPRO (pixel-level): {spro:.2f}%")

    # Save Anomaly Maps if requested (TIFF float32 for official MVTec evaluator + PNG for visualization)
    if getattr(args, "save_maps", False):
        try:
            import tifffile
        except ImportError:
            tifffile = None

        save_root = ROOT / "outputs" / "anomaly_maps" / ("gct" if use_gct else "baseline") / category / "test"
        print(f"[MAPS] Saving anomaly maps to: {save_root.relative_to(ROOT)} ...")
        for sub_type in ["good", "logical_anomalies", "structural_anomalies"]:
            sub_dir = test_path / sub_type
            if not sub_dir.exists():
                continue
            out_sub_dir = save_root / sub_type
            out_sub_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted(list(sub_dir.glob("*.png")) + list(sub_dir.glob("*.jpg"))):
                _, amap = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size, crop_size=crop_size, gamma=gamma)
                
                # Get original image dimensions
                with Image.open(p) as raw_img:
                    orig_w, orig_h = raw_img.size

                # Resize anomaly map to match original image resolution
                amap_pil = Image.fromarray(amap.astype(np.float32))
                amap_orig = np.array(amap_pil.resize((orig_w, orig_h), resample=Image.BILINEAR))

                # 1. Save official TIFF float32 format (Required by MVTec evaluate_experiment.py)
                if tifffile is not None:
                    tifffile.imwrite(str(out_sub_dir / f"{p.stem}.tiff"), amap_orig.astype(np.float32))
                
                # 2. Save normalized PNG for easy human visualization
                amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
                amap_vis  = Image.fromarray((amap_norm * 255).astype(np.uint8))
                amap_vis.save(out_sub_dir / f"{p.stem}.png")

        print(f"[MAPS] Done saving maps (.tiff & .png) for {category}.")

    return {
        "category": category,
        "model": model_name,
        "use_gct": use_gct,
        "logical_auroc": float(log_auroc),
        "structural_auroc": float(struct_auroc),
        "mean_auroc": float(mean_auroc),
        "logical_f1": float(log_f1),
        "structural_f1": float(struct_f1),
        "mean_f1": float(mean_f1),
        "spro": float(spro),
        "latency_ms": float(latency_ms),
        "fps": float(fps)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViTill-GCT on MVTec LOCO AD")
    parser.add_argument("--config",    type=str, default="src/configs/loco_strict.json")
    parser.add_argument("--category",  type=str, default="breakfast_box")
    parser.add_argument("--use_gct",   action="store_true", help="Evaluate GCT model")
    parser.add_argument("--gamma",     type=float, default=None, help="GCT active score weight gamma (default: 1.0)")
    parser.add_argument("--save_maps", action="store_true", help="Save predicted anomaly maps to disk for official MVTec evaluation")
    parser.add_argument("--cpu",       action="store_true")
    args = parser.parse_args()
    evaluate(args)
