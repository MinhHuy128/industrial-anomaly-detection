"""
Evaluation script for ViTill-GCT (Paper-Strict Branch).

Anomaly scoring faithful to Dinomaly paper:
  - Per-patch cosine distance: a_i = 1 - cos_sim(en_i, de_i)
  - Average over feature groups → anomaly map [H, W]
  - Upsample to input resolution (448×448)
  - Gaussian smoothing (σ=4)
  - Image-level score = max patch score (top 1% percentile for speed)
  - Separate AUROC for Logical / Structural anomalies (MVTec LOCO AD format)
  - Pixel-level sPRO via GT mask overlap
  - Full pipeline latency (DINOv2-Register + Bottleneck + Decoder)
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
# AUROC
# ─────────────────────────────────────────────────────────────────────────────
def compute_auroc(labels, scores):
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
# ANOMALY MAP (paper formula: cosine distance)
# ─────────────────────────────────────────────────────────────────────────────
def compute_anomaly_map(en_list, de_list, out_size: int = 448) -> np.ndarray:
    """
    Per-patch cosine distance, averaged over feature groups, upsampled to out_size.
    Paper Eq.4: a(i) = 1 - cos_sim(f_enc(i), f_dec(i))
    """
    anomaly_map = torch.zeros(1, 1, 1, 1, device=en_list[0].device)
    for en, de in zip(en_list, de_list):
        # en, de: [B, C, H, W] — compute cosine distance per spatial position
        a_map = 1.0 - F.cosine_similarity(en, de, dim=1, eps=1e-8)  # [B, H, W]
        a_map = a_map.unsqueeze(1)                                    # [B, 1, H, W]
        a_map = F.interpolate(a_map, size=(out_size, out_size),
                               mode='bilinear', align_corners=False)
        anomaly_map = anomaly_map + a_map

    anomaly_map = anomaly_map / len(en_list)  # average over groups
    anomaly_map = anomaly_map.squeeze().cpu().numpy()  # [H, W]

    # Gaussian smooth (σ=4, same as typical anomaly detection practice)
    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    return anomaly_map


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE-LEVEL SCORE (max or top-1% mean)
# ─────────────────────────────────────────────────────────────────────────────
def image_score(anomaly_map: np.ndarray, max_ratio: float = 0.01) -> float:
    """Top max_ratio fraction average (paper uses max = top 1%)."""
    flat = anomaly_map.flatten()
    k    = max(1, int(len(flat) * max_ratio))
    return float(np.sort(flat)[-k:].mean())


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ON ONE IMAGE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_one(backbone, model, img_path: Path, transform, device, use_gct: bool,
              target_layers: list, out_size: int) -> tuple:
    """Returns (image_anomaly_score, anomaly_map_np)."""
    img   = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    feat_list, cls_token = extract_intermediate_features(backbone, img_t, target_layers)

    if use_gct:
        en, de, _ = model(feat_list, cls_token)
    else:
        en, de = model(feat_list)

    amap  = compute_anomaly_map(en, de, out_size=out_size)
    score = image_score(amap)
    return score, amap


# ─────────────────────────────────────────────────────────────────────────────
# SPRO PIXEL-LEVEL
# ─────────────────────────────────────────────────────────────────────────────
def compute_spro(backbone, model, test_path: Path, transform, device,
                 use_gct: bool, target_layers: list, img_size: int) -> float:
    """
    Approximate sPRO: pixel-level overlap between predicted anomaly map and GT masks.
    MVTec LOCO AD GT layout: ground_truth/<anom_type>/<stem>/<stem>.png
    """
    gt_root     = test_path.parent / "ground_truth"
    anom_types  = ["logical_anomalies", "structural_anomalies"]
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

            _, amap = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size)
            all_gt.append(gt_mask.flatten())
            all_pred.append(amap.flatten())

    if len(all_gt) == 0:
        return -1.0

    all_gt   = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    spro_vals = []
    for fpr_limit in np.linspace(0, 0.30, 100):
        thresh      = np.percentile(all_pred, (1 - fpr_limit) * 100)
        pred_bin    = all_pred >= thresh
        tp = np.sum(pred_bin & all_gt)
        fn = np.sum(~pred_bin & all_gt)
        spro_vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0.)

    return float(np.mean(spro_vals)) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(args):
    # ── Config ──────────────────────────────────────────────────────────────
    path = Path(args.config)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    device         = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                                  else cfg["project"]["device"])
    use_gct        = args.use_gct
    category       = args.category
    img_size       = cfg["dataset"]["img_size"]  # 448
    target_layers  = cfg["model"].get("target_layers", [2, 3, 4, 5, 6, 7, 8, 9])
    model_name     = "DINOMALY + GCT (Paper-Strict)" if use_gct else "DINOMALY BASELINE (Paper-Strict)"

    print(f"[EVAL] Model: {model_name}  |  Category: {category}  |  Device: {device}")

    # ── Build Model ─────────────────────────────────────────────────────────
    embed_dim    = cfg["model"]["embed_dim"]
    num_decoder  = cfg["model"]["decoder_layers"]

    if use_gct:
        model = ViTillGCT(embed_dim=embed_dim, num_decoder_layers=num_decoder,
                          target_layers=target_layers).to(device)
    else:
        model = ViTillBaseline(embed_dim=embed_dim, num_decoder_layers=num_decoder,
                               target_layers=target_layers).to(device)

    # ── Load Checkpoint ─────────────────────────────────────────────────────
    prefix    = "gct" if use_gct else "baseline"
    subfolder = "gct" if use_gct else "baseline"
    ckpt_path = ROOT / cfg["train"]["save_dir"] / subfolder / f"{prefix}_{category}_strict.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Checkpoint not found: {ckpt_path}\n"
            f"        Run: python src/train.py --category {category}"
            + (" --use_gct" if use_gct else "")
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[CHECKPOINT] Loaded: {ckpt_path.relative_to(ROOT)}")
    model.eval()

    # ── Load DINOv2-Register ─────────────────────────────────────────────────
    backbone = load_dinov2_register(device)

    # ── Transform (same as training: Resize(448,448) → CenterCrop(392)) ────
    crop_size = cfg["dataset"]["crop_size"]   # 392
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # ── Latency Benchmark (Full Pipeline) ────────────────────────────────────
    test_path  = ROOT / cfg["dataset"]["data_path"] / category / "test"
    good_dir   = test_path / "good"
    sample_imgs = sorted(list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg")))

    if len(sample_imgs) == 0:
        raise FileNotFoundError(f"[ERROR] No 'good' test images found in: {good_dir}")

    # Warmup (10 runs to initialize CUDA kernels)
    for _ in range(10):
        infer_one(backbone, model, sample_imgs[0], transform, device, use_gct, target_layers, img_size)

    # Benchmark 50 runs
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        infer_one(backbone, model, sample_imgs[0], transform, device, use_gct, target_layers, img_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 50 * 1000.0

    # ── Check Dataset ────────────────────────────────────────────────────────
    if not test_path.exists():
        raise FileNotFoundError(f"[ERROR] Test path not found: {test_path}")

    # ── Collect Scores ────────────────────────────────────────────────────────
    good_s, good_l = [], []
    for p in sorted(list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))):
        s, _ = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size)
        good_s.append(s); good_l.append(0)

    log_dir    = test_path / "logical_anomalies"
    struct_dir = test_path / "structural_anomalies"
    log_s, log_l, struct_s, struct_l = [], [], [], []

    for p in sorted(list(log_dir.glob("*.png")) + list(log_dir.glob("*.jpg"))):
        s, _ = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size)
        log_s.append(s); log_l.append(1)

    for p in sorted(list(struct_dir.glob("*.png")) + list(struct_dir.glob("*.jpg"))):
        s, _ = infer_one(backbone, model, p, transform, device, use_gct, target_layers, img_size)
        struct_s.append(s); struct_l.append(1)

    if not good_s:
        raise RuntimeError("[ERROR] No 'good' test images found.")
    if not log_s:
        raise RuntimeError("[ERROR] No logical_anomalies images found.")
    if not struct_s:
        raise RuntimeError("[ERROR] No structural_anomalies images found.")

    auroc_logical    = compute_auroc(good_l + log_l,    good_s + log_s)
    auroc_structural = compute_auroc(good_l + struct_l, good_s + struct_s)
    auroc_mean       = (auroc_logical + auroc_structural) / 2.0

    # ── sPRO ─────────────────────────────────────────────────────────────────
    spro = compute_spro(backbone, model, test_path, transform, device,
                        use_gct, target_layers, img_size)
    spro_label = f"{spro:.2f} %" if spro >= 0 else "N/A (GT masks not found)"

    # ── Print Results ─────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  MODEL EVALUATION METRICS ({model_name}): {category.upper()}")
    print("=" * 70)
    print(f"  • Logical Anomaly AUROC   : {auroc_logical:.2f} %")
    print(f"  • Structural Anomaly AUROC: {auroc_structural:.2f} %")
    print(f"  • Mean AUROC Score        : {auroc_mean:.2f} %")
    print(f"  • sPRO (pixel-level)      : {spro_label}")
    print(f"  • Full Pipeline Latency   : {latency_ms:.2f} ms / image (FPS: {1000/latency_ms:.1f})")
    print(f"    (DINOv2-Register ViT-B/14 + Bottleneck + Decoder)")
    print(f"  • Status                  : SUCCESSFUL - Evaluated")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="src/configs/loco_strict.json")
    parser.add_argument("--category", type=str, default="breakfast_box")
    parser.add_argument("--use_gct",  action="store_true")
    parser.add_argument("--cpu",      action="store_true")
    args = parser.parse_args()
    evaluate(args)
