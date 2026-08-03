import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import time
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.dinomaly_baseline import DinomalyBaseline
from src.models.dinomaly_gct import DinomalyGCT

# ─────────────────────────────────────────────────────────────
# HELPER: Load DINOv2 Backbone
# ─────────────────────────────────────────────────────────────
def load_dinov2_backbone(device):
    print("[BACKBONE] Loading pretrained DINOv2 ViT-B/14 encoder...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print("[BACKBONE] Successfully loaded and frozen DINOv2 backbone.")
    return backbone

# ─────────────────────────────────────────────────────────────
# HELPER: Compute AUROC via scikit-learn (or manual fallback)
# ─────────────────────────────────────────────────────────────
def compute_auroc(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores) * 100.0
    except ImportError:
        labels = np.array(labels)
        scores = np.array(scores)
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 50.0
        rank_sum = sum(
            np.sum(p > neg) + 0.5 * np.sum(p == neg)
            for p in pos
        )
        return (rank_sum / (len(pos) * len(neg))) * 100.0

# ─────────────────────────────────────────────────────────────
# HELPER: Compute anomaly score for one image (patch-level MSE)
# ─────────────────────────────────────────────────────────────
def get_anomaly_score(backbone, model, img_tensor, device, is_gct):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        features = backbone.forward_features(img_tensor)
        patch_tokens = features["x_norm_patchtokens"]   # [1, 1024, 768]
        cls_token    = features["x_norm_clstoken"]       # [1, 768]

        if is_gct:
            out = model(patch_tokens, dinov2_cls_token=cls_token)
            rec = out["reconstructed_patches"]
        else:
            rec = model(patch_tokens)

        # Patch-level MSE: mean over embed dim, then max over patches
        patch_errors = F.mse_loss(rec, patch_tokens, reduction="none").mean(dim=-1)  # [1, 1024]
        score = patch_errors.max().item()   # image-level anomaly score = max patch error
    return score

# ─────────────────────────────────────────────────────────────
# HELPER: Evaluate one folder (good=0, anomaly=1)
# ─────────────────────────────────────────────────────────────
def evaluate_folder(folder_path, label, backbone, model, device, transform, is_gct):
    from PIL import Image
    scores, labels = [], []
    if not folder_path.exists():
        print(f"  [SKIP] Folder not found: {folder_path}")
        return scores, labels

    img_paths = sorted(
        list(folder_path.glob("*.png")) +
        list(folder_path.glob("*.jpg")) +
        list(folder_path.glob("*.bmp"))
    )
    if len(img_paths) == 0:
        print(f"  [SKIP] No images in: {folder_path}")
        return scores, labels

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        img_t = transform(img).unsqueeze(0)
        score = get_anomaly_score(backbone, model, img_t, device, is_gct)
        scores.append(score)
        labels.append(label)
    return scores, labels

# ─────────────────────────────────────────────────────────────
# HELPER: Compute approximate sPRO using pixel-level anomaly map
# ─────────────────────────────────────────────────────────────
def compute_spro_approx(backbone, model, test_path, device, transform, is_gct, img_size):
    """
    Approximate sPRO via pixel-level anomaly map overlap with GT masks.
    Requires test/<category>/ground_truth/ folder (MVTec LOCO AD standard).

    If ground truth masks are unavailable, returns -1.0 to indicate not computed.
    """
    from PIL import Image
    gt_root = test_path / "ground_truth"
    anomaly_dirs = ["logical_anomalies", "structural_anomalies"]

    all_gt_masks = []
    all_pred_maps = []

    for anom_type in anomaly_dirs:
        anom_dir = test_path / anom_type
        gt_dir   = gt_root / anom_type
        if not anom_dir.exists() or not gt_dir.exists():
            continue

        img_paths = sorted(
            list(anom_dir.glob("*.png")) +
            list(anom_dir.glob("*.jpg"))
        )
        for p in img_paths:
            # GT mask
            mask_name = p.stem + ".png"
            mask_path = gt_dir / mask_name
            if not mask_path.exists():
                continue

            gt_mask = np.array(Image.open(mask_path).convert("L").resize(
                (img_size, img_size))) > 0  # binary

            # Anomaly Map from model
            img = Image.open(p).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = backbone.forward_features(img_t)
                patch_tokens = features["x_norm_patchtokens"]
                cls_token    = features["x_norm_clstoken"]

                if is_gct:
                    out = model(patch_tokens, dinov2_cls_token=cls_token)
                    rec = out["reconstructed_patches"]
                else:
                    rec = model(patch_tokens)

                # Patch-level error map -> upscale to img_size
                patch_err = F.mse_loss(rec, patch_tokens, reduction="none").mean(dim=-1)  # [1, 1024]
                h = w = int(patch_err.shape[-1] ** 0.5)
                anomaly_map = patch_err.reshape(1, 1, h, w)
                anomaly_map = F.interpolate(anomaly_map, size=(img_size, img_size), mode="bilinear", align_corners=False)
                anomaly_map = anomaly_map.squeeze().cpu().numpy()

            all_gt_masks.append(gt_mask.flatten())
            all_pred_maps.append(anomaly_map.flatten())

    if len(all_gt_masks) == 0:
        return -1.0

    all_gt   = np.concatenate(all_gt_masks)
    all_pred = np.concatenate(all_pred_maps)

    # Normalized overlap at FPR threshold 0.30 (standard sPRO convention)
    fpr_thresholds = np.linspace(0, 0.30, 100)
    spro_values = []
    for fpr_limit in fpr_thresholds:
        thresh = np.percentile(all_pred, (1 - fpr_limit) * 100)
        pred_binary = all_pred >= thresh
        tp = np.sum(pred_binary & all_gt)
        fn = np.sum(~pred_binary & all_gt)
        if (tp + fn) > 0:
            spro_values.append(tp / (tp + fn))
        else:
            spro_values.append(0.0)

    return float(np.mean(spro_values)) * 100.0

# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────
def evaluate(args):
    # ── Load Config ──────────────────────────────────────────
    path = Path(args.config)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except ImportError:
            json_path = path.with_suffix(".json")
            with open(json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else cfg["project"]["device"])
    model_name = "DINOMALY + GCT" if args.use_gct else "DINOMALY BASELINE"
    img_size   = cfg["dataset"]["img_size"]
    print(f"[EVAL] Model: {model_name} | Category: {args.category} | Device: {device}")

    # ── Load Model ───────────────────────────────────────────
    model_prefix = "gct" if args.use_gct else "baseline"
    if args.use_gct:
        model = DinomalyGCT(
            embed_dim=cfg["model"]["embed_dim"],
            num_decoder_layers=cfg["model"]["decoder_layers"]
        ).to(device)
    else:
        model = DinomalyBaseline(
            embed_dim=cfg["model"]["embed_dim"],
            num_decoder_layers=cfg["model"]["decoder_layers"]
        ).to(device)

    # ── Load Checkpoint ──────────────────────────────────────
    subfolder = "gct" if args.use_gct else "baseline"
    possible_ckpts = [
        ROOT / cfg["train"]["save_dir"] / subfolder / f"{model_prefix}_{args.category}_best.pth",
        ROOT / cfg["train"]["save_dir"] / f"{model_prefix}_{args.category}_best.pth",
    ]
    ckpt_loaded = False
    for ckpt_path in possible_ckpts:
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[CHECKPOINT] Loaded: {ckpt_path.relative_to(ROOT)}")
            ckpt_loaded = True
            break
    if not ckpt_loaded:
        raise FileNotFoundError(
            f"[ERROR] No checkpoint found for '{args.category}'. "
            f"Please run train.py first."
        )
    model.eval()

    # ── Load DINOv2 Backbone ─────────────────────────────────
    backbone = load_dinov2_backbone(device)

    # ── Measure FULL PIPELINE Latency (DINOv2 + Decoder) ────
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from PIL import Image as PILImage
    # Use a real dummy image (or first available good image) for latency
    good_dir = ROOT / cfg["dataset"]["data_path"] / args.category / "test" / "good"
    sample_imgs = sorted(list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg")))
    if len(sample_imgs) > 0:
        dummy_pil   = PILImage.open(sample_imgs[0]).convert("RGB")
        dummy_img_t = transform(dummy_pil).unsqueeze(0).to(device)
    else:
        # Fallback: synthetic noise image (clearly labeled)
        print("[LATENCY] Warning: No real test images found, using synthetic input for latency benchmark.")
        dummy_img_t = torch.randn(1, 3, img_size, img_size).to(device)

    # Warmup
    for _ in range(10):
        get_anomaly_score(backbone, model, dummy_img_t, device, args.use_gct)

    # Benchmark (full pipeline: DINOv2 + Decoder)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    NUM_RUNS = 50
    for _ in range(NUM_RUNS):
        get_anomaly_score(backbone, model, dummy_img_t, device, args.use_gct)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / NUM_RUNS * 1000.0

    # ── Check Test Dataset Existence ─────────────────────────
    test_path = ROOT / cfg["dataset"]["data_path"] / args.category / "test"
    if not test_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Test dataset not found at: {test_path}\n"
            f"        Please ensure MVTec LOCO AD dataset is mounted at: {ROOT / cfg['dataset']['data_path']}"
        )

    # ── Evaluate Logical AUROC ───────────────────────────────
    good_s,  good_l  = evaluate_folder(test_path / "good",                 0, backbone, model, device, transform, args.use_gct)
    log_s,   log_l   = evaluate_folder(test_path / "logical_anomalies",    1, backbone, model, device, transform, args.use_gct)
    struct_s,struct_l= evaluate_folder(test_path / "structural_anomalies", 1, backbone, model, device, transform, args.use_gct)

    if len(good_s) == 0:
        raise RuntimeError("[ERROR] No 'good' test images found. Check dataset structure.")
    if len(log_s) == 0:
        raise RuntimeError("[ERROR] No logical_anomalies images found. Check dataset structure.")
    if len(struct_s) == 0:
        raise RuntimeError("[ERROR] No structural_anomalies images found. Check dataset structure.")

    auroc_logical    = compute_auroc(good_l + log_l,    good_s + log_s)
    auroc_structural = compute_auroc(good_l + struct_l, good_s + struct_s)
    auroc_mean       = (auroc_logical + auroc_structural) / 2.0

    # ── Compute sPRO (pixel-level, with GT masks) ────────────
    spro_score = compute_spro_approx(backbone, model, test_path, device, transform, args.use_gct, img_size)
    spro_label = f"{spro_score:.2f} %" if spro_score >= 0 else "N/A (GT masks not found)"

    # ── Print Results ────────────────────────────────────────
    print("=" * 65)
    print(f"  MODEL EVALUATION METRICS ({model_name}): {args.category.upper()}")
    print("=" * 65)
    print(f"  • Logical Anomaly AUROC   : {auroc_logical:.2f} %")
    print(f"  • Structural Anomaly AUROC: {auroc_structural:.2f} %")
    print(f"  • Mean AUROC Score        : {auroc_mean:.2f} %")
    print(f"  • sPRO (pixel-level)      : {spro_label}")
    print(f"  • Full Pipeline Latency   : {latency_ms:.2f} ms / image (FPS: {1000.0/latency_ms:.1f})")
    print(f"    (includes DINOv2 ViT-B/14 + Bottleneck + Decoder)")
    print(f"  • Status                  : SUCCESSFUL - Evaluated")
    print("=" * 65)

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dinomaly Baseline / GCT on MVTec LOCO AD")
    parser.add_argument("--config",   type=str, default="src/configs/baseline_loco.json")
    parser.add_argument("--category", type=str, default="breakfast_box")
    parser.add_argument("--use_gct",  action="store_true", help="Enable GCT Model evaluation")
    parser.add_argument("--cpu",      action="store_true", help="Force CPU mode")

    args = parser.parse_args()
    evaluate(args)
