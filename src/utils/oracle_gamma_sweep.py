"""
WARNING: ORACLE SENSITIVITY ANALYSIS TOOL — FOR RESEARCH PURPOSES ONLY.

This script evaluates gamma values on the TEST SET to find the theoretical
upper bound (Oracle Upper Bound) of the GCT V2 architecture.
Results MUST NOT be reported as the primary benchmark metric to prevent Data Leakage.

Official benchmark uses gamma=1.0 (A-priori Equal Weighting).
See: src/configs/loco_strict.json -> eval.gct_gamma
"""
import sys
import json
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.vitill_gct import ViTillGCT, load_dinov2_register, extract_intermediate_features
from src.eval import compute_auroc, compute_anomaly_map, image_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using: {device}")

with open(ROOT / 'src/configs/loco_strict.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

categories = cfg['dataset']['categories']
target_layers = cfg['model'].get('target_layers', [2, 3, 4, 5, 6, 7, 8, 9])
img_size = cfg['dataset']['img_size']
crop_size = cfg['dataset']['crop_size']

transform = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("[BACKBONE] Loading DINOv2-Register...")
backbone = load_dinov2_register(device)

# Gamma sweep range for theoretical upper bound exploration
gammas = [0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

print("\n" + "=" * 80)
print("  ORACLE SENSITIVITY ANALYSIS MATRIX [0.8 -> 2.0]")
print("=" * 80)

best_per_cat = {}

for cat in categories:
    ckpt_dir = ROOT / 'experiments' / 'gct'
    ckpt_path = ckpt_dir / f'gct_{cat}_strict.pth'
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / f'gct_{cat}_best.pth'
    if not ckpt_path.exists():
        print(f"\n[SKIP] Checkpoint not found for {cat}")
        continue

    model = ViTillGCT(
        embed_dim=cfg['model']['embed_dim'],
        num_decoder_layers=cfg['model']['decoder_layers'],
        target_layers=target_layers
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt)
    model.eval()

    test_path = ROOT / cfg['dataset']['data_path'] / cat / 'test'
    good_paths   = sorted(list((test_path / 'good').glob('*.png')) + list((test_path / 'good').glob('*.jpg')))
    log_paths    = sorted(list((test_path / 'logical_anomalies').glob('*.png')) + list((test_path / 'logical_anomalies').glob('*.jpg')))
    struct_paths = sorted(list((test_path / 'structural_anomalies').glob('*.png')) + list((test_path / 'structural_anomalies').glob('*.jpg')))

    def collect_features(paths):
        patch_scores, gct_scores = [], []
        with torch.no_grad():
            for p in paths:
                img_t = transform(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
                feat_list, cls_token = extract_intermediate_features(backbone, img_t, target_layers)
                en, de, gct_loss = model(feat_list, cls_token)
                amap = compute_anomaly_map(en, de, out_size=img_size)
                patch_scores.append(image_score(amap))
                gct_scores.append(float(gct_loss.item()))
        return np.array(patch_scores), np.array(gct_scores)

    print(f"\n[EVALUATING ORACLE] Category: {cat.upper()}...")
    g_p, g_g = collect_features(good_paths)
    l_p, l_g = collect_features(log_paths)
    s_p, s_g = collect_features(struct_paths)

    best_mean = -1.0
    best_gamma = 0.8
    best_log_auroc = -1.0
    best_struct_auroc = -1.0

    print(f"  Gamma | Logical AUROC | Struct AUROC | Mean AUROC")
    print(f"  ------|---------------|--------------|-----------")

    for g in gammas:
        good_scores   = g_p + g * g_g
        log_scores    = l_p + g * l_g
        struct_scores = s_p + g * s_g

        l_labels = np.concatenate([np.zeros(len(g_p)), np.ones(len(l_p))])
        l_scores = np.concatenate([good_scores, log_scores])
        auroc_l  = compute_auroc(l_labels, l_scores)

        s_labels = np.concatenate([np.zeros(len(g_p)), np.ones(len(s_p))])
        s_scores = np.concatenate([good_scores, struct_scores])
        auroc_s  = compute_auroc(s_labels, s_scores)

        mean_a   = (auroc_l + auroc_s) / 2.0
        print(f"  {g:5.1f} | {auroc_l:12.2f}% | {auroc_s:11.2f}% | {mean_a:9.2f}%")

        if mean_a > best_mean:
            best_mean = mean_a
            best_gamma = g
            best_log_auroc = auroc_l
            best_struct_auroc = auroc_s

    best_per_cat[cat] = {
        'best_gamma': best_gamma,
        'log_auroc': best_log_auroc,
        'struct_auroc': best_struct_auroc,
        'mean_auroc': best_mean
    }

print("\n" + "=" * 80)
print("  SUMMARY: ORACLE UPPER BOUND SCORES PER CATEGORY")
print("=" * 80)
overall_log, overall_struct, overall_mean = [], [], []
for cat, res in best_per_cat.items():
    print(f"  {cat:20s} | Peak Gamma: {res['best_gamma']:3.1f} | Logical: {res['log_auroc']:6.2f}% | Struct: {res['struct_auroc']:6.2f}% | Mean: {res['mean_auroc']:6.2f}%")
    overall_log.append(res['log_auroc'])
    overall_struct.append(res['struct_auroc'])
    overall_mean.append(res['mean_auroc'])

if overall_mean:
    print("-" * 80)
    print(f"  OVERALL ORACLE UPPER BOUND: Logical: {np.mean(overall_log):.2f}% | Struct: {np.mean(overall_struct):.2f}% | Mean: {np.mean(overall_mean):.2f}%")
    print("=" * 80)
    print("\n[NOTE] These are ORACLE UPPER BOUND values — NOT the official benchmark.")
    print("[NOTE] Official benchmark uses gamma=1.0 (equal weighting). See loco_strict.json.")
