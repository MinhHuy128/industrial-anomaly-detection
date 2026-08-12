"""
SCORE DISTRIBUTION ANALYSIS TOOL — EXPERIMENTAL RIGOR (GCT V2)

Measures distribution metrics (mean, std, min, max) of:
  - Score_patch (Local reconstruction error)
  - Score_gct   (Global CLS cosine distance)

across Good, Logical Anomaly, and Structural Anomaly image categories.

This analysis provides empirical evidence answering whether gamma=1.0
acts as a true balanced 1:1 weighting or if one stream dominates in magnitude.
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
from src.eval import compute_anomaly_map, image_score

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

def collect_distributions(category):
    ckpt_dir = ROOT / 'experiments' / 'gct'
    ckpt_path = ckpt_dir / f'gct_{category}_strict.pth'
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / f'gct_{category}_best.pth'
    if not ckpt_path.exists():
        print(f"[SKIP] Checkpoint not found for {category}")
        return None

    model = ViTillGCT(
        embed_dim=cfg['model']['embed_dim'],
        num_decoder_layers=cfg['model']['decoder_layers'],
        target_layers=target_layers
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt)
    model.eval()

    test_path = ROOT / cfg['dataset']['data_path'] / category / 'test'
    groups = {
        'Good': sorted(list((test_path / 'good').glob('*.png')) + list((test_path / 'good').glob('*.jpg'))),
        'Logical': sorted(list((test_path / 'logical_anomalies').glob('*.png')) + list((test_path / 'logical_anomalies').glob('*.jpg'))),
        'Structural': sorted(list((test_path / 'structural_anomalies').glob('*.png')) + list((test_path / 'structural_anomalies').glob('*.jpg'))),
    }

    results = {}
    with torch.no_grad():
        for group_name, paths in groups.items():
            if not paths:
                continue
            patch_scores, gct_scores = [], []
            for p in paths:
                img_t = transform(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
                feat_list, cls_token = extract_intermediate_features(backbone, img_t, target_layers)
                en, de, gct_loss = model(feat_list, cls_token)
                amap = compute_anomaly_map(en, de, out_size=img_size)
                patch_scores.append(image_score(amap))
                gct_scores.append(float(gct_loss.item()))

            p_arr, g_arr = np.array(patch_scores), np.array(gct_scores)
            results[group_name] = {
                'patch': {'mean': np.mean(p_arr), 'std': np.std(p_arr), 'min': np.min(p_arr), 'max': np.max(p_arr)},
                'gct':   {'mean': np.mean(g_arr), 'std': np.std(g_arr), 'min': np.min(g_arr), 'max': np.max(g_arr)},
            }
    return results

print("\n" + "=" * 90)
print("  SCORE MAGNITUDE & DISTRIBUTION ANALYSIS (Score_patch vs Score_gct)")
print("=" * 90)

for cat in categories:
    print(f"\n▶ CATEGORY: {cat.upper()}")
    dist = collect_distributions(cat)
    if not dist:
        continue
    for group, metrics in dist.items():
        p, g = metrics['patch'], metrics['gct']
        print(f"  [{group:10s}] Patch Score -> Mean: {p['mean']:.4f} | Std: {p['std']:.4f} | Min: {p['min']:.4f} | Max: {p['max']:.4f}")
        print(f"              GCT Score   -> Mean: {g['mean']:.4f} | Std: {g['std']:.4f} | Min: {g['min']:.4f} | Max: {g['max']:.4f}")

print("\n" + "=" * 90)
print("  ANALYSIS COMPLETED SUCCESSFULLY.")
print("=" * 90)
