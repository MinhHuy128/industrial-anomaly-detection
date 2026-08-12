import sys
import json
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.vitill_gct import ViTillGCT, ViTillBaseline, load_dinov2_register, extract_intermediate_features
from src.eval import compute_anomaly_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using: {device}")

with open(ROOT / 'src/configs/loco_strict.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# 🎯 Chạy cho toàn bộ 5 Categories
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

out_dir = Path('docs/figures')
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[EXPORTING HEATMAPS] Target directory: {out_dir.resolve()}")

total_saved = 0

for cat in categories:
    print(f"\n========================================================")
    print(f"  PROCESSING CATEGORY: {cat.upper()}")
    print(f"========================================================")
    
    gct_ckpt  = ROOT / 'experiments/gct' / f'gct_{cat}_strict.pth'
    base_ckpt = ROOT / 'experiments/baseline' / f'baseline_{cat}_strict.pth'

    if not gct_ckpt.exists():
        print(f"  [WARN] GCT Checkpoint missing: {gct_ckpt}")
        continue

    model_gct = ViTillGCT(embed_dim=768, num_decoder_layers=8, target_layers=target_layers).to(device)
    ckpt_g = torch.load(gct_ckpt, map_location=device)
    model_gct.load_state_dict(ckpt_g['model_state'])
    model_gct.eval()

    has_base = base_ckpt.exists()
    if has_base:
        model_base = ViTillBaseline(embed_dim=768, num_decoder_layers=8, target_layers=target_layers).to(device)
        ckpt_b = torch.load(base_ckpt, map_location=device)
        model_base.load_state_dict(ckpt_b['model_state'])
        model_base.eval()

    test_path = ROOT / cfg['dataset']['data_path'] / cat / 'test'
    
    # 🎯 Tạo ảnh cho cả Logical Anomalies lẫn Structural Anomalies
    for anomaly_type in ['logical_anomalies', 'structural_anomalies']:
        anom_dir = test_path / anomaly_type
        gt_dir   = ROOT / cfg['dataset']['data_path'] / cat / 'ground_truth' / anomaly_type
        if not anom_dir.exists():
            continue

        # Lấy 5 ảnh mẫu cho mỗi loại bất thường
        img_paths = sorted(list(anom_dir.glob('*.png')) + list(anom_dir.glob('*.jpg')))[:5]
        
        for idx, p in enumerate(img_paths):
            stem = p.stem
            mask_p = gt_dir / stem / f"{stem}.png"
            if not mask_p.exists():
                mask_p = gt_dir / f"{stem}.png"

            if mask_p.exists():
                gt_mask = np.array(Image.open(mask_p).convert('L').resize((crop_size, crop_size)))
            else:
                gt_mask = np.zeros((crop_size, crop_size))

            raw_img = Image.open(p).convert('RGB').resize((img_size, img_size), Image.BICUBIC)
            w, h = raw_img.size
            left = (w - crop_size) // 2
            top  = (h - crop_size) // 2
            raw_crop = raw_img.crop((left, top, left + crop_size, top + crop_size))

            img_t = transform(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                feats_g, cls_g = extract_intermediate_features(backbone, img_t, target_layers)
                en_g, de_g, _ = model_gct(feats_g, cls_g)
                amap_g = compute_anomaly_map(en_g, de_g, out_size=crop_size)

                if has_base:
                    feats_b, _ = extract_intermediate_features(backbone, img_t, target_layers)
                    en_b, de_b = model_base(feats_b)
                    amap_b = compute_anomaly_map(en_b, de_b, out_size=crop_size)

            num_cols = 4 if has_base else 3
            fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

            axes[0].imshow(raw_crop); axes[0].set_title("Input RGB Image"); axes[0].axis('off')
            axes[1].imshow(gt_mask, cmap='gray'); axes[1].set_title("Ground Truth Mask"); axes[1].axis('off')
            
            if has_base:
                axes[2].imshow(raw_crop); axes[2].imshow(amap_b, cmap='jet', alpha=0.5); axes[2].set_title("Baseline Heatmap"); axes[2].axis('off')
                axes[3].imshow(raw_crop); axes[3].imshow(amap_g, cmap='jet', alpha=0.5); axes[3].set_title("GCT V2 Heatmap"); axes[3].axis('off')
            else:
                axes[2].imshow(raw_crop); axes[2].imshow(amap_g, cmap='jet', alpha=0.5); axes[2].set_title("GCT V2 Heatmap"); axes[2].axis('off')

            plt.tight_layout()
            save_file = out_dir / f"heatmap_{cat}_{anomaly_type}_{idx+1}.png"
            plt.savefig(save_file, dpi=200, bbox_inches='tight')
            plt.close()
            total_saved += 1
            print(f"  [SAVED] {save_file.name}")

print(f"\n========================================================")
print(f"  SUCCESSFULLY GENERATED {total_saved} HEATMAP IMAGES IN docs/figures/")
print(f"========================================================")
