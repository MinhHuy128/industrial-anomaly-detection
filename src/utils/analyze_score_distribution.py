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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    backbone = load_dinov2_register(device)

    for cat in categories:
        dist = collect_distributions(cat, cfg, target_layers, transform, backbone, device, img_size)
        if not dist:
            continue
        print(f"\nCategory: {cat}")
        for group, metrics in dist.items():
            p, g = metrics['patch'], metrics['gct']
            print(f"  [{group:10s}] Patch -> mean: {p['mean']:.4f}, std: {p['std']:.4f}")
            print(f"              GCT   -> mean: {g['mean']:.4f}, std: {g['std']:.4f}")


def collect_distributions(category, cfg, target_layers, transform, backbone, device, img_size):
    ckpt_dir = ROOT / 'experiments' / 'gct'
    ckpt_path = ckpt_dir / f'gct_{category}_strict.pth'
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / f'gct_{category}_best.pth'
    if not ckpt_path.exists():
        print(f"skip {category}: checkpoint not found")
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


if __name__ == '__main__':
    main()
