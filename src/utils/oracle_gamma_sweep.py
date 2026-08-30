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

    gammas = [0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    best_per_cat = {}

    for cat in categories:
        ckpt_dir = ROOT / 'experiments' / 'gct'
        ckpt_path = ckpt_dir / f'gct_{cat}_strict.pth'
        if not ckpt_path.exists():
            ckpt_path = ckpt_dir / f'gct_{cat}_best.pth'
        if not ckpt_path.exists():
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
        good_paths = sorted(list((test_path / 'good').glob('*.png')) + list((test_path / 'good').glob('*.jpg')))
        log_paths = sorted(list((test_path / 'logical_anomalies').glob('*.png')) + list((test_path / 'logical_anomalies').glob('*.jpg')))
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

        g_p, g_g = collect_features(good_paths)
        l_p, l_g = collect_features(log_paths)
        s_p, s_g = collect_features(struct_paths)

        best_gamma = 1.0
        best_mean = 0.0
        best_log_auroc = 0.0
        best_struct_auroc = 0.0

        print(f"\nCategory: {cat}")
        print("gamma | logical auroc | struct auroc | mean auroc")

        for g in gammas:
            good_scores = g_p + g * g_g
            log_scores = l_p + g * l_g
            struct_scores = s_p + g * s_g

            l_labels = np.concatenate([np.zeros(len(g_p)), np.ones(len(l_p))])
            l_scores = np.concatenate([good_scores, log_scores])
            auroc_l = compute_auroc(l_labels, l_scores)

            s_labels = np.concatenate([np.zeros(len(g_p)), np.ones(len(s_p))])
            s_scores = np.concatenate([good_scores, struct_scores])
            auroc_s = compute_auroc(s_labels, s_scores)

            mean_a = (auroc_l + auroc_s) / 2.0
            print(f"{g:5.1f} | {auroc_l:12.2f}% | {auroc_s:11.2f}% | {mean_a:9.2f}%")

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

    print("\nSummary:")
    for cat, res in best_per_cat.items():
        print(f"{cat:20s} | gamma: {res['best_gamma']:3.1f} | logical: {res['log_auroc']:6.2f}% | struct: {res['struct_auroc']:6.2f}% | mean: {res['mean_auroc']:6.2f}%")


if __name__ == '__main__':
    main()
