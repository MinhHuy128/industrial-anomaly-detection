"""
Evaluation script for ranking and adapter ablation models.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torchvision import transforms
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from src.models.vitill_gct import load_dinov2_register, extract_intermediate_features
from experiments.vlm_ablation.models.vitill_gct_ranking import ViTillGCTRanking
from experiments.vlm_ablation.train_vlm_ranking import build_model_for_mode, load_config


class MVTecLocoTestDataset:
    def __init__(self, category_root: Path, img_size: int = 448, crop_size: int = 392):
        self.test_dir = category_root / "test"
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

        self.samples = []
        for p in sorted((self.test_dir / "good").glob("*.*")):
            if p.suffix.lower() in [".png", ".jpg", ".bmp"]:
                self.samples.append((p, 0, "good"))

        log_dir = self.test_dir / "logical_anomalies"
        if log_dir.exists():
            for p in sorted(log_dir.glob("*.*")):
                if p.suffix.lower() in [".png", ".jpg", ".bmp"]:
                    self.samples.append((p, 1, "logical_anomalies"))

        str_dir = self.test_dir / "structural_anomalies"
        if str_dir.exists():
            for p in sorted(str_dir.glob("*.*")):
                if p.suffix.lower() in [".png", ".jpg", ".bmp"]:
                    self.samples.append((p, 1, "structural_anomalies"))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, defect_type = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label, defect_type, path.name


def evaluate_ranking_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    category: str,
    device_str: Optional[str] = None
) -> Dict:
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    mode = ckpt.get("mode", "gct_ranking")

    model, _ = build_model_for_mode(mode, cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    backbone = load_dinov2_register(device)
    backbone.eval()

    clip_model = None
    if model.use_adapter:
        try:
            import clip
            clip_model, _ = clip.load("ViT-B/16", device=device)
            clip_model.eval()
        except Exception as e:
            print(f"[WARN] CLIP backend unavailable: {e}")

    # Load dataset
    cat_root = data_root / category
    dataset = MVTecLocoTestDataset(
        category_root=cat_root,
        img_size=cfg.get("dataset", {}).get("img_size", 448),
        crop_size=cfg.get("dataset", {}).get("crop_size", 392)
    )

    gct_gamma = cfg.get("eval", {}).get("gct_gamma", 1.0)
    top_ratio = cfg.get("eval", {}).get("top_ratio", 0.01)
    target_layers = cfg.get("model", {}).get("target_layers", [2, 3, 4, 5, 6, 7, 8, 9])

    all_scores = []
    all_labels = []
    all_types = []
    all_names = []
    latencies = []

    for i in range(len(dataset)):
        x, label, defect_type, name = dataset[i]
        x_tensor = x.unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            feat_list, cls_token = extract_intermediate_features(backbone, x_tensor, target_layers, return_cls=True)
            vlm_feat = None
            if clip_model is not None:
                x_clip = F.interpolate(x_tensor, size=(224, 224), mode="bicubic", align_corners=False)
                vlm_feat = clip_model.encode_image(x_clip)
                vlm_feat = F.normalize(vlm_feat, dim=-1)

            en, de, gct_loss, _ = model(feat_list, cls_token=cls_token, vlm_feat=vlm_feat)
            score = model.compute_anomaly_score_from_maps(en, de, gct_loss, gamma=gct_gamma, top_ratio=top_ratio)

        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000.0)

        all_scores.append(float(score.item()))
        all_labels.append(int(label))
        all_types.append(defect_type)
        all_names.append(name)

    # Compute AUROC metrics
    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels)
    types_np = np.array(all_types)

    # logical
    log_mask = (types_np == "good") | (types_np == "logical_anomalies")
    if np.sum(types_np == "logical_anomalies") > 0 and np.sum(types_np == "good") > 0:
        log_auroc = float(roc_auc_score(labels_np[log_mask], scores_np[log_mask])) * 100.0
    else:
        log_auroc = 0.0

    # structural
    str_mask = (types_np == "good") | (types_np == "structural_anomalies")
    if np.sum(types_np == "structural_anomalies") > 0 and np.sum(types_np == "good") > 0:
        str_auroc = float(roc_auc_score(labels_np[str_mask], scores_np[str_mask])) * 100.0
    else:
        str_auroc = 0.0

    mean_auroc = (log_auroc + str_auroc) / 2.0
    avg_latency = float(np.mean(latencies[1:])) if len(latencies) > 1 else float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    results = {
        "category": category,
        "mode": mode,
        "logical_auroc": round(log_auroc, 2),
        "structural_auroc": round(str_auroc, 2),
        "mean_auroc": round(mean_auroc, 2),
        "latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1),
        "gate_weight": ckpt.get("final_gate_weight", 0.0),
        "total_test_samples": len(dataset),
        "raw_predictions": {
            "names": all_names,
            "types": all_types,
            "labels": all_labels,
            "scores": [round(s, 6) for s in all_scores]
        }
    }

    out_json = checkpoint_path.parent / "eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[{category}] {mode} - Logical: {log_auroc:.2f}% | Structural: {str_auroc:.2f}% | Mean: {mean_auroc:.2f}%")
    return results



def main():
    parser = argparse.ArgumentParser(description="Evaluate ViTill-GCT Ranking checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint best.pt")
    parser.add_argument("--data_path", type=str, default="data/mvtec_loco", help="Path to MVTec LOCO dataset root")
    parser.add_argument("--category", type=str, required=True, help="Category name")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    args = parser.parse_args()

    data_root = ROOT / args.data_path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    evaluate_ranking_checkpoint(
        checkpoint_path=ckpt_path,
        data_root=data_root,
        category=args.category,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
