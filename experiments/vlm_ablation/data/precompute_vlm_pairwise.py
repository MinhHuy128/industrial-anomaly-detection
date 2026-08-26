"""
Offline VLM Pairwise Precomputing & High-Confidence Filtering Script.
Precomputes pairwise relative anomaly comparisons and multi-modal feature embeddings
offline to avoid costly VLM calls during training loops.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.vlm_ablation.data.synthetic_anomaly_pairs import generate_synthetic_pair

CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors"
]


def load_clip_vlm_evaluator(device: torch.device):
    """Loads CLIP ViT-B/16 or ViT-L/14 model for zero-shot relative comparison."""
    try:
        import clip
        print("[VLM PRECOMPUTE] Loading OpenAI CLIP ViT-B/16 backend...")
        model, preprocess = clip.load("ViT-B/16", device=device)
        model.eval()
        return model, preprocess, "clip"
    except ImportError:
        print("[WARN] OpenAI clip package not found. Attempting transformers CLIP...")
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        return model, processor, "transformers_clip"


def evaluate_pair_clip(
    model,
    preprocess,
    backend_type: str,
    img_a: Image.Image,
    img_b: Image.Image,
    category: str,
    device: torch.device
) -> Tuple[int, float, str]:
    """Evaluates relative anomaly probability between two images using CLIP.

    Args:
        model: Loaded VLM model.
        preprocess: Image preprocess function or processor.
        backend_type: 'clip' or 'transformers_clip'.
        img_a: First PIL Image.
        img_b: Second PIL Image.
        category: Category name (e.g. 'pushpins').
        device: Torch device.

    Returns:
        preferred_anomaly (int): +1 if B is more anomalous, -1 if A is more anomalous.
        confidence (float): Relative confidence score [0.5, 1.0].
        rationale (str): Reason description.
    """
    normal_prompt = f"a high-quality flawless industrial photo of {category} with all parts complete and correct"
    anomaly_prompt = f"a defective damaged industrial photo of {category} with missing items, swapped parts, or structural errors"

    if backend_type == "clip":
        import clip
        t_a = preprocess(img_a).unsqueeze(0).to(device)
        t_b = preprocess(img_b).unsqueeze(0).to(device)
        text_tokens = clip.tokenize([normal_prompt, anomaly_prompt]).to(device)

        with torch.no_grad():
            feat_a = model.encode_image(t_a)
            feat_b = model.encode_image(t_b)
            text_feats = model.encode_text(text_tokens)

            feat_a = F.normalize(feat_a, dim=-1)
            feat_b = F.normalize(feat_b, dim=-1)
            text_feats = F.normalize(text_feats, dim=-1)

            # Contrastive similarity: Anomaly minus Normal for each image
            sim_a_norm = float(torch.sum(feat_a * text_feats[0:1]))
            sim_a_anom = float(torch.sum(feat_a * text_feats[1:2]))
            sim_b_norm = float(torch.sum(feat_b * text_feats[0:1]))
            sim_b_anom = float(torch.sum(feat_b * text_feats[1:2]))

            score_a = sim_a_anom - sim_a_norm
            score_b = sim_b_anom - sim_b_norm

            logit_scale = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 100.0

    else:
        processor = preprocess
        inputs_a = processor(text=[normal_prompt, anomaly_prompt], images=img_a, return_tensors="pt", padding=True).to(device)
        inputs_b = processor(text=[normal_prompt, anomaly_prompt], images=img_b, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out_a = model(**inputs_a)
            out_b = model(**inputs_b)
            logits_a = out_a.logits_per_image[0]  # [normal_logit, anomaly_logit]
            logits_b = out_b.logits_per_image[0]
            score_a = float((logits_a[1] - logits_a[0]).item())
            score_b = float((logits_b[1] - logits_b[0]).item())
            logit_scale = 1.0  # Already scaled in transformers output

    # Differential defect score
    score_diff = score_b - score_a
    scaled_diff = float(np.clip(score_diff * logit_scale, -20.0, 20.0))
    prob_b_higher = 1.0 / (1.0 + math.exp(-scaled_diff))

    if prob_b_higher >= 0.5:
        preferred = 1
        confidence = prob_b_higher
        rationale = f"Image B shows higher defect divergence (score diff {score_diff:+.4f}, prob {prob_b_higher:.3f})"
    else:
        preferred = -1
        confidence = 1.0 - prob_b_higher
        rationale = f"Image A shows higher defect divergence (score diff {score_diff:+.4f}, prob {confidence:.3f})"

    return preferred, confidence, rationale


def precompute_category_pairs(
    dataset_root: Path,
    category: str,
    output_dir: Path,
    num_pairs_per_image: int = 4,
    min_confidence: float = 0.75,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> Dict:
    """Generates synthetic pairs, evaluates VLM relative preferences, and saves filtered cache.

    Args:
        dataset_root: Root path of MVTec LOCO AD.
        category: Name of the category.
        output_dir: Destination directory for caching.
        num_pairs_per_image: Number of synthetic pairs to generate per normal image.
        min_confidence: Threshold for high-confidence pair retention.
        device: Compute device.
        seed: Random seed.

    Returns:
        Dictionary summarizing cached statistics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cat_dir = dataset_root / category / "train" / "good"
    if not cat_dir.exists():
        raise FileNotFoundError(f"Training dir not found: {cat_dir}")

    img_paths = sorted(
        list(cat_dir.glob("*.png")) +
        list(cat_dir.glob("*.jpg")) +
        list(cat_dir.glob("*.bmp"))
    )
    if len(img_paths) < 2:
        raise RuntimeError(f"Insufficient images in {cat_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_json = output_dir / f"{category}_pairwise_vlm_labels.json"
    embed_pt = output_dir / f"{category}_vlm_embeddings.pt"

    print(f"\n[PRECOMPUTE] Processing category: {category} ({len(img_paths)} train images)...")

    # Load VLM
    model, preprocess, backend = load_clip_vlm_evaluator(device)

    all_pairs = []
    filtered_pairs = []
    embeddings_dict = {}

    for idx, p_a in enumerate(img_paths):
        img_a = Image.open(p_a).convert("RGB")

        # Cache VLM embedding for image A
        if backend == "clip":
            with torch.no_grad():
                feat = model.encode_image(preprocess(img_a).unsqueeze(0).to(device))
                feat = F.normalize(feat, dim=-1).cpu()
                embeddings_dict[p_a.name] = feat
        else:
            inputs = preprocess(images=img_a, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inputs)
                feat = F.normalize(feat, dim=-1).cpu()
                embeddings_dict[p_a.name] = feat

        for k in range(num_pairs_per_image):
            other_idx = (idx + random.randint(1, len(img_paths) - 1)) % len(img_paths)
            img_b_base = Image.open(img_paths[other_idx]).convert("RGB")

            p1_img, p2_img, groundtruth_rank, meta = generate_synthetic_pair(img_a, img_b_base)
            pred_rank, confidence, rationale = evaluate_pair_clip(
                model=model,
                preprocess=preprocess,
                backend_type=backend,
                img_a=p1_img,
                img_b=p2_img,
                category=category,
                device=device
            )

            pair_record = {
                "pair_id": f"{category}_{idx:04d}_{k:02d}",
                "base_image": p_a.name,
                "groundtruth_rank": groundtruth_rank,
                "vlm_pred_rank": pred_rank,
                "confidence": round(float(confidence), 4),
                "is_agreement": bool(groundtruth_rank == pred_rank),
                "rationale": rationale,
                "meta": meta
            }
            all_pairs.append(pair_record)

            if confidence >= min_confidence and groundtruth_rank == pred_rank:
                filtered_pairs.append(pair_record)

    summary = {
        "category": category,
        "total_generated_pairs": len(all_pairs),
        "high_confidence_filtered_pairs": len(filtered_pairs),
        "filter_retention_rate": round(len(filtered_pairs) / max(1, len(all_pairs)), 4),
        "min_confidence_threshold": min_confidence,
        "cached_embeddings_count": len(embeddings_dict)
    }

    # Save to disk
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "filtered_pairs": filtered_pairs,
            "all_pairs": all_pairs
        }, f, indent=2)

    torch.save(embeddings_dict, embed_pt)

    print(f"  ✓ Saved {len(filtered_pairs)}/{len(all_pairs)} high-confidence pairs -> {cache_json.name}")
    print(f"  ✓ Cached {len(embeddings_dict)} VLM embeddings -> {embed_pt.name}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Precompute offline VLM pairwise rankings and embeddings.")
    parser.add_argument("--data_path", type=str, default="data/mvtec_loco", help="Path to MVTec LOCO dataset root")
    parser.add_argument("--output_dir", type=str, default="data/vlm_cache", help="Output directory for VLM cache")
    parser.add_argument("--categories", nargs="+", default=["pushpins", "screw_bag"], help="Categories to precompute")
    parser.add_argument("--pairs_per_img", type=int, default=4, help="Number of pairs per training sample")
    parser.add_argument("--min_conf", type=float, default=0.60, help="Confidence threshold to keep pairs (default: 0.60)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = ROOT / args.data_path
    out_dir = ROOT / args.output_dir
    device = torch.device(args.device)

    print("=" * 80)
    print("🚀 OFFLINE VLM PAIRWISE PRECOMPUTE & HIGH-CONFIDENCE FILTERING")
    print(f"Dataset Root: {data_root}")
    print(f"Output Cache: {out_dir}")
    print(f"Categories:   {args.categories}")
    print(f"Min Confidence: {args.min_conf}")
    print("=" * 80)

    for cat in args.categories:
        try:
            precompute_category_pairs(
                dataset_root=data_root,
                category=cat,
                output_dir=out_dir,
                num_pairs_per_image=args.pairs_per_img,
                min_confidence=args.min_conf,
                device=device,
                seed=args.seed
            )
        except Exception as e:
            print(f"[ERROR] Failed for category {cat}: {e}")


if __name__ == "__main__":
    main()
