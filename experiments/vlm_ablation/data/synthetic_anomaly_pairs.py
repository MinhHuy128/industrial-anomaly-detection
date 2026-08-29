"""
Synthetic anomaly pair generation utilities.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


def apply_patch_swap(
    img: Image.Image,
    source_img: Image.Image,
    patch_size_range: Tuple[int, int] = (32, 96)
) -> Tuple[Image.Image, Dict[str, Union[str, List[int]]]]:
    w, h = img.size
    pw = random.randint(patch_size_range[0], min(patch_size_range[1], w // 2))
    ph = random.randint(patch_size_range[0], min(patch_size_range[1], h // 2))

    src_x = random.randint(0, source_img.size[0] - pw)
    src_y = random.randint(0, source_img.size[1] - ph)
    tgt_x = random.randint(0, w - pw)
    tgt_y = random.randint(0, h - ph)

    patch = source_img.crop((src_x, src_y, src_x + pw, src_y + ph))
    perturbed = img.copy()
    perturbed.paste(patch, (tgt_x, tgt_y))

    meta = {
        "type": "patch_swap",
        "bbox": [tgt_x, tgt_y, tgt_x + pw, tgt_y + ph],
        "patch_size": [pw, ph]
    }
    return perturbed, meta


def apply_cutout(
    img: Image.Image,
    patch_size_range: Tuple[int, int] = (24, 80),
    fill_mode: str = "mean"
) -> Tuple[Image.Image, Dict[str, Union[str, List[int]]]]:
    w, h = img.size
    pw = random.randint(patch_size_range[0], min(patch_size_range[1], w // 3))
    ph = random.randint(patch_size_range[0], min(patch_size_range[1], h // 3))

    x = random.randint(0, w - pw)
    y = random.randint(0, h - ph)

    perturbed = img.copy()
    if fill_mode == "black":
        fill_color = (0, 0, 0)
        box_img = Image.new("RGB", (pw, ph), fill_color)
    elif fill_mode == "blur":
        box_img = img.crop((x, y, x + pw, y + ph)).filter(ImageFilter.GaussianBlur(radius=8))
    else:  # mean
        np_img = np.array(img)
        mean_c = tuple(np_img.mean(axis=(0, 1)).astype(int))
        box_img = Image.new("RGB", (pw, ph), mean_c)

    perturbed.paste(box_img, (x, y))
    meta = {
        "type": "cutout",
        "fill_mode": fill_mode,
        "bbox": [x, y, x + pw, y + ph]
    }
    return perturbed, meta


def generate_synthetic_pair(
    img_a: Image.Image,
    img_b: Image.Image,
    anomaly_prob: float = 0.5
) -> Tuple[Image.Image, Image.Image, int, Dict[str, Union[str, List[int]]]]:
    perturb_fn = random.choice([apply_patch_swap, apply_cutout])
    if perturb_fn == apply_patch_swap:
        perturbed_img, meta = apply_patch_swap(img_a, img_b)
    else:
        perturbed_img, meta = apply_cutout(img_a)

    if random.random() < anomaly_prob:
        return img_a, perturbed_img, 1, meta
    else:
        return perturbed_img, img_a, -1, meta



class MVTecLocoPairwiseDataset(Dataset):
    """Pairwise Training Dataset loading normal images and generating/loading synthetic pairs."""

    def __init__(
        self,
        category_root: Path,
        vlm_cache_path: Optional[Union[str, Path]] = None,
        vlm_embed_path: Optional[Union[str, Path]] = None,
        img_size: int = 448,
        crop_size: int = 392,
        synthetic_on_the_fly: bool = True
    ):
        """
        Args:
            category_root: Path to category directory containing train/good.
            vlm_cache_path: Optional path to precomputed VLM pairwise cache JSON.
            vlm_embed_path: Optional path to precomputed VLM embeddings .pt.
            img_size: Resize dimension before cropping.
            crop_size: Center crop dimension.
            synthetic_on_the_fly: If True, dynamically generates synthetic pairs during training.
        """
        self.category_root = Path(category_root)
        train_dir = self.category_root / "train" / "good"
        if not train_dir.exists():
            raise FileNotFoundError(f"Directory not found: {train_dir}")

        self.img_paths = sorted(
            list(train_dir.glob("*.png")) +
            list(train_dir.glob("*.jpg")) +
            list(train_dir.glob("*.bmp"))
        )
        if len(self.img_paths) < 2:
            raise RuntimeError(f"At least 2 training images required, found {len(self.img_paths)} in {train_dir}")

        self.synthetic_on_the_fly = synthetic_on_the_fly
        self.vlm_pairs = []
        if vlm_cache_path and Path(vlm_cache_path).exists():
            with open(vlm_cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                self.vlm_pairs = cached_data.get("filtered_pairs", [])

        self.vlm_embeddings = {}
        if vlm_embed_path and Path(vlm_embed_path).exists():
            self.vlm_embeddings = torch.load(vlm_embed_path, map_location="cpu")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a training item.

        Returns:
            x_normal: [3, 392, 392] Base normal image tensor for reconstruction.
            x_pair_1: [3, 392, 392] First image in comparison pair.
            x_pair_2: [3, 392, 392] Second image in comparison pair.
            target_rank: torch.Tensor scalar (+1 if pair_2 > pair_1 in anomaly score, else -1).
            vlm_feat: [512] Precomputed VLM visual embedding vector.
        """
        base_path = self.img_paths[idx]
        base_img = Image.open(base_path).convert("RGB")
        x_normal = self.transform(base_img)

        # Select a second image for pairing
        rand_idx = (idx + random.randint(1, len(self.img_paths) - 1)) % len(self.img_paths)
        second_img = Image.open(self.img_paths[rand_idx]).convert("RGB")

        if self.synthetic_on_the_fly:
            p1_img, p2_img, rank_label, _ = generate_synthetic_pair(base_img, second_img)
            x_p1 = self.transform(p1_img)
            x_p2 = self.transform(p2_img)
            target_rank = torch.tensor(rank_label, dtype=torch.float32)
        else:
            x_p1 = x_normal
            x_p2 = self.transform(second_img)
            target_rank = torch.tensor(0.0, dtype=torch.float32)

        vlm_feat = self.vlm_embeddings.get(base_path.name, torch.zeros(512, dtype=torch.float32))
        if isinstance(vlm_feat, torch.Tensor) and vlm_feat.dim() > 1:
            vlm_feat = vlm_feat.squeeze(0)

        return x_normal, x_p1, x_p2, target_rank, vlm_feat
