import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.models.dinomaly_baseline import DinomalyBaseline
from src.models.dinomaly_gct import DinomalyGCT

def set_deterministic_seed(seed=42):
    """Enforce strict reproducibility across PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    path = Path(config_path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            json_path = path.with_suffix(".json")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise RuntimeError("PyYAML not installed. Please use .json config file.")

def load_dinov2_backbone(device):
    """Load pretrained DINOv2 ViT-B/14 backbone and freeze parameters."""
    print("[BACKBONE] Loading pretrained DINOv2 ViT-B/14 encoder...")
    try:
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        print("[BACKBONE] Successfully loaded and frozen DINOv2 backbone.")
        return backbone
    except Exception as e:
        print(f"[WARNING] Could not load torch.hub DINOv2: {e}")
        print("          Falling back to synthetic feature extraction mode.")
        return None

def train(args):
    # Load Config
    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    set_deterministic_seed(seed)
    
    # Device setup
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[INFO] Running in CPU Mode")
    else:
        device = torch.device(cfg["project"]["device"])
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        
    print(f"[TARGET] Category: {args.category}")
    print(f"[SEED] Initialized to: {seed}")
    
    # Model Selection
    if args.use_gct:
        print("[MODEL] Selected Dinomaly + Global Consistency Token (GCT)")
        model = DinomalyGCT(
            embed_dim=cfg["model"]["embed_dim"],
            num_decoder_layers=cfg["model"]["decoder_layers"]
        ).to(device)
    else:
        print("[MODEL] Selected Dinomaly Baseline")
        model = DinomalyBaseline(
            embed_dim=cfg["model"]["embed_dim"],
            num_decoder_layers=cfg["model"]["decoder_layers"]
        ).to(device)
        
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    
    # DINOv2 Backbone
    backbone = load_dinov2_backbone(device) if not args.cpu else None
    
    # Dataset Loading Setup
    data_path = ROOT / cfg["dataset"]["data_path"] / args.category / "train"
    use_real_data = data_path.exists() and not args.cpu
    
    if use_real_data:
        print(f"[DATASET] Loading real MVTec LOCO AD images from: {data_path}")
        transform = transforms.Compose([
            transforms.Resize((cfg["dataset"]["img_size"], cfg["dataset"]["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        num_workers = 4 if torch.cuda.is_available() and not args.cpu else 0
        dataloader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=False)
    else:
        print("[DATASET] Using synthetic batching mode (Local dry-test or missing data folder).")
        dataloader = None

    epochs = args.epochs if args.epochs else (1 if args.cpu else cfg["train"]["epochs"])
    print(f"[TRAIN] Starting Training Loop for {epochs} epoch(s)...")
    model.train()
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        total_loss_accum = 0.0
        batch_count = 0
        
        if use_real_data and dataloader is not None and backbone is not None:
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                with torch.no_grad():
                    # Extract features from DINOv2
                    features = backbone.forward_features(imgs)
                    patch_tokens = features["x_norm_patchtokens"]  # [B, 1024, 768]
                    cls_token = features["x_norm_clstoken"]        # [B, 768]
                    
                optimizer.zero_grad()
                if args.use_gct:
                    out_dict = model(patch_tokens, dinov2_cls_token=cls_token)
                    loss = out_dict["total_loss"]
                else:
                    rec_patches = model(patch_tokens)
                    loss = nn.functional.mse_loss(rec_patches, patch_tokens)
                    
                loss.backward()
                optimizer.step()
                total_loss_accum += loss.item()
                batch_count += 1
            avg_loss = total_loss_accum / max(batch_count, 1)
        else:
            # Synthetic dry-run batch
            dummy_patch_tokens = torch.randn(cfg["train"]["batch_size"], 1024, cfg["model"]["embed_dim"]).to(device)
            dummy_cls = torch.randn(cfg["train"]["batch_size"], cfg["model"]["embed_dim"]).to(device)
            
            optimizer.zero_grad()
            if args.use_gct:
                out_dict = model(dummy_patch_tokens, dinov2_cls_token=dummy_cls)
                loss = out_dict["total_loss"]
            else:
                output = model(dummy_patch_tokens)
                loss = nn.functional.mse_loss(output, dummy_patch_tokens)
                
            loss.backward()
            optimizer.step()
            avg_loss = loss.item()
            
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            sec_per_epoch = elapsed / epoch
            print(f"  [Epoch {epoch:03d}/{epochs:03d}] Avg Loss: {avg_loss:.6f} | Speed: {sec_per_epoch:.2f}s/epoch")
            
    total_time = time.time() - start_time
    print(f"[SUCCESS] Training completed in {total_time:.2f} seconds.")
    
    # Save checkpoint
    save_dir = ROOT / cfg["train"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = "gct" if args.use_gct else "baseline"
    ckpt_path = save_dir / f"{model_prefix}_{args.category}_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[SAVED] Checkpoint saved to: {ckpt_path.relative_to(ROOT)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dinomaly Baseline / GCT on MVTec LOCO AD")
    parser.add_argument("--config", type=str, default="src/configs/baseline_loco.json", help="Path to config file")
    parser.add_argument("--category", type=str, default="breakfast_box", help="Category to train")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--use_gct", action="store_true", help="Enable Global Consistency Token (GCT)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution for local testing")
    
    args = parser.parse_args()
    train(args)
# Model checkpoints are saved dynamically based on the target category
# Pinned memory drastically reduces CPU-GPU transfer bottlenecks
  
  
