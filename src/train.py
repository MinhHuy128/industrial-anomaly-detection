import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(args):
    set_deterministic_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Using device: {device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/baseline_loco.json")
    parser.add_argument("--category", type=str, default="breakfast_box")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)
