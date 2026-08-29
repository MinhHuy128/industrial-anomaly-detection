"""
Ablation study runner for ranking and adapter modules.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from experiments.vlm_ablation.train_vlm_ranking import load_config, train_category
from experiments.vlm_ablation.eval_vlm_ranking import evaluate_ranking_checkpoint

ABLATION_MODES = [
    ("gct_baseline", "GCT Baseline"),
    ("gct_ranking", "GCT + Ranking"),
    ("gct_adapter", "GCT + Adapter"),
    ("gct_adapter_modulation", "GCT + Adapter + Modulation"),
    ("gct_adapter_modulation_ranking", "GCT + Adapter + Modulation + Ranking")
]


def run_ablation_study(
    config_path: str,
    categories: List[str],
    total_iters: int = 5000,
    device_str: str = "cuda"
) -> Dict:
    cfg = load_config(config_path)
    data_root = ROOT / cfg.get("dataset", {}).get("data_path", "data/mvtec_loco")
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ablation study on: {', '.join(categories)} (iters={total_iters}, seed={cfg.get('project', {}).get('seed', 42)})\n")

    ablation_records = []

    for cat in categories:
        print(f"--- Category: {cat} ---")

        for mode_key, mode_name in ABLATION_MODES:
            print(f"[{cat}] Mode: {mode_name}")
            train_summary = train_category(
                category=cat,
                cfg=cfg,
                mode=mode_key,
                total_iters_override=total_iters,
                device_str=device_str
            )

            ckpt_path = ROOT / train_summary["checkpoint"]
            eval_summary = evaluate_ranking_checkpoint(
                checkpoint_path=ckpt_path,
                data_root=data_root,
                category=cat,
                device_str=device_str
            )

            record = {
                "category": cat,
                "mode_key": mode_key,
                "mode_name": mode_name,
                "logical_auroc": eval_summary["logical_auroc"],
                "structural_auroc": eval_summary["structural_auroc"],
                "mean_auroc": eval_summary["mean_auroc"],
                "gate_weight": eval_summary["gate_weight"],
                "latency_ms": eval_summary["latency_ms"],
                "checkpoint": train_summary["checkpoint"]
            }
            ablation_records.append(record)

    out_json = results_dir / "ablation_vlm_ranking_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "categories": categories,
            "ablation_records": ablation_records
        }, f, indent=2)

    md_lines = [
        "# Ablation Study: VLM Ranking & Gated Adapter\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Categories Evaluated:** {', '.join(categories)}  ",
        f"**Base Seed:** {cfg.get('project', {}).get('seed', 42)}\n",
        "| Category | Configuration | Logical AUROC (%) | Structural AUROC (%) | Mean AUROC (%) | Gate Weight | Latency (ms) |",
        "|---|---|---|---|---|---|---|"
    ]

    for r in ablation_records:
        md_lines.append(
            f"| `{r['category']}` | {r['mode_name']} | **{r['logical_auroc']:.2f}** | {r['structural_auroc']:.2f} | {r['mean_auroc']:.2f} | `{r['gate_weight']:.4f}` | {r['latency_ms']:.1f} |"
        )

    md_content = "\n".join(md_lines)
    out_md = results_dir / "ablation_vlm_ranking_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nAblation study finished. Results saved to {out_json.relative_to(ROOT)} and {out_md.relative_to(ROOT)}")
    return {"ablation_records": ablation_records}



def main():
    parser = argparse.ArgumentParser(description="Run 5-stage ablation study for VLM ranking & gated adapter.")
    parser.add_argument("--config", type=str, default="experiments/vlm_ablation/configs/vlm_ranking_loco.yaml", help="Path to YAML config")
    parser.add_argument("--categories", nargs="+", default=["pushpins", "screw_bag"], help="Categories to test")
    parser.add_argument("--all", action="store_true", help="Run across all 5 MVTec LOCO categories")
    parser.add_argument("--iters", type=int, default=5000, help="Training iterations per mode")
    parser.add_argument("--device", type=str, default="cuda" if sys.platform != "darwin" else "cpu")
    args = parser.parse_args()

    cats = [
        "breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"
    ] if args.all else args.categories

    run_ablation_study(
        config_path=args.config,
        categories=cats,
        total_iters=args.iters,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
