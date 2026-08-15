"""
Automated benchmark script across all 5 MVTec LOCO AD categories.
Runs both Baseline (Dinomaly) and ViTill-GCT V2, reporting full comparison metrics.
Outputs results to results/benchmark_official.json and results/benchmark_official.md.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.eval import evaluate

CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors"
]

def run_full_benchmark(config_path: str, save_maps: bool = False):
    print("=" * 80)
    print("🚀 STARTING OFFICIAL BENCHMARK ACROSS ALL 5 MVTEC LOCO AD CATEGORIES")
    print("=" * 80)

    results = {
        "baseline": {},
        "gct": {}
    }

    # 1. Evaluate ViTill-GCT V2
    print("\n" + "─" * 40)
    print("📌 [1/2] EVALUATING PROPOSED MODEL: ViTill-GCT V2")
    print("─" * 40)
    for cat in CATEGORIES:
        args = types.SimpleNamespace(
            config=config_path,
            category=cat,
            use_gct=True,
            gamma=1.0,
            save_maps=save_maps,
            cpu=False
        )
        print(f"\n>>> Running ViTill-GCT V2 on: {cat} ...")
        res = evaluate(args)
        results["gct"][cat] = res

    # 2. Evaluate Dinomaly Baseline
    print("\n" + "─" * 40)
    print("📌 [2/2] EVALUATING BASELINE MODEL: Dinomaly")
    print("─" * 40)
    for cat in CATEGORIES:
        args = types.SimpleNamespace(
            config=config_path,
            category=cat,
            use_gct=False,
            gamma=0.0,
            save_maps=save_maps,
            cpu=False
        )
        print(f"\n>>> Running Baseline on: {cat} ...")
        res = evaluate(args)
        results["baseline"][cat] = res

    # Compute Means
    for m_key in ["baseline", "gct"]:
        m_dict = results[m_key]
        mean_log    = sum(m_dict[c]["logical_auroc"] for c in CATEGORIES) / len(CATEGORIES)
        mean_struct = sum(m_dict[c]["structural_auroc"] for c in CATEGORIES) / len(CATEGORIES)
        mean_all    = sum(m_dict[c]["mean_auroc"] for c in CATEGORIES) / len(CATEGORIES)
        mean_log_f1 = sum(m_dict[c].get("logical_f1", 0.0) for c in CATEGORIES) / len(CATEGORIES)
        mean_str_f1 = sum(m_dict[c].get("structural_f1", 0.0) for c in CATEGORIES) / len(CATEGORIES)
        mean_f1_all = sum(m_dict[c].get("mean_f1", 0.0) for c in CATEGORIES) / len(CATEGORIES)
        mean_spro   = sum(m_dict[c]["spro"] for c in CATEGORIES) / len(CATEGORIES)
        mean_lat    = sum(m_dict[c]["latency_ms"] for c in CATEGORIES) / len(CATEGORIES)
        mean_fps    = sum(m_dict[c]["fps"] for c in CATEGORIES) / len(CATEGORIES)

        m_dict["MEAN"] = {
            "category": "MEAN",
            "logical_auroc": mean_log,
            "structural_auroc": mean_struct,
            "mean_auroc": mean_all,
            
            
            
            "spro": mean_spro,
            "latency_ms": mean_lat,
            "fps": mean_fps
        }

    # Save to JSON
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_official.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Format Markdown Table
    md_path = out_dir / "benchmark_official.md"
    md_content = generate_markdown_report(results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print("🏆 FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(md_content)
    print(f"\n[OK] Results saved to:\n  - {json_path.relative_to(ROOT)}\n  - {md_path.relative_to(ROOT)}")

def generate_markdown_report(results: dict) -> str:
    lines = []
    lines.append("### Table 1: ViTill-GCT V2 (Proposed Dual-Stream)\n")
    lines.append("| Category | Logical AUROC (%) | Structural AUROC (%) | Mean AUROC (%) | sPRO (AUPRO ≤ 0.30) (%) | Latency (ms) | FPS |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cat in CATEGORIES + ["MEAN"]:
        d = results["gct"][cat]
        bold = "**" if cat == "MEAN" else ""
        lines.append(f"| {bold}{cat.upper()}{bold} | {d['logical_auroc']:.2f}% | {d['structural_auroc']:.2f}% | {d['mean_auroc']:.2f}% | {d['spro']:.2f}% | {d['latency_ms']:.2f} ms | {d['fps']:.1f} |")

    lines.append("\n### Table 2: Dinomaly Baseline (Paper-Strict)\n")
    lines.append("| Category | Logical AUROC (%) | Structural AUROC (%) | Mean AUROC (%) | sPRO (AUPRO ≤ 0.30) (%) | Latency (ms) | FPS |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cat in CATEGORIES + ["MEAN"]:
        d = results["baseline"][cat]
        bold = "**" if cat == "MEAN" else ""
        lines.append(f"| {bold}{cat.upper()}{bold} | {d['logical_auroc']:.2f}% | {d['structural_auroc']:.2f}% | {d['mean_auroc']:.2f}% | {d['spro']:.2f}% | {d['latency_ms']:.2f} ms | {d['fps']:.1f} |")

    lines.append("\n### Table 3: Performance Delta (ViTill-GCT V2 vs. Baseline)\n")
    lines.append("| Metric | Baseline | ViTill-GCT V2 | Improvement (Delta) |")
    lines.append("|:---|:---:|:---:|:---:|")
    b_mean = results["baseline"]["MEAN"]
    g_mean = results["gct"]["MEAN"]
    lines.append(f"| **Logical AUROC** | {b_mean['logical_auroc']:.2f}% | **{g_mean['logical_auroc']:.2f}%** | **+{g_mean['logical_auroc'] - b_mean['logical_auroc']:.2f}%** 🏆 |")
    lines.append(f"| **Structural AUROC** | {b_mean['structural_auroc']:.2f}% | **{g_mean['structural_auroc']:.2f}%** | **+{g_mean['structural_auroc'] - b_mean['structural_auroc']:.2f}%** |")
    lines.append(f"| **Mean AUROC** | {b_mean['mean_auroc']:.2f}% | **{g_mean['mean_auroc']:.2f}%** | **+{g_mean['mean_auroc'] - b_mean['mean_auroc']:.2f}%** 🏆 |")
    lines.append(f"| **sPRO (AUPRO ≤ 0.30)** | {b_mean['spro']:.2f}% | **{g_mean['spro']:.2f}%** | {g_mean['spro'] - b_mean['spro']:+.2f}% |")
    lines.append(f"| **Latency (batch=1)** | {b_mean['latency_ms']:.2f} ms | **{g_mean['latency_ms']:.2f} ms** | Real-Time ({g_mean['fps']:.1f} FPS) |")

    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full MVTec LOCO AD benchmark")
    parser.add_argument("--config",    type=str, default="src/configs/loco_strict.json")
    parser.add_argument("--save_maps", action="store_true", help="Save anomaly maps for official evaluation")
    args = parser.parse_args()
    run_full_benchmark(args.config, save_maps=args.save_maps)
