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


def run_full_benchmark(config_path: str, save_maps: bool = False, verbose: bool = False):
    print("Evaluating 5 MVTec LOCO categories...\n")

    results = {
        "baseline": {},
        "gct": {}
    }

    print("Evaluating ViTill-GCT...")
    for cat in CATEGORIES:
        args = types.SimpleNamespace(
            config=config_path,
            category=cat,
            use_gct=True,
            gamma=1.0,
            save_maps=save_maps,
            verbose=verbose,
            cpu=False
        )
        res = evaluate(args)
        results["gct"][cat] = res

    print("\nEvaluating Baseline...")
    for cat in CATEGORIES:
        args = types.SimpleNamespace(
            config=config_path,
            category=cat,
            use_gct=False,
            gamma=0.0,
            save_maps=save_maps,
            verbose=verbose,
            cpu=False
        )
        res = evaluate(args)
        results["baseline"][cat] = res

    for m_key in ["baseline", "gct"]:
        m_dict = results[m_key]
        mean_log = sum(m_dict[c]["logical_auroc"] for c in CATEGORIES) / len(CATEGORIES)
        mean_struct = sum(m_dict[c]["structural_auroc"] for c in CATEGORIES) / len(CATEGORIES)
        mean_all = sum(m_dict[c]["mean_auroc"] for c in CATEGORIES) / len(CATEGORIES)
        mean_log_f1 = sum(m_dict[c].get("logical_f1", 0.0) for c in CATEGORIES) / len(CATEGORIES)
        mean_str_f1 = sum(m_dict[c].get("structural_f1", 0.0) for c in CATEGORIES) / len(CATEGORIES)
        mean_f1_all = sum(m_dict[c].get("mean_f1", 0.0) for c in CATEGORIES) / len(CATEGORIES)
        mean_spro = sum(m_dict[c]["spro"] for c in CATEGORIES) / len(CATEGORIES)
        mean_lat = sum(m_dict[c]["latency_ms"] for c in CATEGORIES) / len(CATEGORIES)
        mean_fps = sum(m_dict[c]["fps"] for c in CATEGORIES) / len(CATEGORIES)

        m_dict["MEAN"] = {
            "category": "MEAN",
            "logical_auroc": mean_log,
            "structural_auroc": mean_struct,
            "mean_auroc": mean_all,
            "logical_f1": mean_log_f1,
            "structural_f1": mean_str_f1,
            "mean_f1": mean_f1_all,
            "spro": mean_spro,
            "latency_ms": mean_lat,
            "fps": mean_fps
        }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_official.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = out_dir / "benchmark_official.md"
    md_content = generate_markdown_report(results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\nBenchmark summary table:")
    print(md_content)
    print(f"\nSaved results to {json_path} and {md_path}")


def generate_markdown_report(results: dict) -> str:
    lines = []
    lines.append("### Table 1: ViTill-GCT (Proposed Dual-Stream)\n")
    lines.append("| Category | Logical AUROC (%) | Structural AUROC (%) | Mean AUROC (%) | sPRO (AUPRO ≤ 0.30) (%) | Latency (ms) | FPS |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cat in CATEGORIES + ["MEAN"]:
        d = results["gct"][cat]
        bold = "**" if cat == "MEAN" else ""
        lines.append(f"| {bold}{cat.upper()}{bold} | {d['logical_auroc']:.2f}% | {d['structural_auroc']:.2f}% | {d['mean_auroc']:.2f}% | {d['spro']:.2f}% | {d['latency_ms']:.2f} ms | {d['fps']:.1f} |")

    lines.append("\n### Table 2: Comparative Baseline (Single-Stream)\n")
    lines.append("| Category | Logical AUROC (%) | Structural AUROC (%) | Mean AUROC (%) | sPRO (AUPRO ≤ 0.30) (%) | Latency (ms) | FPS |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cat in CATEGORIES + ["MEAN"]:
        d = results["baseline"][cat]
        bold = "**" if cat == "MEAN" else ""
        lines.append(f"| {bold}{cat.upper()}{bold} | {d['logical_auroc']:.2f}% | {d['structural_auroc']:.2f}% | {d['mean_auroc']:.2f}% | {d['spro']:.2f}% | {d['latency_ms']:.2f} ms | {d['fps']:.1f} |")

    lines.append("\n### Table 3: Performance Comparison\n")
    lines.append("| Metric | Baseline | ViTill-GCT | Improvement |")
    lines.append("|:---|:---:|:---:|:---:|")
    b_mean = results["baseline"]["MEAN"]
    g_mean = results["gct"]["MEAN"]
    lines.append(f"| **Logical AUROC** | {b_mean['logical_auroc']:.2f}% | **{g_mean['logical_auroc']:.2f}%** | **+{g_mean['logical_auroc'] - b_mean['logical_auroc']:.2f}%** |")
    lines.append(f"| **Structural AUROC** | {b_mean['structural_auroc']:.2f}% | **{g_mean['structural_auroc']:.2f}%** | **+{g_mean['structural_auroc'] - b_mean['structural_auroc']:.2f}%** |")
    lines.append(f"| **Mean AUROC** | {b_mean['mean_auroc']:.2f}% | **{g_mean['mean_auroc']:.2f}%** | **+{g_mean['mean_auroc'] - b_mean['mean_auroc']:.2f}%** |")
    lines.append(f"| **sPRO (AUPRO ≤ 0.30)** | {b_mean['spro']:.2f}% | **{g_mean['spro']:.2f}%** | {g_mean['spro'] - b_mean['spro']:+.2f}% |")
    lines.append(f"| **Latency (batch=1)** | {b_mean['latency_ms']:.2f} ms | **{g_mean['latency_ms']:.2f} ms** | {g_mean['fps']:.1f} FPS |")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full MVTec LOCO AD benchmark")
    parser.add_argument("--config", type=str, default="src/configs/loco_strict.json")
    parser.add_argument("--save_maps", action="store_true", help="Save anomaly maps for official evaluation")
    parser.add_argument("--verbose", action="store_true", help="Print detailed evaluation metrics")
    args = parser.parse_args()
    run_full_benchmark(args.config, save_maps=args.save_maps, verbose=args.verbose)
