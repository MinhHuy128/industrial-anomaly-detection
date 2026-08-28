"""
Paired bootstrap confidence intervals for model comparison.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def compute_bootstrap_ci(
    labels: np.ndarray,
    scores_base: np.ndarray,
    scores_new: np.ndarray,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict:
    np.random.seed(seed)
    n_samples = len(labels)

    base_point = float(roc_auc_score(labels, scores_base)) * 100.0
    new_point = float(roc_auc_score(labels, scores_new)) * 100.0
    delta_point = new_point - base_point

    boot_base = []
    boot_new = []
    boot_delta = []

    for _ in range(n_bootstraps):
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_labels = labels[idxs]

        if len(np.unique(sample_labels)) < 2:
            continue

        s_base_b = scores_base[idxs]
        s_new_b = scores_new[idxs]

        auc_b = float(roc_auc_score(sample_labels, s_base_b)) * 100.0
        auc_n = float(roc_auc_score(sample_labels, s_new_b)) * 100.0
        d_auc = auc_n - auc_b

        boot_base.append(auc_b)
        boot_new.append(auc_n)
        boot_delta.append(d_auc)

    boot_delta = np.array(boot_delta)
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_delta, alpha * 100.0))
    ci_upper = float(np.percentile(boot_delta, (1.0 - alpha) * 100.0))

    p_value = float(np.mean(boot_delta <= 0.0))
    is_statistically_significant = bool(ci_lower > 0.0 and delta_point >= 1.0)
    verdict = "Significant" if is_statistically_significant else "Not significant"

    return {
        "n_bootstraps": len(boot_delta),
        "confidence_level": confidence_level,
        "baseline_auroc": round(base_point, 2),
        "new_model_auroc": round(new_point, 2),
        "delta_auroc": round(delta_point, 2),
        "ci_95_lower": round(ci_lower, 2),
        "ci_95_upper": round(ci_upper, 2),
        "p_value": round(p_value, 4),
        "is_significant": is_statistically_significant,
        "verdict": verdict,
        "gate_verdict": verdict
    }



def evaluate_from_eval_files(
    baseline_eval_json: Path,
    new_eval_json: Path,
    output_report_md: Optional[Path] = None
) -> Dict:
    with open(baseline_eval_json, "r", encoding="utf-8") as f:
        data_base = json.load(f)
    with open(new_eval_json, "r", encoding="utf-8") as f:
        data_new = json.load(f)

    category = data_new.get("category", "unknown")
    preds_base = data_base["raw_predictions"]
    preds_new = data_new["raw_predictions"]

    labels = np.array(preds_base["labels"])
    scores_base = np.array(preds_base["scores"])
    scores_new = np.array(preds_new["scores"])
    types = np.array(preds_base["types"])

    log_mask = (types == "good") | (types == "logical_anomalies")
    res_logical = compute_bootstrap_ci(
        labels[log_mask], scores_base[log_mask], scores_new[log_mask]
    )
    res_all = compute_bootstrap_ci(labels, scores_base, scores_new)

    full_report = {
        "category": category,
        "logical_anomaly_bootstrap": res_logical,
        "overall_bootstrap": res_all
    }

    md_content = f"""# Statistical Comparison
**Category:** `{category}`  
**Bootstraps:** $B = {res_logical['n_bootstraps']}$  

## Logical Anomaly AUROC
* **Baseline:** {res_logical['baseline_auroc']}%
* **New Model:** {res_logical['new_model_auroc']}%
* **Delta ($\\\\Delta$):** **{res_logical['delta_auroc']:+.2f}%**
* **95% CI:** `[{res_logical['ci_95_lower']:+.2f}%, {res_logical['ci_95_upper']:+.2f}%]`
* **p-value:** `{res_logical['p_value']:.4f}`
* **Significance:** **`{res_logical['verdict']}`**
"""

    if output_report_md:
        with open(output_report_md, "w", encoding="utf-8") as f:
            f.write(md_content)

    print(f"[{category}] Logical AUROC Delta: {res_logical['delta_auroc']:+.2f}% | 95% CI: [{res_logical['ci_95_lower']:+.2f}%, {res_logical['ci_95_upper']:+.2f}%] | p={res_logical['p_value']:.4f} ({res_logical['verdict']})")
    return full_report



def main():
    parser = argparse.ArgumentParser(description="Statistical Bootstrap CI Verification Tool.")
    parser.add_argument("--base_json", type=str, default=None, help="Path to baseline eval_results.json")
    parser.add_argument("--new_json", type=str, default=None, help="Path to new model eval_results.json")
    parser.add_argument("--out_md", type=str, default="results/statistical_gate_audit.md")
    args = parser.parse_args()

    if args.base_json and args.new_json:
        evaluate_from_eval_files(
            baseline_eval_json=Path(args.base_json),
            new_eval_json=Path(args.new_json),
            output_report_md=Path(args.out_md)
        )
    else:
        print("Running bootstrap CI verification self-test...")
        np.random.seed(42)
        labels = np.array([0] * 50 + [1] * 50)
        scores_base = np.array([0.1] * 50 + [0.8] * 50) + np.random.normal(0, 0.1, 100)
        scores_new = np.array([0.05] * 50 + [0.9] * 50) + np.random.normal(0, 0.1, 100)
        res = compute_bootstrap_ci(labels, scores_base, scores_new, n_bootstraps=500, seed=42)
        print(f"Baseline AUROC: {res['baseline_auroc']:.2f}% | New Model AUROC: {res['new_model_auroc']:.2f}%")
        print(f"Delta AUROC: {res['delta_auroc']:+.2f}% | 95% CI: [{res['ci_95_lower']:+.2f}%, {res['ci_95_upper']:+.2f}%] | p={res['p_value']:.4f}")
        print(f"Gate Verdict: {res['gate_verdict']}")


if __name__ == "__main__":
    main()
