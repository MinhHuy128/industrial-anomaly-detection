import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors"
]

MODELS = ["gct", "baseline"]
EVAL_DIR = ROOT / "mvtec_loco_ad_evaluation"
EVAL_SCRIPT = EVAL_DIR / "evaluate_experiment.py"
PRINT_SCRIPT = EVAL_DIR / "print_metrics.py"
DATASET_DIR = (ROOT / "data" / "mvtec_loco") if (ROOT / "data" / "mvtec_loco").exists() else (ROOT / "data" / "mvtec_loco_ad")
MAPS_DIR = ROOT / "outputs" / "anomaly_maps"
OUTPUT_DIR = ROOT / "results" / "official_metrics"


def check_prerequisites():
    errors = []

    if not EVAL_SCRIPT.exists():
        errors.append(f"Missing evaluation script at: {EVAL_SCRIPT}")

    if not DATASET_DIR.exists():
        errors.append(f"Missing dataset directory at: {DATASET_DIR}")

    for model in MODELS:
        maps_root = MAPS_DIR / model
        if not maps_root.exists():
            errors.append(f"Missing anomaly maps for '{model}' at: {maps_root}")

    if errors:
        print("Prerequisites check failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


def run_official_eval_for_category(model: str, category: str):
    maps_dir = MAPS_DIR / model
    output_dir = OUTPUT_DIR / model / category
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--dataset_base_dir", str(DATASET_DIR),
        "--anomaly_maps_dir", str(maps_dir),
        "--output_dir", str(output_dir),
        "--object_name", category,
    ]

    print(f"Evaluating {model} on {category}...")
    result = subprocess.run(cmd, cwd=str(EVAL_DIR))
    if result.returncode != 0:
        print(f"Warning: evaluation script returned non-zero code for {model}/{category}")


def print_official_summary_tables():
    if not PRINT_SCRIPT.exists():
        return

    print("\nSummary tables:")
    print("\nLocalization: sPRO @ FPR = 0.05")
    subprocess.run([
        sys.executable, str(PRINT_SCRIPT),
        "--metrics_folder", str(OUTPUT_DIR),
        "--metric_type", "localization",
        "--integration_limit", "0.05"
    ], cwd=str(EVAL_DIR))

    print("\nLocalization: sPRO @ FPR = 0.30")
    subprocess.run([
        sys.executable, str(PRINT_SCRIPT),
        "--metrics_folder", str(OUTPUT_DIR),
        "--metric_type", "localization",
        "--integration_limit", "0.3"
    ], cwd=str(EVAL_DIR))

    print("\nClassification: Image AUROC")
    subprocess.run([
        sys.executable, str(PRINT_SCRIPT),
        "--metrics_folder", str(OUTPUT_DIR),
        "--metric_type", "classification"
    ], cwd=str(EVAL_DIR))


def main():
    check_prerequisites()

    for model in MODELS:
        name = "ViTill-GCT" if model == "gct" else "Comparative Baseline"
        print(f"\nModel: {name}")
        for cat in CATEGORIES:
            run_official_eval_for_category(model, cat)

    print_official_summary_tables()
    print(f"\nSaved metrics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
