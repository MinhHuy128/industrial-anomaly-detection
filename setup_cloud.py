import os
import sys
import subprocess
import shutil
from pathlib import Path

# Force unbuffered stdout output for real-time terminal printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent

# Direct download link for official MVTec LOCO AD dataset (includes ground_truth masks)
MVTEC_LOCO_URL = "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz"

def log(msg):
    print(msg, flush=True)

def run_command(cmd, cwd=None, ignore_error=False):
    log(f"[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0 and not ignore_error:
        log(f"[WARNING] Command exited with code: {res.returncode}")
    return res.returncode

def setup_requirements():
    req_file = ROOT / "requirements.txt"
    if req_file.exists():
        log("[SETUP] Installing Python dependencies...")
        run_command(f"pip install -r {req_file} scikit-learn pillow", ignore_error=True)

def setup_dataset():
    data_dir = ROOT / "data" / "mvtec_loco"
    data_dir.mkdir(parents=True, exist_ok=True)

    categories = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

    # Check if dataset already fully present (including ground_truth)
    if all((data_dir / cat / "test" / "ground_truth").exists() for cat in categories):
        log("[SETUP] Dataset with ground_truth masks already present. Skipping download.")
        return

    tar_path = ROOT / "data" / "mvtec_loco_anomaly_detection.tar.xz"

    # Download if not already downloaded
    if not tar_path.exists():
        log("[SETUP] Downloading official MVTec LOCO AD dataset (~3.9 GB)...")
        log(f"[SETUP] Source: {MVTEC_LOCO_URL}")
        ret = run_command(
            f'wget -O "{tar_path}" "{MVTEC_LOCO_URL}"',
            ignore_error=False
        )
        if ret != 0:
            log("[ERROR] Download failed. Please download manually and place at:")
            log(f"        {tar_path}")
            sys.exit(1)
    else:
        log(f"[SETUP] Archive already exists at {tar_path}, skipping download.")

    # Extract
    log("[SETUP] Extracting dataset (this may take a few minutes)...")
    run_command(
        f'tar -xf "{tar_path}" -C "{data_dir}" --strip-components=1',
        ignore_error=False
    )

    # Verify
    missing = [cat for cat in categories if not (data_dir / cat).exists()]
    if missing:
        log(f"[WARNING] Missing categories after extraction: {missing}")
    else:
        log("[SETUP] All 5 categories extracted successfully!")

    # Verify ground_truth masks
    has_gt = all((data_dir / cat / "test" / "ground_truth").exists() for cat in categories)
    if has_gt:
        log("[SETUP] Ground truth masks confirmed present — sPRO computation ready!")
    else:
        log("[WARNING] Ground truth masks not found. sPRO will return N/A.")

    # Clean up archive to save disk space
    try:
        tar_path.unlink()
        log("[SETUP] Cleaned up archive file to save disk space.")
    except Exception:
        pass

    log("[SETUP] Dataset setup complete at data/mvtec_loco/")

def main():
    log("=" * 60)
    log("[SETUP] AUTOMATED CLOUD ENVIRONMENT SETUP FOR MVTEC LOCO AD")
    log("=" * 60)

    setup_requirements()
    setup_dataset()

    log("=" * 60)
    log("[SUCCESS] CLOUD SETUP COMPLETED! READY TO EVALUATE.")
    log("[USAGE]   python src/eval.py --category breakfast_box")
    log("[USAGE]   python src/eval.py --category breakfast_box --use_gct")
    log("=" * 60)

if __name__ == "__main__":
    main()
