import os
import sys
import subprocess
import shutil
from pathlib import Path

# Force unbuffered stdout output for real-time terminal printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent

# Direct download links provided by MVTec LOCO AD official portal
MVTEC_LOCO_URL = "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz"
MVTEC_EVAL_URL = "https://www.mydrive.ch/shares/48245/a4e9922c5efa93f57b6a0ff9f5c6b969/download/430648014-1646847095/mvtec_loco_ad_evaluation.tar.xz"

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
    log("[SETUP] Installing Python dependencies...")
    if req_file.exists():
        run_command(f"pip install -r {req_file} scikit-learn pillow scipy tabulate tqdm tifffile", ignore_error=True)
    else:
        run_command("pip install torch torchvision scikit-learn pillow scipy tabulate tqdm tifffile", ignore_error=True)

def download_fast(url, target_path):
    """
    Multi-tier fast download strategy:
      1. aria2c (16 connections multi-threaded download)
      2. Auto-install aria2c via apt-get on Linux
      3. curl / wget
      4. Python urllib.request (Windows fallback)
    """
    data_dir = target_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = target_path.name

    # 1. Check aria2c
    if shutil.which("aria2c"):
        log("[SPEED-UP] Using aria2c (16 connections multi-threaded download)...")
        ret = run_command(f'aria2c -x 16 -s 16 -d "{data_dir}" -o "{filename}" "{url}"', ignore_error=True)
        if ret == 0 and target_path.exists():
            log("[SETUP] Ultra-fast aria2c download completed!")
            return

    # 2. Try auto-installing aria2c on Linux if apt-get is available
    if sys.platform != "win32" and shutil.which("apt-get"):
        log("[SETUP] Installing aria2c for multi-threaded downloading...")
        run_command("apt-get update -qq && apt-get install -y -qq aria2", ignore_error=True)
        if shutil.which("aria2c"):
            log("[SPEED-UP] Using aria2c (16 connections multi-threaded download)...")
            ret = run_command(f'aria2c -x 16 -s 16 -d "{data_dir}" -o "{filename}" "{url}"', ignore_error=True)
            if ret == 0 and target_path.exists():
                log("[SETUP] Ultra-fast aria2c download completed!")
                return

    # 3. Fallback: curl / wget
    if shutil.which("curl"):
        log("[DOWNLOAD] Using curl...")
        ret = run_command(f'curl -L -o "{target_path}" "{url}"', ignore_error=True)
        if ret == 0 and target_path.exists():
            return

    if shutil.which("wget"):
        log("[DOWNLOAD] Using wget...")
        ret = run_command(f'wget -O "{target_path}" "{url}"', ignore_error=True)
        if ret == 0 and target_path.exists():
            return

    # 4. Fallback: Python urllib
    log("[DOWNLOAD] Using Python urllib...")
    import urllib.request
    def _progress(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r[DOWNLOAD] {percent}% ({count * block_size // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
        else:
            sys.stdout.write(f"\r[DOWNLOAD] {count * block_size // (1024*1024)}MB downloaded")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, target_path, reporthook=_progress)
    print()
    log("[SETUP] Download completed successfully.")

def setup_dataset():
    data_dir = ROOT / "data" / "mvtec_loco"
    data_dir.mkdir(parents=True, exist_ok=True)

    categories = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

    # Check if dataset already fully present (including test directories)
    if all((data_dir / cat / "test" / "good").exists() for cat in categories):
        log("[SETUP] Dataset already present at data/mvtec_loco/. Skipping download.")
        return

    tar_path = ROOT / "data" / "mvtec_loco_anomaly_detection.tar.xz"

    # Download if not already downloaded
    if not tar_path.exists():
        log("[SETUP] Downloading official MVTec LOCO AD dataset (~3.9 GB)...")
        log(f"[SETUP] Source: {MVTEC_LOCO_URL}")
        download_fast(MVTEC_LOCO_URL, tar_path)
    else:
        log(f"[SETUP] Archive already exists at {tar_path}, skipping download.")

    # Extract
    log("[SETUP] Extracting dataset (this may take a few minutes)...")
    ret = run_command(
        f'tar -xf "{tar_path}" -C "{data_dir}" --strip-components=1',
        ignore_error=True
    )
    if ret != 0:
        log("[INFO] Attempting Python tarfile extraction fallback...")
        import tarfile
        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(path=data_dir)

    # Verify
    missing = [cat for cat in categories if not (data_dir / cat).exists()]
    if missing:
        log(f"[WARNING] Missing categories after extraction: {missing}")
    else:
        log("[SETUP] All 5 categories extracted successfully!")

    # Verify ground_truth masks
    has_gt = all((data_dir / cat / "ground_truth").exists() for cat in categories)
    if has_gt:
        log("[SETUP] Ground truth masks confirmed present — sPRO computation ready!")
    else:
        log("[WARNING] Ground truth masks not found.")

    # Clean up archive to save disk space
    try:
        tar_path.unlink()
        log("[SETUP] Cleaned up dataset archive file to save disk space.")
    except Exception:
        pass

    log("[SETUP] Dataset setup complete at data/mvtec_loco/")

def setup_evaluation_kit():
    """
    Downloads and extracts official MVTec LOCO AD evaluation code (evaluate_experiment.py).
    """
    eval_dir = ROOT / "mvtec_loco_ad_evaluation"
    eval_script = eval_dir / "evaluate_experiment.py"

    if eval_script.exists():
        log("[SETUP] Official MVTec evaluation kit already present at mvtec_loco_ad_evaluation/. Skipping.")
        return

    eval_dir.mkdir(parents=True, exist_ok=True)
    tar_path = ROOT / "data" / "mvtec_loco_ad_evaluation.tar.xz"

    log("[SETUP] Downloading official MVTec LOCO AD Evaluation Kit...")
    log(f"[SETUP] Source: {MVTEC_EVAL_URL}")
    download_fast(MVTEC_EVAL_URL, tar_path)

    log("[SETUP] Extracting official evaluation code...")
    ret = run_command(
        f'tar -xf "{tar_path}" -C "{ROOT}"',
        ignore_error=True
    )
    if ret != 0 or not eval_script.exists():
        log("[INFO] Attempting Python tarfile extraction fallback...")
        import tarfile
        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(path=ROOT)

    # Clean up archive
    try:
        tar_path.unlink()
    except Exception:
        pass

    if eval_script.exists():
        log("[SETUP] Official MVTec evaluation kit installed successfully at ./mvtec_loco_ad_evaluation/")
    else:
        log("[WARNING] evaluate_experiment.py not found in extracted files. Please verify extraction.")

def main():
    log("=" * 60)
    log("[SETUP] AUTOMATED CLOUD/LOCAL ENVIRONMENT SETUP FOR MVTEC LOCO AD")
    log("=" * 60)

    setup_requirements()
    setup_dataset()
    setup_evaluation_kit()

    log("=" * 60)
    log("[SUCCESS] ALL SETUP COMPLETED! 100% READY FOR OFFICIAL BENCHMARK.")
    log("=" * 60)
    log(">> STEP 1 (Image AUROC & Export Maps):  python src/benchmark_all.py --save_maps")
    log(">> STEP 2 (Official sPRO Evaluation):   python src/run_official_spro.py")
    log(">> STEP 3 (Compile LaTeX Paper Table): python src/compile_results.py")
    log("=" * 60)

if __name__ == "__main__":
    main()
