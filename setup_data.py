import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request
import tarfile

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent

MVTEC_LOCO_URL = "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz"
MVTEC_EVAL_URL = "https://www.mydrive.ch/shares/48245/a4e9922c5efa93f57b6a0ff9f5c6b969/download/430648014-1646847095/mvtec_loco_ad_evaluation.tar.xz"


def run_command(cmd, cwd=None, ignore_error=False):
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0 and not ignore_error:
        print(f"Command failed with code {res.returncode}: {cmd}")
    return res.returncode


def setup_requirements():
    req_file = ROOT / "requirements.txt"
    print("Checking dependencies...")
    if req_file.exists():
        run_command(f"pip install -r {req_file} scikit-learn pillow scipy tabulate tqdm tifffile", ignore_error=True)
    else:
        run_command("pip install torch torchvision scikit-learn pillow scipy tabulate tqdm tifffile", ignore_error=True)


def download_fast(url, target_path):
    data_dir = target_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = target_path.name

    if shutil.which("aria2c"):
        ret = run_command(f'aria2c -x 16 -s 16 -d "{data_dir}" -o "{filename}" "{url}"', ignore_error=True)
        if ret == 0 and target_path.exists():
            return

    if sys.platform != "win32" and shutil.which("apt-get"):
        run_command("apt-get update -qq && apt-get install -y -qq aria2", ignore_error=True)
        if shutil.which("aria2c"):
            ret = run_command(f'aria2c -x 16 -s 16 -d "{data_dir}" -o "{filename}" "{url}"', ignore_error=True)
            if ret == 0 and target_path.exists():
                return

    if shutil.which("curl"):
        ret = run_command(f'curl -L -o "{target_path}" "{url}"', ignore_error=True)
        if ret == 0 and target_path.exists():
            return

    if shutil.which("wget"):
        ret = run_command(f'wget -O "{target_path}" "{url}"', ignore_error=True)
        if ret == 0 and target_path.exists():
            return

    def _progress(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = count * block_size // (1024 * 1024)
            mb_total = total_size // (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {percent}% ({mb_downloaded}MB / {mb_total}MB)")
        else:
            sys.stdout.write(f"\rDownloading: {count * block_size // (1024 * 1024)}MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, target_path, reporthook=_progress)
    print()


def setup_dataset():
    data_dir = ROOT / "data" / "mvtec_loco"
    data_dir.mkdir(parents=True, exist_ok=True)

    categories = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

    if all((data_dir / cat / "test" / "good").exists() for cat in categories):
        print("Dataset already exists at data/mvtec_loco, skipping download.")
        return

    tar_path = ROOT / "data" / "mvtec_loco_anomaly_detection.tar.xz"

    if not tar_path.exists():
        print(f"Downloading MVTec LOCO AD dataset from {MVTEC_LOCO_URL}...")
        download_fast(MVTEC_LOCO_URL, tar_path)

    print("Extracting dataset...")
    ret = run_command(f'tar -xf "{tar_path}" -C "{data_dir}" --strip-components=1', ignore_error=True)
    if ret != 0:
        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(path=data_dir)

    missing = [cat for cat in categories if not (data_dir / cat).exists()]
    if missing:
        print(f"Warning: missing categories: {missing}")

    try:
        tar_path.unlink()
    except Exception:
        pass

    print("Dataset setup ready at data/mvtec_loco")


def setup_evaluation_kit():
    eval_dir = ROOT / "mvtec_loco_ad_evaluation"
    eval_script = eval_dir / "evaluate_experiment.py"

    if eval_script.exists():
        return

    eval_dir.mkdir(parents=True, exist_ok=True)
    tar_path = ROOT / "data" / "mvtec_loco_ad_evaluation.tar.xz"

    print("Downloading MVTec evaluation kit...")
    download_fast(MVTEC_EVAL_URL, tar_path)

    ret = run_command(f'tar -xf "{tar_path}" -C "{ROOT}"', ignore_error=True)
    if ret != 0 or not eval_script.exists():
        with tarfile.open(tar_path, "r:xz") as tar:
            tar.extractall(path=ROOT)

    try:
        tar_path.unlink()
    except Exception:
        pass


def main():
    print("Setting up MVTec LOCO AD environment...")
    setup_requirements()
    setup_dataset()
    setup_evaluation_kit()
    print("Done.")


if __name__ == "__main__":
    main()
