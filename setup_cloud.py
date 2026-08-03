import os
import sys
import subprocess
import shutil
from pathlib import Path

# Force unbuffered stdout output for real-time terminal printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
KAGGLE_DATASET = "pmhuy454/mvtec-loco"

def log(msg):
    print(msg, flush=True)

def run_command(cmd, cwd=None, ignore_error=False):
    log(f"[EXEC] {cmd}")
    env = os.environ.copy()
    env["GIT_ASKPASS"] = ""
    env["GIT_TERMINAL_PROMPT"] = "0"
    
    res = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if res.returncode != 0 and not ignore_error:
        log(f"[WARNING] Command exited with code: {res.returncode}")
    return res.returncode

def setup_requirements():
    req_file = ROOT / "requirements.txt"
    if req_file.exists():
        log("[SETUP] Installing Python dependencies from requirements.txt...")
        run_command(f"pip install -r {req_file} kaggle scikit-learn pillow matplotlib", ignore_error=True)

def setup_dinomaly_baseline():
    dinomaly_path = ROOT / "Dinomaly"
    if not dinomaly_path.exists():
        log("[SETUP] Cloning baseline Dinomaly repository...")
        run_command("git clone https://github.com/kqwang/Dinomaly.git Dinomaly", cwd=ROOT, ignore_error=True)
    else:
        log("[SETUP] Dinomaly baseline repository already present.")

def setup_dataset():
    data_dir = ROOT / "data" / "mvtec_loco"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    categories = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
    if all((data_dir / cat).exists() for cat in categories):
        log("[SETUP] All dataset categories already present in data/mvtec_loco/")
        return

    log(f"[SETUP] Downloading MVTec LOCO AD dataset from Kaggle ({KAGGLE_DATASET})...")
    run_command(f"kaggle datasets download -d {KAGGLE_DATASET} -p {ROOT} --unzip", ignore_error=True)

    possible_source_dirs = [
        ROOT / "mvtec-loco",
        ROOT / "mvtec_loco",
        ROOT / "mvtec_loco_caption",
        ROOT
    ]
    
    for src in possible_source_dirs:
        if src.exists():
            for cat in categories:
                cat_src = src / cat
                cat_dst = data_dir / cat
                if cat_src.exists() and not cat_dst.exists():
                    log(f"[SETUP] Moving category '{cat}' -> data/mvtec_loco/{cat}")
                    shutil.move(str(cat_src), str(cat_dst))
                    
    for temp_dir in [ROOT / "mvtec-loco", ROOT / "mvtec_loco", ROOT / "mvtec_loco_caption"]:
        if temp_dir.exists() and temp_dir != data_dir:
            shutil.rmtree(str(temp_dir), ignore_errors=True)
            
    for zip_file in ROOT.glob("*.zip"):
        log(f"[SETUP] Cleaning zip file {zip_file.name} to free disk space...")
        try:
            zip_file.unlink()
        except Exception:
            pass

    log("[SETUP] Dataset structure verified inside data/mvtec_loco!")

def main():
    log("=" * 60)
    log("[SETUP] AUTOMATED CLOUD ENVIRONMENT SETUP FOR MVTEC LOCO AD")
    log("=" * 60)
    
    setup_requirements()
    setup_dinomaly_baseline()
    setup_dataset()
    
    log("=" * 60)
    log("[SUCCESS] CLOUD SETUP COMPLETED! READY TO TRAIN & EVALUATE.")
    log("=" * 60)

if __name__ == "__main__":
    main()
  
