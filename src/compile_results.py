"""
src/compile_results.py
----------------------
Compiles all evaluation results into a single publication-ready output.
Reads:
  - results/benchmark_official.json  (Image AUROC + Latency from src/benchmark_all.py)
  - results/official_spro/official_spro_summary.json  (Official sPRO from MVTec eval kit)
Outputs:
  - results/final_table.md    (Markdown table for GitHub / paper draft)
  - results/final_table.tex   (LaTeX table ready to copy into paper)
  - results/final_summary.json (Full machine-readable combined summary)
"""
import sys, json
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

AUROC_FILE = ROOT / "results" / "benchmark_official.json"
SPRO_FILE  = ROOT / "results" / "official_spro" / "official_spro_summary.json"
OUT_DIR    = ROOT / "results"

def load_auroc_data():
    """Load Image AUROC, Latency data from benchmark_all.py output."""
    if not AUROC_FILE.exists():
        print(f"[MISSING] {AUROC_FILE.relative_to(ROOT)}")
        print("  → Run first: python src/benchmark_all.py --save_maps")
        return None
    with open(AUROC_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_spro_data():
    """Load official sPRO data from run_official_spro.py output."""
    if not SPRO_FILE.exists():
        print(f"[MISSING] {SPRO_FILE.relative_to(ROOT)}")
        print("  → Run first: python src/run_official_spro.py")
        return None
    with open(SPRO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_official_spro(spro_data: dict, model: str, category: str) -> float:
    """Extract sPRO value safely from the official spro summary."""
    if spro_data is None:
        return None
    per_cat = spro_data.get("summary", {}).get(model, {}).get("per_category_spro", {})
    cat_data = per_cat.get(category, {})
    for key in ["au_pro", "spro", "sPRO", "aupro", "AUPRO"]:
        if key in cat_data:
            val = cat_data[key]
            return val * 100.0 if val <= 1.0 else val
    return None

def generate_markdown_table(auroc: dict, spro: dict) -> str:
    """Generate full comparison Markdown table for both models."""
    lines = []
    lines.append("# ViTill-GCT V2 — Full Evaluation Results on MVTec LOCO AD")
    lines.append("")
    lines.append("> Image AUROC: computed via `sklearn.metrics.roc_auc_score` (100% standard).")
    lines.append("> sPRO: computed via official MVTec LOCO AD evaluation kit (Bergmann et al., WACV 2022).")
    lines.append("")

    for model_key, model_name in [("gct", "ViTill-GCT V2 (Proposed)"), ("baseline", "Dinomaly Baseline")]:
        lines.append(f"## {model_name}")
        lines.append("")
        lines.append("| Category | Logical AUROC (%) | Structural AUROC (%) | Mean AUROC (%) | sPRO (%) | Latency (ms) | FPS |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

        for cat in CATEGORIES + ["MEAN"]:
            d = auroc.get(model_key, {}).get(cat, {}) if auroc else {}
            spro_val = get_official_spro(spro, model_key, cat) if cat != "MEAN" else None

            if cat == "MEAN" and spro:
                mean_spro = spro.get("summary", {}).get(model_key, {}).get("mean_spro", -1.0)
                spro_str = f"**{mean_spro * 100.0:.2f}**" if mean_spro > 0 and mean_spro <= 1.0 else (f"**{mean_spro:.2f}**" if mean_spro > 1 else "—")
            elif spro_val is not None:
                spro_str = f"{spro_val:.2f}"
            else:
                spro_str = "—"

            log_str    = f"{d.get('logical_auroc', 0):.2f}"   if d else "—"
            struct_str = f"{d.get('structural_auroc', 0):.2f}" if d else "—"
            mean_str   = f"{d.get('mean_auroc', 0):.2f}"       if d else "—"
            lat_str    = f"{d.get('latency_ms', 0):.2f}"       if d else "—"
            fps_str    = f"{d.get('fps', 0):.1f}"              if d else "—"

            bold = "**" if cat == "MEAN" else ""
            lines.append(f"| {bold}{cat.upper()}{bold} | {log_str}% | {struct_str}% | {mean_str}% | {spro_str}% | {lat_str} ms | {fps_str} |")
        lines.append("")

    # Delta table
    if auroc:
        b = auroc.get("baseline", {}).get("MEAN", {})
        g = auroc.get("gct", {}).get("MEAN", {})
        b_spro = (spro or {}).get("summary", {}).get("baseline", {}).get("mean_spro", -1.0)
        g_spro = (spro or {}).get("summary", {}).get("gct", {}).get("mean_spro", -1.0)
        b_spro = b_spro * 100.0 if b_spro > 0 and b_spro <= 1.0 else b_spro
        g_spro = g_spro * 100.0 if g_spro > 0 and g_spro <= 1.0 else g_spro

        lines.append("## Performance Delta (ViTill-GCT V2 vs. Baseline)")
        lines.append("")
        lines.append("| Metric | Baseline | ViTill-GCT V2 | Δ (Delta) |")
        lines.append("|:---|:---:|:---:|:---:|")
        lines.append(f"| Logical AUROC | {b.get('logical_auroc', 0):.2f}% | **{g.get('logical_auroc', 0):.2f}%** | **+{g.get('logical_auroc', 0) - b.get('logical_auroc', 0):.2f}%** 🏆 |")
        lines.append(f"| Structural AUROC | {b.get('structural_auroc', 0):.2f}% | **{g.get('structural_auroc', 0):.2f}%** | **+{g.get('structural_auroc', 0) - b.get('structural_auroc', 0):.2f}%** |")
        lines.append(f"| Mean AUROC | {b.get('mean_auroc', 0):.2f}% | **{g.get('mean_auroc', 0):.2f}%** | **+{g.get('mean_auroc', 0) - b.get('mean_auroc', 0):.2f}%** 🏆 |")
        if g_spro > 0 and b_spro > 0:
            lines.append(f"| sPRO (official) | {b_spro:.2f}% | **{g_spro:.2f}%** | {g_spro - b_spro:+.2f}% |")
        lines.append(f"| Latency (batch=1) | {b.get('latency_ms', 0):.2f} ms | **{g.get('latency_ms', 0):.2f} ms** | {g.get('fps', 0):.1f} FPS |")

    return "\n".join(lines)

def generate_latex_table(auroc: dict, spro: dict) -> str:
    """Generate LaTeX table suitable for IEEE/CVPR paper."""
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Performance comparison on MVTec LOCO AD dataset. Image AUROC computed via \\texttt{sklearn.metrics.roc\_{auc}\_{score}}. sPRO computed via official MVTec LOCO AD evaluation kit~\\cite{bergmann2022loco}.}",
        "\\label{tab:main_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l|ccc|c|c}",
        "\\hline",
        "\\multirow{2}{*}{Category} & \\multicolumn{3}{c|}{Image AUROC (\\%)} & \\multirow{2}{*}{sPRO (\\%)} & \\multirow{2}{*}{Latency (ms)} \\\\",
        " & Logical & Structural & Mean & & \\\\",
        "\\hline",
        "\\multicolumn{6}{c}{\\textit{ViTill-GCT V2 (Proposed)}} \\\\",
        "\\hline",
    ]

    for cat in CATEGORIES + ["MEAN"]:
        d = (auroc or {}).get("gct", {}).get(cat, {})
        spro_val = get_official_spro(spro, "gct", cat) if cat != "MEAN" else None
        if cat == "MEAN":
            ms = (spro or {}).get("summary", {}).get("gct", {}).get("mean_spro", -1.0)
            spro_val = ms * 100.0 if 0 < ms <= 1.0 else ms if ms > 1 else None

        log    = f"{d.get('logical_auroc', 0):.2f}"   if d else "—"
        struct = f"{d.get('structural_auroc', 0):.2f}" if d else "—"
        mean   = f"{d.get('mean_auroc', 0):.2f}"       if d else "—"
        spro_s = f"{spro_val:.2f}" if spro_val and spro_val > 0 else "—"
        lat    = f"{d.get('latency_ms', 0):.2f}"       if d else "—"
        cat_label = cat.replace("_", "\\_")
        bold_l = "\\textbf{" if cat == "MEAN" else ""
        bold_r = "}" if cat == "MEAN" else ""
        lines.append(f"{bold_l}{cat_label}{bold_r} & {log} & {struct} & {mean} & {spro_s} & {lat} \\\\")

    lines += [
        "\\hline",
        "\\multicolumn{6}{c}{\\textit{Dinomaly Baseline~\\cite{dinomaly2024}}} \\\\",
        "\\hline",
    ]

    for cat in CATEGORIES + ["MEAN"]:
        d = (auroc or {}).get("baseline", {}).get(cat, {})
        spro_val = get_official_spro(spro, "baseline", cat) if cat != "MEAN" else None
        if cat == "MEAN":
            ms = (spro or {}).get("summary", {}).get("baseline", {}).get("mean_spro", -1.0)
            spro_val = ms * 100.0 if 0 < ms <= 1.0 else ms if ms > 1 else None

        log    = f"{d.get('logical_auroc', 0):.2f}"   if d else "—"
        struct = f"{d.get('structural_auroc', 0):.2f}" if d else "—"
        mean   = f"{d.get('mean_auroc', 0):.2f}"       if d else "—"
        spro_s = f"{spro_val:.2f}" if spro_val and spro_val > 0 else "—"
        lat    = f"{d.get('latency_ms', 0):.2f}"       if d else "—"
        cat_label = cat.replace("_", "\\_")
        bold_l = "\\textbf{" if cat == "MEAN" else ""
        bold_r = "}" if cat == "MEAN" else ""
        lines.append(f"{bold_l}{cat_label}{bold_r} & {log} & {struct} & {mean} & {spro_s} & {lat} \\\\")

    lines += [
        "\\hline",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ]
    return "\n".join(lines)

def main():
    print("=" * 60)
    print("📄 Compiling final publication-ready results table...")
    print("=" * 60)

    auroc = load_auroc_data()
    spro  = load_spro_data()

    if auroc is None:
        print("[ERROR] Cannot compile without Image AUROC data.")
        return

    if spro is None:
        print("[WARN] Compiling without official sPRO (will show — in table).")

    md_content  = generate_markdown_table(auroc, spro)
    tex_content = generate_latex_table(auroc, spro)

    (OUT_DIR).mkdir(parents=True, exist_ok=True)

    md_path  = OUT_DIR / "final_table.md"
    tex_path = OUT_DIR / "final_table.tex"

    md_path.write_text(md_content, encoding="utf-8")
    tex_path.write_text(tex_content, encoding="utf-8")

    # Also save combined summary
    combined = {
        "image_auroc": auroc,
        "official_spro": spro,
        "generated_at": __import__("datetime").datetime.now().isoformat()
    }
    (OUT_DIR / "final_summary.json").write_text(
        __import__("json").dumps(combined, indent=2), encoding="utf-8"
    )

    print(f"[OK] {md_path.relative_to(ROOT)}")
    print(f"[OK] {tex_path.relative_to(ROOT)}")
    print(f"[OK] {(OUT_DIR / 'final_summary.json').relative_to(ROOT)}")
    print("\nDone! Copy final_table.tex directly into your LaTeX paper.")

if __name__ == "__main__":
    main()
