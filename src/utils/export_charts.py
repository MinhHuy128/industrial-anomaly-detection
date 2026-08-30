import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
def main():
    out_dir = ROOT / 'docs' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_file = ROOT / 'results' / 'benchmark_official.json'
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Missing benchmark file: {benchmark_file}")

    with open(benchmark_file, 'r', encoding='utf-8') as f:
        bench_data = json.load(f)

    cat_keys = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors', 'MEAN']
    display_names = ['Breakfast Box', 'Juice Bottle', 'Pushpins', 'Screw Bag', 'Splicing Conn.', 'MEAN']

    baseline_logical = [bench_data['baseline'][k]['logical_auroc'] for k in cat_keys]
    gct_v2_logical = [bench_data['gct'][k]['logical_auroc'] for k in cat_keys]

    x = np.arange(len(display_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, baseline_logical, width, label='Comparative Baseline', color='#7f7f7f', alpha=0.85)
    rects2 = ax.bar(x + width/2, gct_v2_logical, width, label='ViTill-GCT V2 (Proposed)', color='#1f77b4', alpha=0.9)

    ax.set_ylabel('Logical Anomaly AUROC (%)', fontsize=12, fontweight='bold')
    ax.set_title('Logical Anomaly Detection AUROC (Baseline vs. ViTill-GCT V2)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(45, 102)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0b559f')

    plt.tight_layout()
    chart1_file = out_dir / 'bar_chart_logical_auroc.png'
    plt.savefig(chart1_file, dpi=300)
    plt.close()

    gamma_file = ROOT / 'results' / 'gamma_sweep_study.json'
    if not gamma_file.exists():
        raise FileNotFoundError(f"Missing gamma study file: {gamma_file}")

    with open(gamma_file, 'r', encoding='utf-8') as f:
        gamma_data = json.load(f)

    gammas = [p['gamma'] for p in gamma_data['sweep_points']]
    mean_auroc_trend = [p['mean_auroc'] for p in gamma_data['sweep_points']]
    baseline_ref = gamma_data.get('baseline_reference', 84.67)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gammas, mean_auroc_trend, marker='o', color='#1f77b4', linewidth=2.5, markersize=7, label='Mean AUROC (%)')
    ax.axhline(y=baseline_ref, color='gray', linestyle='--', label=f'Baseline Benchmark ({baseline_ref:.2f}%)')

    ax.set_xlabel('GCT Gamma Weight (gamma)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Mean AUROC (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Gamma Weight on Overall Mean AUROC', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.set_ylim(84.0, 88.0)

    for g, m in zip(gammas[::2], mean_auroc_trend[::2]):
        ax.annotate(f'{m:.2f}%', (g, m), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart2_file = out_dir / 'line_chart_gamma_sensitivity.png'
    plt.savefig(chart2_file, dpi=300)
    plt.close()

    print(f"Saved charts to {out_dir}")


if __name__ == '__main__':
    main()
