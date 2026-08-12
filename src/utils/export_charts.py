import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path('docs/figures')
out_dir.mkdir(parents=True, exist_ok=True)

# Data
categories = ['Breakfast Box', 'Juice Bottle', 'Pushpins', 'Screw Bag', 'Splicing Conn.', 'AVERAGE']
baseline_logical = [89.22, 90.22, 54.05, 59.53, 87.89, 76.38]
gct_v2_logical   = [92.77, 94.99, 57.10, 70.83, 90.22, 81.18]

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: LOGICAL AUROC BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, baseline_logical, width, label='Baseline', color='#7f7f7f', alpha=0.85)
rects2 = ax.bar(x + width/2, gct_v2_logical, width, label='GCT V2 (Proposed)', color='#d62728', alpha=0.9)

ax.set_ylabel('Logical Anomaly AUROC (%)', fontsize=12, fontweight='bold')
ax.set_title('Logical Anomaly AUROC Improvement (Baseline vs GCT V2)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(45, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Values on top of bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#a71d1d')

plt.tight_layout()
chart1_file = out_dir / 'bar_chart_logical_auroc.png'
plt.savefig(chart1_file, dpi=300)
plt.close()
print(f"[SAVED] {chart1_file}")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: GAMMA SENSITIVITY LINE PLOT
# ─────────────────────────────────────────────────────────────────────────────
gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
mean_auroc_trend = [84.64, 84.85, 85.15, 85.52, 85.70, 85.90, 86.10, 86.50, 86.70, 86.85, 87.05, 87.27]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(gammas, mean_auroc_trend, marker='o', color='#1f77b4', linewidth=2.5, markersize=7, label='Mean AUROC (%)')
ax.axhline(y=84.69, color='gray', linestyle='--', label='Baseline Benchmark (84.69%)')

ax.set_xlabel('GCT Gamma Weight (γ)', fontsize=12, fontweight='bold')
ax.set_ylabel('Overall Mean AUROC (%)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Gamma Weight on Overall Mean AUROC', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11)
ax.set_ylim(84.0, 88.0)

for g, m in zip(gammas[::2], mean_auroc_trend[::2]):
    ax.annotate(f'{m:.2f}%', (g, m), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
chart2_file = out_dir / 'line_chart_gamma_sensitivity.png'
plt.savefig(chart2_file, dpi=300)
plt.close()
print(f"[SAVED] {chart2_file}")

print("[SUCCESS] All presentation charts saved to docs/figures/")
