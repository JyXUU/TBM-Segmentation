import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# === Create directory to save visualizations ===
metrics_dir = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/S-2103-004858_PAS_1of2_crop"
os.makedirs(metrics_dir, exist_ok=True)

# === Fold-wise Metrics from Image ===
data = {
    "Fold": ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"],
    "Dice": [0.6127, 0.7608, 0.6816, 0.7121, 0.7672],
    "IoU": [0.4416, 0.6140, 0.5170, 0.5529, 0.6224],
    "Precision": [0.7933, 0.7780, 0.6301, 0.8186, 0.8223],
    "Sensitivity": [0.4990, 0.7444, 0.7423, 0.6300, 0.7191],
    "Specificity": [0.9792, 0.9660, 0.9303, 0.9777, 0.9751],
    "clDice": [0.7081, 0.8560, 0.7639, 0.7949, 0.8551],
    "MCC": [0.6161, 0.7238, 0.6682, 0.7137, 0.7350]
}
df = pd.DataFrame(data)
df_melt = df.melt(id_vars="Fold", var_name="Metric", value_name="Score")

# === Compute per-metric means and stds across folds ===
metric_names = df.columns[1:]  # exclude 'Fold'
fold_indices = np.arange(len(df))  # Fold 0 to Fold 4

plt.figure(figsize=(10, 6))
colors = sns.color_palette("Set2", len(metric_names))

for idx, metric in enumerate(metric_names):
    scores = df[metric].values
    mean = np.mean(scores)
    std = np.std(scores)

    # Line plot across folds
    plt.plot(fold_indices, scores, label=metric, marker='o', color=colors[idx])

    # Horizontal dashed mean line
    plt.hlines(mean, xmin=fold_indices[0], xmax=fold_indices[-1],
               linestyles='dashed', colors=colors[idx], alpha=0.6)

# === Format plot ===
plt.xticks(fold_indices, df["Fold"], rotation=0)
plt.ylim(0, 1)
plt.title("Metric Trends Across Folds", fontsize=15)
plt.xlabel("Fold", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.legend(loc='lower right', frameon=True, fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(metrics_dir, "lineplot_metric_trends.png"), dpi=300)
plt.close()

