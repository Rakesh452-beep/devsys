import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from data import load_raw, clean, TCO_TARGETS, TCS_TARGETS, TCO2_TARGETS, REFLOW_TARGETS, CAL_CODE_TARGETS, FEATURE_BASE, PCM_COLS, POS_COLS, ALL_TARGETS, EXTREME_LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
EDA_DIR = os.path.join(BASE_DIR, 'outputs', 'eda')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

df = load_raw()
df, info = clean(df)
print('shape:', df.shape)

df.describe().to_csv(os.path.join(EDA_DIR, 'summary_stats.csv'))
df.corr(numeric_only=True).to_csv(os.path.join(EDA_DIR, 'correlation_matrix.csv'))

# 1. Target distributions by derivative
for target in TCO_TARGETS + TCS_TARGETS + TCO2_TARGETS:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    temps = ['-40C', '25C', '85C']
    for ax, temp in zip(axes, temps):
        subset = df[df['derivative'] == 'ACC-2G']
        ax.hist(subset[target], bins=30, alpha=0.7)
        ax.set_title(f'{target} ACC-2G {temp}')
        ax.set_xlabel(target)
        ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'{target}_by_derivative.png'), dpi=100)
    plt.close()
print('Done target distribution plots')

# 2. Feature-target correlation heatmap
feat_cols = [c for c in FEATURE_BASE + PCM_COLS + POS_COLS if c in df.columns]
target_cols = [c for c in ALL_TARGETS if c in df.columns]
corr = df[feat_cols + target_cols].corr(numeric_only=True).loc[feat_cols, target_cols]
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap='coolwarm', aspect='auto')
ax.set_xticks(range(len(target_cols)))
ax.set_xticklabels(target_cols, rotation=90)
ax.set_yticks(range(len(feat_cols)))
ax.set_yticklabels(feat_cols)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_target_correlation.png'), dpi=100)
plt.close()
print('Done correlation heatmap')

# 3. Offset distributions by temperature
for axis in ['x', 'y', 'z']:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    temps = ['m40C', '25C', '85C']
    labels = ['-40C', '25C', '85C']
    for ax, temp, label in zip(axes, temps, labels):
        col = f'offset_{axis}_{temp}_mg'
        if col in df.columns:
            ax.hist(df[col], bins=30, alpha=0.7)
            ax.set_title(f'{col} {label}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'offset_{axis}_by_temp.png'), dpi=100)
    plt.close()
print('Done offset by temp plots')
