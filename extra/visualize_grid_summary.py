#!/usr/bin/env python3
"""
Visualize grid search summary CSV as heatmaps and top-k table.

Creates heatmaps of F1 by (layer, n_states) for each fps_sample and tolerance,
optionally faceted by n_pca if present. Also saves a top-k CSV with the best
rows by a chosen tolerance.
"""

import argparse
import os
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def heatmap_f1(df: pd.DataFrame, fps_sample: int, tol_key: str, out_dir: str, pca_value: int | None):
    sub = df[df['fps_sample'] == fps_sample].copy()
    if pca_value is not None and 'n_pca' in sub.columns:
        sub = sub[sub['n_pca'] == pca_value]
        suffix = f"_pca{pca_value}"
    else:
        suffix = ""
    if sub.empty:
        return
    pivot = sub.pivot_table(index='layer', columns='n_states', values=f'{tol_key}_f1', aggfunc='mean')
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1)
    plt.title(f'F1 ({tol_key}) | fps_sample={fps_sample}{suffix}')
    plt.ylabel('layer')
    plt.xlabel('n_states')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'heatmap_{tol_key}_fps{fps_sample}{suffix}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize grid summary CSV')
    parser.add_argument('summary_csv', type=str, help='Path to grid_summary.csv')
    parser.add_argument('--out_dir', type=str, default='grid_viz', help='Output directory for plots')
    parser.add_argument('--tolerances', nargs='+', type=str, default=['frames_5','frames_10','frames_15','frames_30'])
    parser.add_argument('--top_k', type=int, default=10, help='Rows to include in top-k CSV')
    parser.add_argument('--top_tol', type=str, default='frames_15', help='Tolerance key to rank by (e.g., frames_15)')
    parser.add_argument('--pca_values', nargs='*', type=int, default=None, help='Specific PCA values to facet; leave empty to auto')

    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)

    # Save top-k by selected tolerance
    score_col = f'{args.top_tol}_f1'
    if score_col not in df.columns:
        raise ValueError(f"Column {score_col} not found in summary CSV")
    top = df.sort_values(by=score_col, ascending=False).head(args.top_k)
    os.makedirs(args.out_dir, exist_ok=True)
    top_path = os.path.join(args.out_dir, 'top_k.csv')
    top.to_csv(top_path, index=False)
    print(f"Saved top-k table to {top_path}")

    # Determine PCA facets
    pca_values: List[int] | None
    if 'n_pca' in df.columns:
        if args.pca_values:
            pca_values = args.pca_values
        else:
            pca_values = sorted(df['n_pca'].dropna().unique().tolist())
    else:
        pca_values = [None]

    # Heatmaps
    for tol_key in args.tolerances:
        if f'{tol_key}_f1' not in df.columns:
            continue
        for fps in sorted(df['fps_sample'].dropna().unique().tolist()):
            for pca in pca_values:
                heatmap_f1(df, fps, tol_key, args.out_dir, pca)


if __name__ == '__main__':
    main()


