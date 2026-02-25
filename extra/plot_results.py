import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_performance_metrics(csv_path, pca_value=None, overlay_pca=False):
    """
    Create performance plots showing F1 scores by frame thresholds for different configurations.
    
    Args:
        csv_path (str): Path to the CSV file containing performance data
        pca_value (int, optional): Filter by specific PCA value. If None, uses all data.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter by PCA value if specified
    if pca_value is not None:
        df = df[df['n_pca'] == pca_value]
    
    # Define frame thresholds and layers
    frame_thresholds = [1, 3, 5, 10, 15, 30, 60]
    layers = list(range(1, 13))  # Layers 1-12
    fps_values = [5, 15, 25]
    state_values = [5, 10, 20, 30]
    
    # Color scheme for layers 1-12
    colors = {
        1: '#1f77b4',
        2: '#ff7f0e',
        3: '#2ca02c',
        4: '#d62728',
        5: '#9467bd',
        6: '#8c564b',
        7: '#e377c2',
        8: '#7f7f7f',
        9: '#bcbd22',
        10: '#17becf',
        11: '#ff9896',
        12: '#c5b0d5'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    title = 'F1 Score Performance by Frame Thresholds'
    if pca_value is not None:
        title += f' (PCA={pca_value})'
    elif overlay_pca:
        title += ' (Overlay by PCA)'
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # Plot for each FPS and State combination
    for i, fps in enumerate(fps_values):
        for j, states in enumerate(state_values):
            ax = axes[i, j]
            
            # Filter data for current FPS and states
            fps_state_data = df[(df['fps_sample'] == fps) & (df['n_states'] == states)]
            
            # Precompute x-axis in seconds using source material fps (25 fps)
            seconds_values = [frame / 25 for frame in frame_thresholds]

            # Plot each layer
            if overlay_pca and pca_value is None:
                # Overlay all available PCA values using different linestyles
                unique_pcas = sorted(df['n_pca'].unique())
                linestyles = ['-', '--', ':', '-.']
                pca_to_style = {p: linestyles[idx % len(linestyles)] for idx, p in enumerate(unique_pcas)}

                for layer in layers:
                    layer_data_all = fps_state_data[fps_state_data['layer'] == layer]
                    if layer_data_all.empty:
                        continue
                    for pca in unique_pcas:
                        layer_data = layer_data_all[layer_data_all['n_pca'] == pca]
                        if layer_data.empty:
                            continue
                        # Extract F1 scores for each frame threshold
                        f1_scores = []
                        for frame in frame_thresholds:
                            col_name = f'frames_{frame}_f1'
                            if col_name in layer_data.columns:
                                f1_scores.append(layer_data[col_name].iloc[0])
                            else:
                                f1_scores.append(np.nan)
                        # Plot line with linestyle for PCA and color for layer
                        ax.plot(
                            seconds_values,
                            f1_scores,
                            color=colors[layer],
                            linestyle=pca_to_style[pca],
                            marker='o',
                            linewidth=2,
                            markersize=5,
                            label=f'Layer {layer} (PCA {pca})'
                        )
                # Build a secondary legend for PCA linestyles
                from matplotlib.lines import Line2D
                pca_legend_lines = [
                    Line2D([0], [0], color='black', linestyle=pca_to_style[p], label=f'PCA {p}')
                    for p in unique_pcas
                ]
            else:
                for layer in layers:
                    layer_data = fps_state_data[fps_state_data['layer'] == layer]
                    if not layer_data.empty:
                        # Extract F1 scores for each frame threshold
                        f1_scores = []
                        for frame in frame_thresholds:
                            col_name = f'frames_{frame}_f1'
                            if col_name in layer_data.columns:
                                f1_scores.append(layer_data[col_name].iloc[0])
                            else:
                                f1_scores.append(np.nan)
                        # Plot line
                        ax.plot(
                            seconds_values,
                            f1_scores,
                            color=colors[layer],
                            marker='o',
                            linewidth=2,
                            markersize=6,
                            label=f'Layer {layer}'
                        )
            
            # Customize subplot
            ax.set_title(f'FPS={fps}, States={states}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Tolerance (s)')
            ax.set_ylabel('F1 Score')
            ax.grid(True, alpha=0.3)
            # Legends
            if overlay_pca and pca_value is None:
                # First legend for layers (avoid duplicates by using unique labels)
                handles, labels = ax.get_legend_handles_labels()
                # Deduplicate labels while keeping order
                seen = set()
                unique_handles_labels = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l.split(' (PCA')[0]))]
                # Keep only layer names in legend
                layer_seen = set()
                layer_handles_labels = []
                for h, l in unique_handles_labels:
                    layer_name = l.split(' (PCA')[0]
                    if layer_name not in layer_seen:
                        layer_seen.add(layer_name)
                        layer_handles_labels.append((h, layer_name))
                if layer_handles_labels:
                    ax.legend([h for h, _ in layer_handles_labels], [l for _, l in layer_handles_labels], loc='upper left', fontsize=8)
                # Add PCA linestyle legend
                if 'pca_legend_lines' in locals() and pca_legend_lines:
                    ax.add_artist(ax.legend(pca_legend_lines, [ln.get_label() for ln in pca_legend_lines], loc='lower right', fontsize=8, title='PCA'))
            else:
                ax.legend()
            ax.set_ylim(0, 0.8)
            
            # Set x-axis ticks (seconds)
            ax.set_xticks(seconds_values)
            ax.set_xticklabels([f"{x:.2f}" for x in seconds_values])
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_performance_metrics_separate(csv_path, pca_value=None):
    """
    Create separate plots for each FPS value.
    
    Args:
        csv_path (str): Path to the CSV file containing performance data
        pca_value (int, optional): Filter by specific PCA value. If None, uses all data.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter by PCA value if specified
    if pca_value is not None:
        df = df[df['n_pca'] == pca_value]
    
    # Define frame thresholds and layers
    frame_thresholds = [1, 3, 5, 10, 15, 30, 60]
    layers = list(range(1, 13))  # Layers 1-12
    fps_values = [5, 15, 25]
    state_values = [5, 10, 20, 30]
    
    # Color scheme for layers 1-12
    colors = {
        1: '#1f77b4',
        2: '#ff7f0e',
        3: '#2ca02c',
        4: '#d62728',
        5: '#9467bd',
        6: '#8c564b',
        7: '#e377c2',
        8: '#7f7f7f',
        9: '#bcbd22',
        10: '#17becf',
        11: '#ff9896',
        12: '#c5b0d5'
    }
    
    # Create separate figure for each FPS
    figures = []
    
    for fps in fps_values:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        title = f'F1 Score Performance - FPS = {fps}'
        if pca_value is not None:
            title += f' (PCA={pca_value})'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for j, states in enumerate(state_values):
            ax = axes[j]
            
            # Filter data for current FPS and states
            fps_state_data = df[(df['fps_sample'] == fps) & (df['n_states'] == states)]
            
            # Precompute x-axis in seconds using source material fps (25 fps)
            seconds_values = [frame / 25 for frame in frame_thresholds]

            # Plot each layer
            for layer in layers:
                layer_data = fps_state_data[fps_state_data['layer'] == layer]
                
                if not layer_data.empty:
                    # Extract F1 scores for each frame threshold
                    f1_scores = []
                    for frame in frame_thresholds:
                        col_name = f'frames_{frame}_f1'
                        if col_name in layer_data.columns:
                            f1_scores.append(layer_data[col_name].iloc[0])
                        else:
                            f1_scores.append(np.nan)
                    
                    # Plot line
                    ax.plot(seconds_values, f1_scores, 
                           color=colors[layer], marker='o', linewidth=2, 
                           markersize=6, label=f'Layer {layer}')
            
            # Customize subplot
            ax.set_title(f'{states} States', fontsize=12, fontweight='bold')
            ax.set_xlabel('Tolerance (s)')
            ax.set_ylabel('F1 Score')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 0.8)
            
            # Set x-axis ticks (seconds)
            ax.set_xticks(seconds_values)
            ax.set_xticklabels([f"{x:.2f}" for x in seconds_values])
        
        plt.tight_layout()
        plt.show()
        figures.append(fig)
    
    return figures

def plot_performance_metrics_by_states(csv_path, pca_value=None):
    """
    Create separate plots for each number of states value, varying FPS across subplots.
    
    Args:
        csv_path (str): Path to the CSV file containing performance data
        pca_value (int, optional): Filter by specific PCA value. If None, uses all data.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter by PCA value if specified
    if pca_value is not None:
        df = df[df['n_pca'] == pca_value]
    
    # Define frame thresholds and layers
    frame_thresholds = [1, 3, 5, 10, 15, 30, 60]
    layers = list(range(1, 13))  # Layers 1-12
    fps_values = [5, 15, 25]
    state_values = [5, 10, 20, 30]
    
    # Color scheme for layers 1-12
    colors = {
        1: '#1f77b4',
        2: '#ff7f0e',
        3: '#2ca02c',
        4: '#d62728',
        5: '#9467bd',
        6: '#8c564b',
        7: '#e377c2',
        8: '#7f7f7f',
        9: '#bcbd22',
        10: '#17becf',
        11: '#ff9896',
        12: '#c5b0d5'
    }
    
    # Create separate figure for each states value
    figures = []
    
    for states in state_values:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        title = f'F1 Score Performance - States = {states}'
        if pca_value is not None:
            title += f' (PCA={pca_value})'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for j, fps in enumerate(fps_values):
            ax = axes[j]
            
            # Filter data for current FPS and states
            fps_state_data = df[(df['fps_sample'] == fps) & (df['n_states'] == states)]
            
            # Precompute x-axis in seconds using source material fps (25 fps)
            seconds_values = [frame / 25 for frame in frame_thresholds]

            # Plot each layer
            for layer in layers:
                layer_data = fps_state_data[fps_state_data['layer'] == layer]
                
                if not layer_data.empty:
                    # Extract F1 scores for each frame threshold
                    f1_scores = []
                    for frame in frame_thresholds:
                        col_name = f'frames_{frame}_f1'
                        if col_name in layer_data.columns:
                            f1_scores.append(layer_data[col_name].iloc[0])
                        else:
                            f1_scores.append(np.nan)
                    
                    # Plot line
                    ax.plot(seconds_values, f1_scores, 
                           color=colors[layer], marker='o', linewidth=2, 
                           markersize=6, label=f'Layer {layer}')
            
            # Customize subplot
            ax.set_title(f'FPS {fps}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Tolerance (s)')
            ax.set_ylabel('F1 Score')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 0.8)
            
            # Set x-axis ticks (seconds)
            ax.set_xticks(seconds_values)
            ax.set_xticklabels([f"{x:.2f}" for x in seconds_values])
        
        plt.tight_layout()
        plt.show()
        figures.append(fig)
    
    return figures

def analyze_best_configurations(csv_path, pca_value=None):
    """
    Analyze and print the best performing configurations.
    
    Args:
        csv_path (str): Path to the CSV file containing performance data
        pca_value (int, optional): Filter by specific PCA value. If None, uses all data.
    """
    df = pd.read_csv(csv_path)
    
    # Filter by PCA value if specified
    if pca_value is not None:
        df = df[df['n_pca'] == pca_value]
    
    # Get F1 score columns
    f1_cols = [col for col in df.columns if col.endswith('_f1')]
    
    pca_text = f" (PCA={pca_value})" if pca_value is not None else ""
    print(f"=== Best Performing Configurations{pca_text} ===\n")
    
    for col in f1_cols:
        frame_num = col.split('_')[1]
        best_row = df.loc[df[col].idxmax()]
        
        print(f"Frame Threshold {frame_num}:")
        print(f"  Best F1 Score: {best_row[col]:.4f}")
        print(f"  Configuration: FPS={best_row['fps_sample']}, States={best_row['n_states']}, Layer={best_row['layer']}, PCA={best_row['n_pca']}")
        print()

def summarize_run_combinations(csv_path):
    """
    Summarize available run combinations (model, fps, states, layer, PCA).
    Prints unique values and basic counts to help decide plotting options.

    Args:
        csv_path (str): Path to the CSV file containing performance data
    """
    df = pd.read_csv(csv_path)

    unique_models = sorted(df['model'].unique()) if 'model' in df.columns else []
    unique_fps = sorted(df['fps_sample'].unique())
    unique_states = sorted(df['n_states'].unique())
    unique_layers = sorted(df['layer'].unique())
    unique_pcas = sorted(df['n_pca'].unique()) if 'n_pca' in df.columns else []

    print("=== Run Combinations Summary ===")
    print(f"Rows: {len(df):,}")
    if unique_models:
        print(f"Models ({len(unique_models)}): {unique_models}")
    print(f"FPS ({len(unique_fps)}): {unique_fps}")
    print(f"States ({len(unique_states)}): {unique_states}")
    print(f"Layers ({len(unique_layers)}): min={min(unique_layers)}, max={max(unique_layers)} (count={len(unique_layers)})")
    if unique_pcas:
        print(f"PCA values ({len(unique_pcas)}): {unique_pcas}")

    # Coverage table: for each (fps, states) how many layers per PCA are present
    if unique_pcas:
        print("\nCoverage by (FPS, States, PCA) -> unique layers count")
        coverage = (
            df.groupby(['fps_sample', 'n_states', 'n_pca'])['layer']
            .nunique()
            .reset_index()
            .sort_values(['fps_sample', 'n_states', 'n_pca'])
        )
        for _, row in coverage.iterrows():
            print(f"  FPS={int(row['fps_sample'])}, States={int(row['n_states'])}, PCA={int(row['n_pca'])}: layers={int(row['layer'])}")

def plot_heatmap_best_layers(csv_path, pca_value=None, facet_pca=False):
    """
    Create heatmap showing best performing layer for each FPS/States combination.
    
    Args:
        csv_path (str): Path to the CSV file containing performance data
        pca_value (int, optional): Filter by specific PCA value. If None, uses all data.
    """
    import seaborn as sns
    
    df = pd.read_csv(csv_path)
    
    # If a specific PCA is requested, filter to it
    if pca_value is not None:
        df = df[df['n_pca'] == pca_value]

    # Focus on frame_30 F1 scores (best performance)
    frame_col = 'frames_30_f1'

    # Determine PCA values to facet if requested
    if facet_pca and pca_value is None and 'n_pca' in df.columns:
        pca_values = sorted(df['n_pca'].unique())
        ncols = 2 * len(pca_values)
        fig, axes = plt.subplots(1, ncols, figsize=(9 * len(pca_values), 5))
        if ncols == 2:
            axes = [axes[0], axes[1]]
        else:
            axes = list(axes)

        for idx, pca in enumerate(pca_values):
            sub = df[df['n_pca'] == pca]
            results = []
            for fps in [5, 15, 25]:
                for states in [5, 10, 20, 30]:
                    subset = sub[(sub['fps_sample'] == fps) & (sub['n_states'] == states)]
                    if not subset.empty:
                        best_idx = subset[frame_col].idxmax()
                        best_layer = subset.loc[best_idx, 'layer']
                        best_score = subset.loc[best_idx, frame_col]
                        results.append({'FPS': fps, 'States': states, 'Best_Layer': best_layer, 'Best_F1': best_score})
            results_df = pd.DataFrame(results)

            ax_layer = axes[2 * idx]
            ax_f1 = axes[2 * idx + 1]

            pivot_layer = results_df.pivot(index='FPS', columns='States', values='Best_Layer')
            sns.heatmap(pivot_layer, annot=True, fmt='g', cmap='Set3', ax=ax_layer)
            ax_layer.set_title(f'Best Layer (Frames=30) - PCA={pca}')

            pivot_f1 = results_df.pivot(index='FPS', columns='States', values='Best_F1')
            sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax_f1)
            ax_f1.set_title(f'Best F1 (Frames=30) - PCA={pca}')

        plt.tight_layout()
        plt.show()
        return fig

    # Non-faceted (single set) behavior
    results = []
    for fps in [5, 15, 25]:
        for states in [5, 10, 20, 30]:
            subset = df[(df['fps_sample'] == fps) & (df['n_states'] == states)]
            if not subset.empty:
                best_idx = subset[frame_col].idxmax()
                best_layer = subset.loc[best_idx, 'layer']
                best_score = subset.loc[best_idx, frame_col]
                results.append({'FPS': fps, 'States': states, 'Best_Layer': best_layer, 'Best_F1': best_score})

    results_df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    pivot_layer = results_df.pivot(index='FPS', columns='States', values='Best_Layer')
    sns.heatmap(pivot_layer, annot=True, fmt='g', cmap='Set3', ax=ax1)
    title1 = 'Best Performing Layer (Frame Threshold = 30)'
    if pca_value is not None:
        title1 += f' - PCA={pca_value}'
    ax1.set_title(title1)

    pivot_f1 = results_df.pivot(index='FPS', columns='States', values='Best_F1')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
    title2 = 'Best F1 Scores (Frame Threshold = 30)'
    if pca_value is not None:
        title2 += f' - PCA={pca_value}'
    ax2.set_title(title2)

    plt.tight_layout()
    plt.show()
    return fig

def plot_multi_model_comparison(
    csv_paths,
    model_names,
    fps_sample: int,
    n_states: int,
    n_pca: int,
    tolerances_frames=None,
    fps_for_seconds: float = 25.0,
    layer_strategy: str = 'best',
    output_png: str = None,
    random_baseline: bool = False,
    baseline_secs=(0.4, 1.0),
    baseline_f1s=(0.32, 0.55),
):
    if tolerances_frames is None:
        tolerances_frames = [1, 3, 5, 10, 15, 30, 60]

    assert len(csv_paths) == len(model_names), "csv_paths and model_names must have the same length"

    seconds_values = [f / float(fps_for_seconds) for f in tolerances_frames]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Multi-Model Comparison (fps={fps_sample}, states={n_states}, pca={n_pca})")
    ax.set_xlabel('Tolerance (s)')
    ax.set_ylabel('F1 Score')
    ax.grid(True, alpha=0.3)

    for csv_path, model_name in zip(csv_paths, model_names):
        df = pd.read_csv(csv_path)
        subset = df[(df.get('fps_sample') == fps_sample) & (df.get('n_states') == n_states) & (df.get('n_pca') == n_pca)]
        if subset.empty:
            print(f"Warning: no rows for {model_name} in {csv_path} with fps={fps_sample}, states={n_states}, pca={n_pca}")
            continue

        f1_cols = [f"frames_{t}_f1" for t in tolerances_frames if f"frames_{t}_f1" in subset.columns]
        if not f1_cols:
            print(f"Warning: no matching F1 columns found in {csv_path} for {model_name}")
            continue

        if layer_strategy == 'best':
            means = subset[f1_cols].mean(axis=1)
            best_idx = means.idxmax()
            best_row = subset.loc[best_idx]
            y_vals = []
            for t in tolerances_frames:
                col = f"frames_{t}_f1"
                y_vals.append(best_row[col] if col in subset.columns else np.nan)
        elif layer_strategy == 'mean':
            y_vals = []
            for t in tolerances_frames:
                col = f"frames_{t}_f1"
                y_vals.append(subset[col].mean() if col in subset.columns else np.nan)
        else:
            raise ValueError("layer_strategy must be 'best' or 'mean'")

        ax.plot(seconds_values, y_vals, marker='o', linewidth=2, markersize=6, label=model_name)

    if random_baseline:
        # Plot provided baseline points in seconds/F1 space
        try:
            bx = list(baseline_secs)
            by = list(baseline_f1s)
            ax.plot(bx, by, linestyle='--', color='black', linewidth=1.5, alpha=0.7)
            ax.scatter(bx, by, color='black', marker='x', s=80, label='Random baseline')
        except Exception:
            # If malformed baseline inputs are provided, skip silently
            pass

    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.set_xticks(seconds_values)
    ax.set_xticklabels([f"{x:.2f}" for x in seconds_values])

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Saved multi-model comparison to {output_png}")

    plt.tight_layout()
    plt.show()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot grid search results')
    parser.add_argument('--csv_path', type=str, default=None, help='Single CSV path for classic plots')
    parser.add_argument('--multi_model', action='store_true', help='Enable multi-model comparison mode')
    parser.add_argument('--csv_paths', nargs='+', type=str, default=None, help='CSV paths for multi-model mode')
    parser.add_argument('--model_names', nargs='+', type=str, default=None, help='Model names for legend (same length as csv_paths)')
    parser.add_argument('--mm_fps', type=int, default=25, help='fps_sample to filter in multi-model mode')
    parser.add_argument('--mm_states', type=int, default=10, help='n_states to filter in multi-model mode')
    parser.add_argument('--mm_pca', type=int, default=20, help='n_pca to filter in multi-model mode')
    parser.add_argument('--mm_tolerances', nargs='+', type=int, default=[1,3,5,10,15,30,60], help='Frame tolerances to plot')
    parser.add_argument('--mm_video_fps', type=float, default=25.0, help='FPS used to convert frames->seconds on x-axis')
    parser.add_argument('--mm_layer_strategy', type=str, choices=['best','mean'], default='best', help='Aggregate across layers by best or mean')
    parser.add_argument('--mm_output_png', type=str, default=None, help='Optional output path to save the multi-model plot')
    parser.add_argument('--mm_random_baseline', action='store_true', help='Overlay random baseline points at 0.4s->0.32 F1 and 1.0s->0.55 F1')
    parser.add_argument('--mm_baseline_secs', nargs='+', type=float, default=None, help='Optional override for baseline seconds (e.g., 0.4 1.0)')
    parser.add_argument('--mm_baseline_f1s', nargs='+', type=float, default=None, help='Optional override for baseline F1s (e.g., 0.32 0.55)')

    args = parser.parse_args()

    if args.multi_model:
        if not args.csv_paths or not args.model_names:
            raise SystemExit('--csv_paths and --model_names are required in multi-model mode')
        plot_multi_model_comparison(
            csv_paths=args.csv_paths,
            model_names=args.model_names,
            fps_sample=args.mm_fps,
            n_states=args.mm_states,
            n_pca=args.mm_pca,
            tolerances_frames=args.mm_tolerances,
            fps_for_seconds=args.mm_video_fps,
            layer_strategy=args.mm_layer_strategy,
            output_png=args.mm_output_png,
            random_baseline=args.mm_random_baseline,
            baseline_secs=tuple(args.mm_baseline_secs) if args.mm_baseline_secs else (0.4, 1.0),
            baseline_f1s=tuple(args.mm_baseline_f1s) if args.mm_baseline_f1s else (0.32, 0.55),
        )
    else:
        csv_path = args.csv_path or "/pfss/mlde/workspaces/mlde_wsp_ChildScope/br/boundry_segmentation/sherlock_clip_vit_l/grid_summary.csv"
        if not Path(csv_path).exists():
            print(f"Error: File {csv_path} not found!")
            raise SystemExit(1)
        summarize_run_combinations(csv_path)
        df = pd.read_csv(csv_path)
        pca_values = sorted(df['n_pca'].unique())
        print(f"Found PCA values: {pca_values}\n")
        print("Creating overlay plot with all PCA values...")
        _ = plot_performance_metrics(csv_path, pca_value=None, overlay_pca=True)
        for pca in pca_values:
            print(f"\n{'='*60}")
            print(f"Processing PCA = {pca}")
            print(f"{'='*60}\n")
            fig1 = plot_performance_metrics(csv_path, pca_value=pca)
            figs = plot_performance_metrics_separate(csv_path, pca_value=pca)
            analyze_best_configurations(csv_path, pca_value=pca)
            fig_heatmap = plot_heatmap_best_layers(csv_path, pca_value=pca)

# Additional utility function to save plots
def save_plots(csv_path, output_dir="plots"):
    """
    Create and save all plots to files, separated by PCA value.
    
    Args:
        csv_path (str): Path to the CSV file
        output_dir (str): Directory to save plots
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique PCA values
    df = pd.read_csv(csv_path)
    pca_values = sorted(df['n_pca'].unique())
    
    print(f"Saving plots for PCA values: {pca_values}")
    
    # Create plots for each PCA value
    for pca in pca_values:
        print(f"\nSaving plots for PCA={pca}...")
        
        # Main grid plot
        fig1 = plot_performance_metrics(csv_path, pca_value=pca)
        fig1.savefig(f"{output_dir}/performance_grid_pca{pca}.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Separate FPS plots
        figs = plot_performance_metrics_separate(csv_path, pca_value=pca)
        for i, fig in enumerate(figs):
            fps_val = [5, 15, 25][i]
            fig.savefig(f"{output_dir}/performance_fps_{fps_val}_pca{pca}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Separate States plots (vary FPS; fixed states)
        figs_states = plot_performance_metrics_by_states(csv_path, pca_value=pca)
        for i, fig in enumerate(figs_states):
            states_val = [5, 10, 20, 30][i]
            fig.savefig(f"{output_dir}/performance_states_{states_val}_pca{pca}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Heatmap
        fig_heat = plot_heatmap_best_layers(csv_path, pca_value=pca)
        fig_heat.savefig(f"{output_dir}/performance_heatmap_pca{pca}.png", dpi=300, bbox_inches='tight')
        plt.close(fig_heat)
    
    print(f"\nAll plots saved to {output_dir}/ directory")