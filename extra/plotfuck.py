#!/usr/bin/env python3
"""
plot_boundary_distribution.py
Visualize the distribution of predicted boundary indices across all videos.

Loads all prediction .pkl files and plots:
1. Histogram of absolute boundary positions
2. Histogram of normalized boundary positions (as fraction of video length)
3. Per-video boundary visualization showing clustering patterns
"""

import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def collect_predictions(pred_dir: str, downsample: int = 1, gt_path: str = None) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    Collect all predictions from .pkl files.
    Returns:
        - predictions: dict mapping vid_id -> list of boundary indices (in original frame space)
        - video_lengths: dict mapping vid_id -> num_frames (if GT available)
    """
    predictions = {}
    video_lengths = {}
    
    # Load GT if provided to get video lengths
    gt_dict = {}
    if gt_path and os.path.exists(gt_path):
        gt_dict = load_pickle(gt_path)
        print(f"[INFO] Loaded GT from {gt_path}")
    
    # Scan for all .pkl files
    pkl_files = [f for f in os.listdir(pred_dir) if f.endswith('.pkl')]
    print(f"[INFO] Found {len(pkl_files)} .pkl files in {pred_dir}")
    
    for pkl_file in sorted(pkl_files):
        vid_id = os.path.splitext(pkl_file)[0]
        pkl_path = os.path.join(pred_dir, pkl_file)
        
        try:
            pred_data = load_pickle(pkl_path)
            preds_ds = list(pred_data.get('bdy_idx_list_smt', []))
            preds_orig = [p * downsample for p in preds_ds]
            predictions[vid_id] = preds_orig
            
            # Get video length from GT if available
            if vid_id in gt_dict:
                num_frames = gt_dict[vid_id].get('num_frames')
                if num_frames is not None:
                    video_lengths[vid_id] = num_frames
        except Exception as e:
            print(f"[WARN] Failed to load {pkl_file}: {e}")
    
    print(f"[INFO] Successfully loaded predictions for {len(predictions)} videos")
    if video_lengths:
        print(f"[INFO] Got video lengths for {len(video_lengths)} videos")
    
    return predictions, video_lengths


def plot_distributions(predictions: Dict[str, List[int]], video_lengths: Dict[str, int], output_path: str = None):
    """Create visualization plots for boundary distributions."""
    
    # Collect all boundaries
    all_boundaries = []
    all_normalized = []
    
    for vid_id, boundaries in predictions.items():
        all_boundaries.extend(boundaries)
        
        # Normalize if we have video length
        if vid_id in video_lengths and video_lengths[vid_id] > 0:
            normalized = [b / video_lengths[vid_id] for b in boundaries]
            all_normalized.extend(normalized)
    
    # Create figure with subplots
    n_plots = 3 if all_normalized else 2
    fig = plt.figure(figsize=(15, 5 * n_plots))
    
    # Plot 1: Histogram of absolute positions
    ax1 = plt.subplot(n_plots, 1, 1)
    if all_boundaries:
        ax1.hist(all_boundaries, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Boundary Frame Index', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'Distribution of Boundary Positions (Absolute)\nTotal boundaries: {len(all_boundaries)}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_pos = np.mean(all_boundaries)
        median_pos = np.median(all_boundaries)
        ax1.axvline(mean_pos, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pos:.1f}')
        ax1.axvline(median_pos, color='green', linestyle='--', linewidth=2, label=f'Median: {median_pos:.1f}')
        ax1.legend()
    
    # Plot 2: Per-video boundary visualization
    ax2 = plt.subplot(n_plots, 1, 2)
    vid_ids = sorted(predictions.keys())
    
    for idx, vid_id in enumerate(vid_ids[:50]):  # Limit to 50 videos for readability
        boundaries = predictions[vid_id]
        y_pos = [idx] * len(boundaries)
        ax2.scatter(boundaries, y_pos, alpha=0.6, s=20)
    
    ax2.set_xlabel('Boundary Frame Index', fontsize=12)
    ax2.set_ylabel('Video Index', fontsize=12)
    ax2.set_title(f'Boundary Positions per Video (showing up to 50 videos)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Normalized positions (if available)
    if all_normalized:
        ax3 = plt.subplot(n_plots, 1, 3)
        ax3.hist(all_normalized, bins=50, edgecolor='black', alpha=0.7, range=(0, 1))
        ax3.set_xlabel('Normalized Boundary Position (0=start, 1=end)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title(f'Distribution of Boundary Positions (Normalized)\nTotal boundaries: {len(all_normalized)}', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        mean_norm = np.mean(all_normalized)
        median_norm = np.median(all_normalized)
        ax3.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.3f}')
        ax3.axvline(median_norm, color='green', linestyle='--', linewidth=2, label=f'Median: {median_norm:.3f}')
        ax3.legend()
        
        # Add quartile lines
        q1, q3 = np.percentile(all_normalized, [25, 75])
        ax3.axvline(q1, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q1: {q1:.3f}')
        ax3.axvline(q3, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q3: {q3:.3f}')
        ax3.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved plot to {output_path}")
    
    plt.show()
    return fig


def print_statistics(predictions: Dict[str, List[int]], video_lengths: Dict[str, int]):
    """Print summary statistics about the boundary distributions."""
    
    print("\n" + "="*70)
    print("BOUNDARY DISTRIBUTION STATISTICS")
    print("="*70)
    
    # Overall stats
    all_boundaries = []
    boundaries_per_video = []
    
    for vid_id, boundaries in predictions.items():
        all_boundaries.extend(boundaries)
        boundaries_per_video.append(len(boundaries))
    
    print(f"\nTotal videos: {len(predictions)}")
    print(f"Total boundaries: {len(all_boundaries)}")
    print(f"Boundaries per video: min={min(boundaries_per_video)}, max={max(boundaries_per_video)}, "
          f"mean={np.mean(boundaries_per_video):.1f}, median={np.median(boundaries_per_video):.1f}")
    
    if all_boundaries:
        print(f"\nAbsolute position statistics:")
        print(f"  Min: {min(all_boundaries)}")
        print(f"  Max: {max(all_boundaries)}")
        print(f"  Mean: {np.mean(all_boundaries):.1f}")
        print(f"  Median: {np.median(all_boundaries):.1f}")
        print(f"  Std: {np.std(all_boundaries):.1f}")
    
    # Normalized stats
    if video_lengths:
        all_normalized = []
        for vid_id, boundaries in predictions.items():
            if vid_id in video_lengths and video_lengths[vid_id] > 0:
                normalized = [b / video_lengths[vid_id] for b in boundaries]
                all_normalized.extend(normalized)
        
        if all_normalized:
            print(f"\nNormalized position statistics (fraction of video length):")
            print(f"  Min: {min(all_normalized):.3f}")
            print(f"  Max: {max(all_normalized):.3f}")
            print(f"  Mean: {np.mean(all_normalized):.3f}")
            print(f"  Median: {np.median(all_normalized):.3f}")
            print(f"  Std: {np.std(all_normalized):.3f}")
            
            q1, q2, q3 = np.percentile(all_normalized, [25, 50, 75])
            print(f"  Quartiles: Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f}")
            
            # Check for clustering
            if np.std(all_normalized) < 0.2:
                print(f"\n  ⚠ Low std deviation suggests boundaries may be clustered!")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize boundary prediction distributions')
    parser.add_argument('--pred-dir', type=str, default="boundry_segmentation/outputs/qwen30b_fps16_ws8_st2_vt1_md0.5_ds1", help='Directory with prediction .pkl files')
    parser.add_argument('--gt-path', type=str, default="boundry_segmentation/k400_mr345_val_min_change_duration0.3.pkl", help='Optional: GT pickle to get video lengths for normalization')
    parser.add_argument('--downsample', type=int, default=1, help='Downsampling factor used in predictions')
    parser.add_argument('--output', type=str, default="boundry_segmentation/dreck", help='Output path for plot (default: display only)')
    args = parser.parse_args()
    
    # Collect predictions
    predictions, video_lengths = collect_predictions(args.pred_dir, args.downsample, args.gt_path)
    
    if not predictions:
        print("[ERROR] No predictions found!")
        return
    
    # Print statistics
    print_statistics(predictions, video_lengths)
    
    # Create plots
    output_path = args.output if args.output else 'boundary_distribution.png'
    plot_distributions(predictions, video_lengths, output_path)


if __name__ == '__main__':
    main()