#!/usr/bin/env python3
"""
Multi-layer DNN Feature Comparison for Event Boundary Detection
Compares event segmentation across different neural network layers.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import your existing extractor (tolerate file name spelling)
try:
    from extract_boundries import VideoFeatureExtractor, EventBoundaryDetector
except ImportError:
    from extract_boundaries import VideoFeatureExtractor, EventBoundaryDetector


def run_all_layers(
    video_path: str,
    layers: List[int] = [0, 3, 6, 9, 11],
    n_states: int = 5,
    fps_sample: float = 2.0,
    n_pca: int = 30,
    output_dir: str = 'layer_comparison',
    # segmentation/time window
    start_time: float | None = None,
    end_time: float | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    # preprocessing
    center_crop: bool = False,
    crop_size: int | None = None,
    batch_size: int = 16,
    model: str = 'facebook/dinov2-base',
    # post-processing
    smooth: str = 'none',
    median_size: int = 3,
    gaussian_sigma: float = 2.0,
    min_segment_seconds: float = 0.0
):
    """
    Run boundary detection on multiple DNN layers and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}

    # Determine true video FPS
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # If frames are provided, convert to seconds for the extractor
    if start_frame is not None:
        start_time = start_frame / video_fps
    if end_frame is not None:
        end_time = end_frame / video_fps

    extractor = VideoFeatureExtractor(model_name=model)
    
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer}")
        print(f"{'='*60}")
        
        # Extract features
        features, frame_indices = extractor.extract_features(
            video_path, 
            layer=layer,
            fps_sample=fps_sample,
            batch_size=batch_size,
            start_time=start_time,
            end_time=end_time,
            center_crop=center_crop,
            crop_size=crop_size
        )
        
        # Detect boundaries
        detector = EventBoundaryDetector(n_components_pca=n_pca)
        layer_results = detector.fit_predict(
            features,
            n_states=n_states,
            smooth=smooth,
            median_size=median_size,
            gaussian_sigma=gaussian_sigma,
            min_segment_seconds=min_segment_seconds,
            fps_sample=fps_sample
        )
        
        # Store results
        results[f'layer_{layer}'] = {
            'layer': layer,
            'features': features,
            'states': layer_results['states'],
            'boundaries': layer_results['boundaries'],
            'n_boundaries': layer_results['n_boundaries'],
            'log_likelihood': layer_results['log_likelihood'],
            'frame_indices': frame_indices,
            'video_fps': video_fps,
            'fps_sample': fps_sample
        }
        
        # Save individual results
        layer_dir = f"{output_dir}/layer_{layer}"
        os.makedirs(layer_dir, exist_ok=True)
        
        # Compute accurate boundary times using true FPS and original frame indices
        b_indices = layer_results['boundaries']
        boundary_times = [
            (frame_indices[idx] / video_fps) for idx in b_indices if idx < len(frame_indices)
        ]

        with open(f"{layer_dir}/results.json", 'w') as f:
            json.dump({
                'layer': layer,
                'n_boundaries': layer_results['n_boundaries'],
                'boundaries': layer_results['boundaries'].tolist(),
                'boundary_times': boundary_times,
                'log_likelihood': float(layer_results['log_likelihood'])
            }, f, indent=2)
        
        print(f"Layer {layer}: {layer_results['n_boundaries']} boundaries found")
    
    return results


def plot_layer_comparison(results: Dict, output_dir: str = 'layer_comparison'):
    """
    Create comparison plots for all layers.
    """
    layers = []
    n_boundaries = []
    log_likelihoods = []
    
    for key, data in results.items():
        layers.append(data['layer'])
        n_boundaries.append(data['n_boundaries'])
        log_likelihoods.append(data['log_likelihood'])
    
    # Sort by layer number
    sorted_idx = np.argsort(layers)
    layers = np.array(layers)[sorted_idx]
    n_boundaries = np.array(n_boundaries)[sorted_idx]
    log_likelihoods = np.array(log_likelihoods)[sorted_idx]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Number of boundaries by layer
    axes[0, 0].bar(layers.astype(str), n_boundaries, color='steelblue')
    axes[0, 0].set_xlabel('DNN Layer')
    axes[0, 0].set_ylabel('Number of Boundaries')
    axes[0, 0].set_title('Event Boundaries Detected by Layer')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (l, n) in enumerate(zip(layers, n_boundaries)):
        axes[0, 0].text(i, n + 1, str(n), ha='center', va='bottom')
    
    # Plot 2: Log likelihood by layer
    axes[0, 1].plot(layers, log_likelihoods, 'o-', color='darkred', markersize=8)
    axes[0, 1].set_xlabel('DNN Layer')
    axes[0, 1].set_ylabel('Log Likelihood')
    axes[0, 1].set_title('Model Fit Quality by Layer')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(layers)
    
    # Plot 3: Boundary alignment heatmap
    max_frames = max(len(data['states']) for data in results.values())
    boundary_matrix = np.zeros((len(layers), max_frames))
    
    for i, layer in enumerate(layers):
        key = f'layer_{layer}'
        states = results[key]['states']
        boundaries = results[key]['boundaries']
        for b in boundaries:
            if b < max_frames:
                boundary_matrix[i, b] = 1
    
    # Downsample for visualization if too many frames
    if max_frames > 500:
        step = max_frames // 500
        boundary_matrix = boundary_matrix[:, ::step]
        x_label = f'Frame (downsampled by {step})'
    else:
        x_label = 'Frame'
    
    im = axes[1, 0].imshow(boundary_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[1, 0].set_yticks(range(len(layers)))
    axes[1, 0].set_yticklabels([f'Layer {l}' for l in layers])
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].set_title('Boundary Positions Across Layers')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Boundary overlap analysis
    overlap_matrix = np.zeros((len(layers), len(layers)))
    
    for i, layer_i in enumerate(layers):
        boundaries_i = set(results[f'layer_{layer_i}']['boundaries'])
        for j, layer_j in enumerate(layers):
            boundaries_j = set(results[f'layer_{layer_j}']['boundaries'])
            
            # Calculate Jaccard similarity with tolerance
            overlap = 0
            for bi in boundaries_i:
                for bj in boundaries_j:
                    if abs(bi - bj) <= 5:  # Within 5 frames
                        overlap += 1
                        break
            
            if len(boundaries_i) > 0:
                overlap_matrix[i, j] = overlap / len(boundaries_i)
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=[f'L{l}' for l in layers],
                yticklabels=[f'Layer {l}' for l in layers],
                ax=axes[1, 1], vmin=0, vmax=1)
    axes[1, 1].set_title('Boundary Overlap Between Layers')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plots to {output_dir}/layer_comparison.png")
    
    # Create additional detailed comparison
    plot_temporal_alignment(results, output_dir)


def plot_temporal_alignment(results: Dict, output_dir: str):
    """
    Create a temporal plot showing where boundaries occur across layers.
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (key, data) in enumerate(results.items()):
        layer = data['layer']
        frame_indices = data['frame_indices']
        video_fps = data['video_fps']
        boundaries = [frame_indices[idx] / video_fps for idx in data['boundaries'] if idx < len(frame_indices)]
        
        # Plot as vertical lines
        for b in boundaries:
            ax.axvline(b, color=colors[i], alpha=0.6, linewidth=1,
                      label=f'Layer {layer}' if b == boundaries[0] else '')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Layer')
    ax.set_title('Temporal Distribution of Event Boundaries Across DNN Layers')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_alignment.png', dpi=150, bbox_inches='tight')
    print(f"Saved temporal alignment to {output_dir}/temporal_alignment.png")


def create_summary_report(results: Dict, output_dir: str):
    """
    Create a markdown report summarizing findings.
    """
    report = f"""# Event Boundary Detection - Layer Comparison Report

## Summary Statistics

| Layer | Type | # Boundaries | Avg Segment Duration (s) | Log Likelihood |
|-------|------|--------------|-------------------------|----------------|
"""
    
    layer_types = {
        0: "Low-level (edges/colors)",
        3: "Low-mid (textures)",
        6: "Mid-level (parts/objects)",
        9: "High-level (scenes)",
        11: "Semantic (concepts)"
    }
    
    for key in sorted(results.keys()):
        data = results[key]
        layer = data['layer']
        n_boundaries = data['n_boundaries']
        total_samples = len(data['states'])
        # Estimate average duration in seconds using sampling rate
        avg_duration = (total_samples / max(1, (n_boundaries + 1))) / float(data['fps_sample'])
        log_like = data['log_likelihood']
        
        layer_type = layer_types.get(layer, "Unknown")
        report += f"| {layer} | {layer_type} | {n_boundaries} | {avg_duration:.1f} | {log_like:.1f} |\n"
    
    report += f"""
## Interpretation Guide

### Layer Characteristics:
- **Layers 0-3**: Detect visual/cinematographic changes (cuts, lighting)
- **Layers 6-8**: Detect object-level changes (people entering/leaving)
- **Layers 9-11**: Detect semantic changes (scene meaning, context)

### For Sherlock Analysis:
- Lower layers typically detect more boundaries (shot changes)
- Higher layers typically detect fewer boundaries (scene changes)
- Semantic layers (9-11) may better align with narrative structure

### Next Steps:
1. Validate detected boundaries against human annotations
2. Select optimal layer based on research objectives
3. Apply methodology to additional datasets

## Files Generated:
- `layer_comparison.png`: Statistical comparison
- `temporal_alignment.png`: Timeline visualization
- `layer_N/results.json`: Individual layer results
"""
    
    with open(f'{output_dir}/report.md', 'w') as f:
        f.write(report)
    
    print(f"Saved summary report to {output_dir}/report.md")


def main():
    parser = argparse.ArgumentParser(
        description='Compare event boundaries across DNN layers'
    )
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--layers', nargs='+', type=int, 
                        default=[0, 3, 6, 9, 11],
                        help='Layers to compare (default: 0 3 6 9 11)')
    parser.add_argument('--n_states', type=int, default=5,
                        help='Number of HMM states')
    parser.add_argument('--fps_sample', type=float, default=2.0,
                        help='Sampling rate in fps')
    parser.add_argument('--n_pca', type=int, default=30,
                        help='Number of PCA components')
    parser.add_argument('--output_dir', type=str, default='layer_comparison',
                        help='Output directory')
    # time/frame window
    parser.add_argument('--start_time', type=float, default=None, help='Start time (s), overrides start_frame')
    parser.add_argument('--end_time', type=float, default=None, help='End time (s), overrides end_frame')
    parser.add_argument('--start_frame', type=int, default=None, help='Start frame (converted using video FPS)')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame (converted using video FPS)')
    # preprocessing
    parser.add_argument('--center_crop', action='store_true', help='Enable center cropping (square)')
    parser.add_argument('--crop_size', type=int, default=None, help='Square crop size in pixels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for feature extraction')
    parser.add_argument('--model', type=str, default='facebook/dinov2-base', help='HuggingFace model id')
    # post-processing
    parser.add_argument('--smooth', type=str, choices=['none','median','gaussian'], default='none',
                        help='Post-process states: none, median, gaussian')
    parser.add_argument('--median_size', type=int, default=3, help='Median filter size (odd)')
    parser.add_argument('--gaussian_sigma', type=float, default=2.0, help='Gaussian sigma for probs smoothing')
    parser.add_argument('--min_segment_seconds', type=float, default=0.0,
                        help='Enforce minimum segment duration in seconds (0 disables)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DNN Layer Comparison for Event Segmentation")
    print("="*60)
    
    # Run on all layers
    results = run_all_layers(
        args.video_path,
        layers=args.layers,
        n_states=args.n_states,
        fps_sample=args.fps_sample,
        n_pca=args.n_pca,
        output_dir=args.output_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        center_crop=args.center_crop,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        model=args.model,
        smooth=args.smooth,
        median_size=args.median_size,
        gaussian_sigma=args.gaussian_sigma,
        min_segment_seconds=args.min_segment_seconds
    )
    
    # Create comparison plots
    plot_layer_comparison(results, args.output_dir)
    
    # Create summary report
    create_summary_report(results, args.output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete. Output files:")
    print("- layer_comparison.png: Statistical comparison across layers")
    print("- temporal_alignment.png: Temporal distribution of boundaries")
    print("- report.md: Summary report")
    print("- layer_N/results.json: Individual layer results")
    print("="*60)


if __name__ == '__main__':
    main()