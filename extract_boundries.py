#!/usr/bin/env python3
"""
Video Event Boundary Detection using transformer vision features (DINO/ViT or CLIP) and HMM
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import median_filter, gaussian_filter1d


class VideoFeatureExtractor:
    """Extract vision features from video frames using ViT/DINO or CLIP backbones."""
    
    def __init__(self, model_name: str = 'facebook/dinov2-base', device: str = None):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Hugging Face model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        name_lower = model_name.lower()
        if 'clip' in name_lower:
            self.model_family = 'clip'
        else:
            # Default to ViT/DINO-style hidden states
            self.model_family = 'vit'

        print(f"Loaded model '{model_name}' as {self.model_family.upper()} on {self.device}")
        
    def extract_features(
        self, 
        video_path: str, 
        layer: int = -1,
        fps_sample: float = 2.0,
        batch_size: int = 32,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        center_crop: bool = False,
        crop_size: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract features from video frames.
        
        Args:
            video_path: Path to video file
            layer: Which DINO layer to use (-1 for last)
            fps_sample: Sample rate (frames per second)
            batch_size: Batch size for processing
            
        Returns:
            features: Array of shape (n_frames, feature_dim)
            frame_indices: List of original frame indices
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps_sample <= 0:
            raise ValueError("fps_sample must be > 0")

        # Calculate frame sampling interval (ensure at least 1)
        frame_interval = max(1, int(video_fps / fps_sample))

        # Compute start/end frames
        start_frame = 0 if start_time is None else max(0, int(start_time * video_fps))
        end_frame = (total_frames - 1) if end_time is None else min(total_frames - 1, int(end_time * video_fps))
        if end_frame < start_frame:
            raise ValueError("end_time must be greater than start_time")

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        segment_frames = end_frame - start_frame + 1

        print(f"Video: {video_fps:.1f} fps, {total_frames} frames total")
        if start_time is not None or end_time is not None:
            start_sec = 0.0 if start_time is None else start_time
            end_sec = (total_frames - 1) / video_fps if end_time is None else end_time
            print(f"Processing segment: {start_sec:.2f}s to {end_sec:.2f}s ({segment_frames} frames)")
        print(f"Sampling at {fps_sample} fps (every {frame_interval} frames)")

        features = []
        frame_indices = []
        frame_batch = []
        batch_indices = []
        local_idx = 0
        global_frame_idx = start_frame

        expected_samples = max(1, segment_frames // frame_interval)
        pbar = tqdm(total=expected_samples, desc="Extracting features")

        while True:
            if global_frame_idx > end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if center_crop:
                h, w = frame.shape[:2]
                # Determine square crop size
                crop_edge = min(h, w) if crop_size is None else min(crop_size, h, w)
                y0 = (h - crop_edge) // 2
                x0 = (w - crop_edge) // 2
                frame = frame[y0:y0+crop_edge, x0:x0+crop_edge]

            if local_idx % frame_interval == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_batch.append(image)
                batch_indices.append(global_frame_idx)

                if len(frame_batch) >= batch_size:
                    batch_features = self._process_batch(frame_batch, layer)
                    features.extend(batch_features)
                    frame_indices.extend(batch_indices)
                    frame_batch = []
                    batch_indices = []
                    pbar.update(len(batch_features))

            local_idx += 1
            global_frame_idx += 1

        # Process remaining frames
        if frame_batch:
            batch_features = self._process_batch(frame_batch, layer)
            features.extend(batch_features)
            frame_indices.extend(batch_indices)
            pbar.update(len(batch_features))

        cap.release()
        pbar.close()
        
        features = np.array(features)
        print(f"Extracted features shape: {features.shape}")
        
        return features, frame_indices
    
    def _process_batch(self, images: List[Image.Image], layer: int) -> List[np.ndarray]:
        """Process a batch of images and return per-image feature vectors."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features_tensor = None

            if self.model_family == 'clip':
                # Use the vision tower directly to avoid invoking the text branch
                if hasattr(self.model, 'vision_model') and self.model.vision_model is not None:
                    vision_out = self.model.vision_model(
                        pixel_values=inputs['pixel_values'],
                        output_hidden_states=True
                    )
                    if getattr(vision_out, 'hidden_states', None) is not None:
                        hidden_states = vision_out.hidden_states[layer]
                        features_tensor = hidden_states.mean(dim=1)
                    elif getattr(vision_out, 'last_hidden_state', None) is not None:
                        features_tensor = vision_out.last_hidden_state.mean(dim=1)
                    elif isinstance(vision_out, (tuple, list)) and len(vision_out) > 0:
                        # Fallback if model returns a tuple (e.g., last_hidden_state first)
                        features_tensor = vision_out[0].mean(dim=1)
                # Fallback to get_image_features if available
                if features_tensor is None and hasattr(self.model, 'get_image_features'):
                    features_tensor = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            else:
                # ViT/DINO-like models
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[layer]
                    features_tensor = hidden_states.mean(dim=1)
                elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    features_tensor = outputs.last_hidden_state.mean(dim=1)

            if features_tensor is None:
                raise RuntimeError(
                    "Could not obtain vision features from model outputs. "
                    "Ensure the selected model exposes vision hidden states or pooled image features."
                )

            features = features_tensor.detach().cpu().numpy()

        return [f for f in features]


class EventBoundaryDetector:
    """Detect event boundaries using HMM on features."""
    
    def __init__(self, n_components_pca: int = 50):
        """
        Initialize the boundary detector.
        
        Args:
            n_components_pca: Number of PCA components to retain
        """
        self.n_components_pca = n_components_pca
        self.pca = PCA(n_components=n_components_pca)
        self.scaler = StandardScaler()
        
    def fit_predict(
        self, 
        features: np.ndarray, 
        n_states: int = 10,
        covariance_type: str = 'full',
        n_iter: int = 100,
        smooth: str = 'none',
        median_size: int = 3,
        gaussian_sigma: float = 2.0,
        min_segment_seconds: float = 0.0,
        fps_sample: float = 2.0
    ) -> Dict:
        """
        Fit HMM and predict event boundaries.
        
        Args:
            features: Feature array (n_frames, feature_dim)
            n_states: Number of hidden states
            covariance_type: Type of covariance matrix
            n_iter: Number of iterations for HMM training
            
        Returns:
            Dictionary with results
        """
        print("\nFitting event boundary detector...")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Reduce dimensionality
        print(f"Reducing dimensions from {features.shape[1]} to {self.n_components_pca}")
        features_reduced = self.pca.fit_transform(features_scaled)
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.2%}")
        
        # Fit HMM
        print(f"Fitting HMM with {n_states} states...")
        model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type=covariance_type,
            n_iter=n_iter,
            verbose=True
        )
        model.fit(features_reduced)
        
        # Get state sequence
        states = model.predict(features_reduced)
        log_likelihood = model.score(features_reduced)

        # Optionally smooth states
        states_smoothed = states.copy()
        if smooth == 'median':
            k = max(1, int(median_size))
            if k % 2 == 0:
                k += 1
            states_smoothed = median_filter(states_smoothed, size=k, mode='nearest')
        elif smooth == 'gaussian':
            try:
                state_probs = model.predict_proba(features_reduced)
                state_probs_smooth = gaussian_filter1d(state_probs, sigma=float(gaussian_sigma), axis=0, mode='nearest')
                states_smoothed = np.argmax(state_probs_smooth, axis=1)
            except AttributeError:
                print("predict_proba not available on HMM model; skipping gaussian smoothing")

        # Enforce minimum segment duration if specified
        def enforce_min_duration(local_states: np.ndarray, min_frames: int) -> np.ndarray:
            """Merge segments shorter than min_frames with previous segment."""
            if min_frames <= 1:
                return local_states
            result = local_states.copy()
            change_points = np.where(np.diff(result) != 0)[0] + 1
            boundaries_local = np.concatenate(([0], change_points, [len(result)])).astype(int)
            for i in range(1, len(boundaries_local)):
                start = boundaries_local[i - 1]
                end = boundaries_local[i]
                if end - start < min_frames and start > 0:
                    result[start:end] = result[start - 1]
            return result

        states_final = states_smoothed
        if min_segment_seconds and min_segment_seconds > 0:
            min_frames = max(1, int(round(min_segment_seconds * float(fps_sample))))
            states_final = enforce_min_duration(states_smoothed, min_frames)

        # Find boundaries (where state changes) on final states
        boundaries = np.where(np.diff(states_final) != 0)[0] + 1  # +1 for correct index
        
        # Calculate state durations using final states
        state_durations = []
        current_state = states_final[0]
        duration = 1
        for state in states_final[1:]:
            if state == current_state:
                duration += 1
            else:
                state_durations.append(duration)
                duration = 1
                current_state = state
        state_durations.append(duration)
        
        return {
            'states': states_final,
            'boundaries': boundaries,
            'n_boundaries': len(boundaries),
            'log_likelihood': log_likelihood,
            'state_durations': state_durations,
            'features_reduced': features_reduced,
            'model': model
        }


def visualize_results(
    results: Dict, 
    frame_indices: List[int],
    fps: float,
    output_dir: str = 'outputs'
):
    """Create visualizations of the results."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Convert frame indices to time
    times = np.array(frame_indices) / fps
    
    # Plot 1: State sequence
    axes[0].plot(times, results['states'], linewidth=0.8)
    axes[0].set_ylabel('State')
    axes[0].set_title('HMM State Sequence')
    axes[0].grid(True, alpha=0.3)
    
    # Add boundary lines
    for boundary_idx in results['boundaries']:
        if boundary_idx < len(times):
            axes[0].axvline(times[boundary_idx], color='red', alpha=0.5, linestyle='--')
    
    # Plot 2: State duration histogram
    axes[1].hist(results['state_durations'], bins=30, edgecolor='black')
    axes[1].set_xlabel('Duration (frames)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('State Duration Distribution')
    
    # Plot 3: PCA components (first 2)
    features = results['features_reduced']
    scatter = axes[2].scatter(
        features[:, 0], 
        features[:, 1], 
        c=results['states'], 
        cmap='tab10',
        s=5,
        alpha=0.6
    )
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].set_title('First Two Principal Components')
    plt.colorbar(scatter, ax=axes[2], label='State')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boundary_analysis.png', dpi=150)
    print(f"Saved visualization to {output_dir}/boundary_analysis.png")
    
    # Save state transition matrix
    plt.figure(figsize=(8, 6))
    trans_mat = results['model'].transmat_
    sns.heatmap(trans_mat, annot=True, fmt='.2f', cmap='Blues')
    plt.title('State Transition Matrix')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.savefig(f'{output_dir}/transition_matrix.png', dpi=150)
    print(f"Saved transition matrix to {output_dir}/transition_matrix.png")


def plot_correlation_matrix(
    features_reduced: np.ndarray,
    states: np.ndarray,
    boundaries: np.ndarray,
    output_dir: str = 'outputs',
    max_frames: int = 500
):
    """
    Create correlation matrix visualization - the standard event segmentation plot.
    Shows block diagonal structure if segmentation is good.
    """

    # Downsample if too many frames (for memory/visibility)
    if len(features_reduced) > max_frames:
        step = max(1, len(features_reduced) // max_frames)
        features_plot = features_reduced[::step]
        states_plot = states[::step]
        boundaries_plot = boundaries // step  # Scale boundaries too
        print(f"Downsampling correlation matrix from {len(features_reduced)} to {len(features_plot)} frames")
    else:
        features_plot = features_reduced
        states_plot = states
        boundaries_plot = boundaries

    # Compute correlation matrix
    print("Computing correlation matrix...")
    # Normalize features
    features_std = features_plot.std(axis=0)
    # Avoid division by zero for any constant components
    features_std[features_std == 0] = 1.0
    features_norm = (features_plot - features_plot.mean(axis=0)) / features_std
    # Compute correlation (frame x frame)
    corr_matrix = np.corrcoef(features_norm)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Raw correlation matrix
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title('Feature Correlation Matrix')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Frame')

    # Add boundary lines
        
    for boundary in boundaries_plot:
        if boundary < len(features_plot):
            ax1.axhline(boundary, color='green', alpha=0.3, linewidth=0.5)
            ax1.axvline(boundary, color='green', alpha=0.3, linewidth=0.5)

    plt.colorbar(im1, ax=ax1, label='Correlation')

    # Plot 2: State-sorted correlation matrix (reorder by state)
    # This shows how well states cluster
    state_order = np.argsort(states_plot)
    corr_sorted = corr_matrix[state_order][:, state_order]
    states_sorted = states_plot[state_order]

    im2 = ax2.imshow(corr_sorted, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_title('State-Sorted Correlation Matrix')
    ax2.set_xlabel('Frame (sorted by state)')
    ax2.set_ylabel('Frame (sorted by state)')

    # Add state boundaries in sorted view
    state_boundaries = np.where(np.diff(states_sorted) != 0)[0]
    for boundary in state_boundaries:
        ax2.axhline(boundary, color='black', alpha=0.5, linewidth=0.5)
        ax2.axvline(boundary, color='black', alpha=0.5, linewidth=0.5)

    plt.colorbar(im2, ax=ax2, label='Correlation')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Saved correlation matrix to {output_dir}/correlation_matrix.png")

    # Also create a cleaner block-averaged version
    plot_block_correlation(features_reduced, states, boundaries, output_dir)


def plot_block_correlation(
    features_reduced: np.ndarray,
    states: np.ndarray, 
    boundaries: np.ndarray,
    output_dir: str = 'outputs'
):
    """
    Create block-averaged correlation matrix (average within each segment).
    This is cleaner and shows the ideal block diagonal structure.
    """
    # Get segments
    boundaries_with_ends = np.concatenate([[0], boundaries, [len(states)]])
    n_segments = len(boundaries_with_ends) - 1

    # Don't plot if too many segments
    if n_segments > 100:
        print(f"Too many segments ({n_segments}) for block correlation plot, skipping...")
        return

    # Compute mean features for each segment
    segment_features = []
    for i in range(n_segments):
        start = boundaries_with_ends[i]
        end = boundaries_with_ends[i + 1]
        segment_mean = features_reduced[start:end].mean(axis=0)
        segment_features.append(segment_mean)

    segment_features = np.array(segment_features)

    # Normalize and compute correlation
    seg_std = segment_features.std(axis=0)
    seg_std[seg_std == 0] = 1.0
    segment_features_norm = (segment_features - segment_features.mean(axis=0)) / seg_std
    segment_corr = np.corrcoef(segment_features_norm)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(segment_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f'Segment-Level Correlation Matrix ({n_segments} segments)')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Segment')

    # Add grid
    for i in range(n_segments):
        ax.axhline(i - 0.5, color='gray', alpha=0.2, linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', alpha=0.2, linewidth=0.5)

    plt.colorbar(im, ax=ax, label='Correlation')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/segment_correlation.png', dpi=150, bbox_inches='tight')
    print(f"Saved segment correlation to {output_dir}/segment_correlation.png")

def save_results(
    results: Dict, 
    frame_indices: List[int],
    video_path: str,
    fps: float,
    output_dir: str = 'outputs'
):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    # Compute boundary times using the original frame indices for accuracy
    boundary_times = [
        (frame_indices[idx] / fps) for idx in results['boundaries'] if idx < len(frame_indices)
    ]

    output = {
        'video_path': video_path,
        'fps': fps,
        'n_frames_analyzed': len(frame_indices),
        'n_states': int(results['states'].max()) + 1,
        'n_boundaries': results['n_boundaries'],
        'log_likelihood': float(results['log_likelihood']),
        'frame_indices': frame_indices,
        'states': results['states'].tolist(),
        'boundaries': results['boundaries'].tolist(),
        'boundary_times': boundary_times,
        'mean_state_duration': float(np.mean(results['state_durations'])),
        'std_state_duration': float(np.std(results['state_durations']))
    }
    
    output_path = f'{output_dir}/boundaries.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved results to {output_path}")
    
    # Also save boundaries as simple text file
    boundary_file = f'{output_dir}/boundary_times.txt'
    with open(boundary_file, 'w') as f:
        f.write("# Event boundaries (seconds)\n")
        for t in output['boundary_times']:
            f.write(f"{t:.2f}\n")
    print(f"Saved boundary times to {boundary_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract event boundaries from video using DINO v2 + HMM')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--n_states', type=int, default=10, help='Number of HMM states')
    parser.add_argument('--layer', type=int, default=9, help='DINO layer to use (0-11, or -1 for last)')
    parser.add_argument('--fps_sample', type=float, default=2.0, help='Sample rate in fps')
    parser.add_argument('--n_pca', type=int, default=50, help='Number of PCA components')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for feature extraction')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--model', type=str, default='facebook/dinov2-base', 
                        help='Vision backbone from HF hub (e.g., facebook/dinov2-base, openai/clip-vit-base-patch16)')
    parser.add_argument('--center_crop', action='store_true', help='Enable center cropping (square)')
    parser.add_argument('--crop_size', type=int, default=None, help='Square crop size in pixels (optional)')
    parser.add_argument('--start_time', type=float, default=None, help='Start time in seconds (inclusive)')
    parser.add_argument('--end_time', type=float, default=None, help='End time in seconds (inclusive)')
    parser.add_argument('--smooth', type=str, choices=['none','median','gaussian'], default='none',
                        help='Post-process states: none, median, gaussian')
    parser.add_argument('--median_size', type=int, default=3, help='Median filter size (odd)')
    parser.add_argument('--gaussian_sigma', type=float, default=2.0, help='Gaussian sigma for probs smoothing')
    parser.add_argument('--min_segment_seconds', type=float, default=0.0,
                        help='Enforce minimum segment duration in seconds (0 disables)')
    # Embedding export options
    parser.add_argument('--export_dir', type=str, default=None,
                        help='If set, save raw embeddings and frame indices here')
    parser.add_argument('--export_pca', action='store_true',
                        help='Also save PCA-reduced embeddings (uses --n_pca)')
    parser.add_argument('--extract_only', action='store_true',
                        help='Only extract and export embeddings, skip HMM and plotting')
    
    args = parser.parse_args()
    
    # Check video exists
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    
    # Get video info
    cap = cv2.VideoCapture(args.video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Extract features
    extractor = VideoFeatureExtractor(model_name=args.model)
    features, frame_indices = extractor.extract_features(
        args.video_path, 
        layer=args.layer,
        fps_sample=args.fps_sample,
        batch_size=args.batch_size,
        start_time=args.start_time,
        end_time=args.end_time,
        center_crop=args.center_crop,
        crop_size=args.crop_size
    )

    # Optional: export embeddings (raw and/or PCA) and optionally exit
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        raw_path = os.path.join(args.export_dir, 'features_raw.npy')
        idx_path = os.path.join(args.export_dir, 'frame_indices.npy')
        meta_path = os.path.join(args.export_dir, 'embeddings_meta.json')
        np.save(raw_path, features)
        np.save(idx_path, np.array(frame_indices, dtype=np.int32))
        meta = {
            'video_path': args.video_path,
            'video_fps': video_fps,
            'fps_sample': args.fps_sample,
            'layer': args.layer,
            'start_time': args.start_time,
            'end_time': args.end_time,
            'center_crop': args.center_crop,
            'crop_size': args.crop_size,
            'n_pca': args.n_pca if args.export_pca else None
        }
        if args.export_pca:
            detector_tmp = EventBoundaryDetector(n_components_pca=args.n_pca)
            features_scaled = detector_tmp.scaler.fit_transform(features)
            features_reduced = detector_tmp.pca.fit_transform(features_scaled)
            pca_path = os.path.join(args.export_dir, f'features_pca_{args.n_pca}.npy')
            np.save(pca_path, features_reduced)
            meta.update({
                'pca_explained_variance_ratio': float(detector_tmp.pca.explained_variance_ratio_.sum())
            })
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved embeddings to {args.export_dir}")
        if args.extract_only:
            print('Extract-only mode: skipping HMM and visualization.')
            return
    
    # Detect boundaries
    detector = EventBoundaryDetector(n_components_pca=args.n_pca)
    results = detector.fit_predict(
        features,
        n_states=args.n_states,
        smooth=args.smooth,
        median_size=args.median_size,
        gaussian_sigma=args.gaussian_sigma,
        min_segment_seconds=args.min_segment_seconds,
        fps_sample=args.fps_sample
    )
    
    # Save and visualize (use true video FPS)
    save_results(results, frame_indices, args.video_path, video_fps, args.output_dir)
    visualize_results(results, frame_indices, video_fps, args.output_dir)

    # Add correlation matrix visualization
    plot_correlation_matrix(
        results['features_reduced'],
        results['states'],
        results['boundaries'],
        args.output_dir
    )
    
    print(f"\n✅ Complete! Found {results['n_boundaries']} event boundaries")
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()