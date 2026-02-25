#!/usr/bin/env python3
"""
UBoCo: Unsupervised Boundary Contrastive Learning for Event Boundary Detection
Improved version with better progress tracking and caching
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import timedelta

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class VideoFeatureExtractor(nn.Module):
    """Feature extractor with frozen ResNet-50 + trainable encoder."""
    
    def __init__(self, encoder_dim: int = 512, device: str = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frozen ResNet-50 (pretrained on ImageNet)
        print("Loading ResNet-50 pretrained on ImageNet...")
        resnet = models.resnet50(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
        for param in self.resnet_backbone.parameters():
            param.requires_grad = False
        self.resnet_backbone.eval()
        
        # Trainable custom encoder (1D CNN + MLP)
        self.encoder = nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, encoder_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_dim)
        )
        
        self.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Feature extractor initialized on {self.device}")
        print(f"  ResNet-50 backbone: 2048D (frozen)")
        print(f"  Custom encoder: {encoder_dim}D (trainable)")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
    
    def load_and_preprocess_frames(
        self, 
        video_path: str, 
        fps_sample: float = 2.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Tuple[List[Image.Image], List[int], float]:
        """Load video frames (caching for efficiency)."""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        frame_interval = max(1, int(video_fps / fps_sample))
        start_frame = 0 if start_time is None else max(0, int(start_time * video_fps))
        end_frame = total_frames - 1 if end_time is None else min(total_frames - 1, int(end_time * video_fps))
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"\n{'='*60}")
        print(f"VIDEO INFO")
        print(f"{'='*60}")
        print(f"Path: {video_path}")
        print(f"Duration: {duration:.1f}s ({total_frames} frames @ {video_fps:.1f} fps)")
        if start_time is not None or end_time is not None:
            seg_start = start_time or 0.0
            seg_end = end_time or duration
            print(f"Segment: {seg_start:.1f}s - {seg_end:.1f}s")
        print(f"Sampling: {fps_sample} fps (every {frame_interval} frames)")
        
        frames = []
        frame_indices = []
        local_idx = 0
        global_idx = start_frame
        
        expected_frames = (end_frame - start_frame + 1) // frame_interval
        pbar = tqdm(total=expected_frames, desc="📹 Loading frames", unit="frame")
        
        while global_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if local_idx % frame_interval == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(image)
                frame_indices.append(global_idx)
                pbar.update(1)
            
            local_idx += 1
            global_idx += 1
        
        cap.release()
        pbar.close()
        
        print(f"✓ Loaded {len(frames)} frames")
        
        return frames, frame_indices, video_fps
    
    def extract_resnet_features(
        self, 
        frames: List[Image.Image], 
        batch_size: int = 32
    ) -> torch.Tensor:
        """Extract ResNet features (frozen backbone)."""
        resnet_features = []
        
        with torch.no_grad():
            self.resnet_backbone.eval()
            pbar = tqdm(
                range(0, len(frames), batch_size), 
                desc="🔬 Extracting ResNet features",
                unit="batch"
            )
            
            for i in pbar:
                batch_frames = frames[i:i+batch_size]
                batch_tensor = torch.stack([
                    self.transform(f) for f in batch_frames
                ]).to(self.device)
                
                features = self.resnet_backbone(batch_tensor).squeeze(-1).squeeze(-1)
                resnet_features.append(features.cpu())
        
        resnet_features = torch.cat(resnet_features, dim=0)  # (L, 2048)
        print(f"✓ ResNet features: {resnet_features.shape}")
        
        return resnet_features
    
    def encode_features(self, resnet_features: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """Apply trainable encoder to ResNet features."""
        resnet_features = resnet_features.unsqueeze(0).to(self.device)  # (1, L, 2048)
        resnet_features = resnet_features.permute(0, 2, 1)  # (1, 2048, L)
        
        if requires_grad:
            self.encoder.train()
            encoded = self.encoder(resnet_features)
        else:
            self.encoder.eval()
            with torch.no_grad():
                encoded = self.encoder(resnet_features)
        
        encoded = encoded.squeeze(0).permute(1, 0)  # (L, encoder_dim)
        return encoded


def compute_tsm(features: torch.Tensor) -> torch.Tensor:
    """Compute Temporal Self-Similarity Matrix using cosine similarity."""
    features_norm = F.normalize(features, p=2, dim=1)
    tsm = torch.matmul(features_norm, features_norm.T)
    return tsm


class RecursiveTSMParser:
    """Recursive TSM Parsing (RTP) algorithm for boundary detection."""
    
    def __init__(
        self, 
        kernel_size: int = 5, 
        min_length: int = 10, 
        threshold_diff: float = 0.1,
        max_depth: Optional[int] = None,
        max_boundaries: Optional[int] = None
    ):
        self.kernel_size = kernel_size
        self.min_length = min_length
        self.threshold_diff = threshold_diff
        self.max_depth = max_depth
        self.max_boundaries = max_boundaries
        
        # Create contrastive kernel
        k = kernel_size
        mid = k // 2
        kernel = np.zeros((k, k))
        kernel[:mid, :mid] = 1
        kernel[:mid, mid+1:] = -1
        kernel[mid+1:, :mid] = -1
        kernel[mid+1:, mid+1:] = 1
        self.kernel = torch.tensor(kernel, dtype=torch.float32)
    
    def diagonal_conv(self, tsm: torch.Tensor) -> torch.Tensor:
        """Convolve kernel along TSM diagonal."""
        k = self.kernel_size
        pad = k // 2
        
        tsm_padded = F.pad(tsm, (pad, pad, pad, pad), value=0)
        L = tsm.shape[0]
        scores = torch.zeros(L)
        kernel = self.kernel.to(tsm.device)
        
        for i in range(L):
            patch = tsm_padded[i:i+k, i:i+k]
            score = (patch * kernel).sum()
            scores[i] = score
        
        return scores
    
    def parse_recursive(
        self, 
        tsm: torch.Tensor, 
        offset: int = 0, 
        depth: int = 0,
        top_k: float = 0.2
    ) -> List[int]:
        """Recursively parse TSM to find boundaries with optional limits."""
        collected: List[int] = []
        
        def recurse(curr_tsm: torch.Tensor, curr_offset: int, curr_depth: int):
            L = curr_tsm.shape[0]
            
            if L < self.min_length:
                return
            if self.max_depth is not None and curr_depth >= self.max_depth:
                return
            if self.max_boundaries is not None and len(collected) >= self.max_boundaries:
                return
            
            scores = self.diagonal_conv(curr_tsm)
            max_score = scores.max().item()
            mean_score = scores.mean().item()
            
            if max_score - mean_score < self.threshold_diff:
                return
            
            k = max(1, int(L * top_k))
            top_values, top_indices = torch.topk(scores, k)
            probs = F.softmax(top_values, dim=0)
            sampled_idx = torch.multinomial(probs, 1).item()
            boundary_idx_local = top_indices[sampled_idx].item()
            
            global_boundary = curr_offset + boundary_idx_local
            collected.append(global_boundary)
            
            # Early stop if reached limit
            if self.max_boundaries is not None and len(collected) >= self.max_boundaries:
                return
            
            # Recurse left
            if boundary_idx_local > 0:
                left_tsm = curr_tsm[:boundary_idx_local, :boundary_idx_local]
                recurse(left_tsm, curr_offset, curr_depth + 1)
                if self.max_boundaries is not None and len(collected) >= self.max_boundaries:
                    return
            
            # Recurse right
            if boundary_idx_local < L - 1:
                right_tsm = curr_tsm[boundary_idx_local+1:, boundary_idx_local+1:]
                recurse(right_tsm, curr_offset + boundary_idx_local + 1, curr_depth + 1)
            
        recurse(tsm, offset, depth)
        return collected


def find_boundaries_peaks(
    tsm: torch.Tensor,
    kernel_size: int = 5,
    distance: int = 20,
    prominence: float = 0.5,
    max_boundaries: Optional[int] = None
) -> List[int]:
    """Peak-based boundary detection without external deps.
    - distance: minimum frame distance between boundaries
    - prominence: required score above local moving average
    """
    rtp = RecursiveTSMParser(kernel_size=kernel_size)
    scores_t = rtp.diagonal_conv(tsm)
    scores = scores_t.detach().cpu().numpy()
    L = scores.shape[0]
    
    if L < 3:
        return []
    
    # Local maxima candidates
    candidates = np.where((scores[1:-1] > scores[:-2]) & (scores[1:-1] >= scores[2:]))[0] + 1
    if candidates.size == 0:
        return []
    
    # Local baseline via moving average
    win = max(3, int(distance))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    baseline = np.convolve(scores, kernel, mode='same')
    proms = scores[candidates] - baseline[candidates]
    valid = candidates[proms >= prominence]
    if valid.size == 0:
        return []
    
    # Greedy selection by score with min distance
    order = np.argsort(scores[valid])[::-1]
    selected: List[int] = []
    for idx in valid[order]:
        if all(abs(idx - s) >= distance for s in selected):
            selected.append(int(idx))
            if max_boundaries is not None and len(selected) >= max_boundaries:
                break
    return sorted(selected)


def compute_boco_loss(
    tsm: torch.Tensor, 
    boundaries: List[int], 
    gap: int = 20
) -> torch.Tensor:
    """Compute Boundary Contrastive (BoCo) loss."""
    L = tsm.shape[0]
    device = tsm.device
    
    mask = torch.zeros((L, L), dtype=torch.float32, device=device)
    boundaries_full = sorted([0] + boundaries + [L])
    
    # Local similarity prior
    local_mask = torch.zeros((L, L), dtype=torch.float32, device=device)
    for i in range(L):
        start = max(0, i - gap)
        end = min(L, i + gap + 1)
        local_mask[i, start:end] = 1
    
    # Semantic coherency prior
    semantic_mask = torch.zeros((L, L), dtype=torch.float32, device=device)
    for i in range(len(boundaries_full) - 1):
        seg_start = boundaries_full[i]
        seg_end = boundaries_full[i + 1]
        semantic_mask[seg_start:seg_end, seg_start:seg_end] = 1
        
        for j in range(i + 1, len(boundaries_full) - 1):
            other_start = boundaries_full[j]
            other_end = boundaries_full[j + 1]
            semantic_mask[seg_start:seg_end, other_start:other_end] = -1
            semantic_mask[other_start:other_end, seg_start:seg_end] = -1
    
    mask = local_mask * semantic_mask
    
    positive_mask = (mask == 1)
    negative_mask = (mask == -1)
    
    if positive_mask.sum() == 0 or negative_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    positive_sim = tsm[positive_mask].mean()
    negative_sim = tsm[negative_mask].mean()
    
    loss = negative_sim - positive_sim
    return loss


def train_uboco(
    feature_extractor: VideoFeatureExtractor,
    video_path: str,
    fps_sample: float = 2.0,
    n_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    rtp_top_k: float = 0.05,
    boco_gap: int = 20,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    output_dir: str = 'outputs',
    # Boundary method controls
    boundary_method: str = 'peaks',
    rtp_kernel_size: int = 5,
    rtp_min_length: int = 50,
    rtp_threshold_diff: float = 0.3,
    rtp_max_depth: Optional[int] = 3,
    rtp_max_boundaries: Optional[int] = 30,
    peaks_distance: int = 30,
    peaks_prominence: float = 0.6,
    peaks_max_boundaries: Optional[int] = 25
):
    """Train UBoCo model with proper progress tracking."""
    
    os.makedirs(output_dir, exist_ok=True)
    device = feature_extractor.device
    
    # Load frames ONCE (cache for efficiency)
    print(f"\n{'='*60}")
    print(f"STEP 1: LOADING VIDEO")
    print(f"{'='*60}")
    start_load = time.time()
    
    frames, frame_indices, video_fps = feature_extractor.load_and_preprocess_frames(
        video_path, fps_sample, start_time, end_time
    )
    
    load_time = time.time() - start_load
    print(f"✓ Loading complete in {load_time:.1f}s")
    
    # Extract ResNet features ONCE (frozen backbone)
    print(f"\n{'='*60}")
    print(f"STEP 2: EXTRACTING RESNET FEATURES (one-time)")
    print(f"{'='*60}")
    start_resnet = time.time()
    
    resnet_features = feature_extractor.extract_resnet_features(frames, batch_size)
    
    resnet_time = time.time() - start_resnet
    print(f"✓ ResNet extraction complete in {resnet_time:.1f}s")
    
    # Initialize components
    rtp = RecursiveTSMParser(
        kernel_size=rtp_kernel_size,
        min_length=rtp_min_length,
        threshold_diff=rtp_threshold_diff,
        max_depth=rtp_max_depth,
        max_boundaries=rtp_max_boundaries
    )
    optimizer = torch.optim.Adam(feature_extractor.encoder.parameters(), lr=lr)
    
    # Training loop with progress tracking
    print(f"\n{'='*60}")
    print(f"STEP 3: SELF-SUPERVISED TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {lr}")
    print(f"BoCo gap: {boco_gap} frames")
    print(f"RTP top-k: {rtp_top_k*100:.0f}%")
    print(f"Boundary method: {boundary_method}")
    if boundary_method == 'rtp':
        print(f"RTP settings -> kernel={rtp_kernel_size}, min_length={rtp_min_length}, threshold_diff={rtp_threshold_diff}, max_depth={rtp_max_depth}, max_boundaries={rtp_max_boundaries}")
    else:
        print(f"Peaks settings -> distance={peaks_distance}, prominence={peaks_prominence}, max_boundaries={peaks_max_boundaries}")
    
    # Prepare directory for per-epoch predictions
    epoch_pred_dir = os.path.join(output_dir, "epoch_predictions")
    os.makedirs(epoch_pred_dir, exist_ok=True)
    epoch_prediction_files = []
    
    loss_history = []
    n_boundaries_history = []
    epoch_times = []
    
    total_start = time.time()
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Encode with current encoder (no grad for inference)
        features = feature_extractor.encode_features(resnet_features, requires_grad=False)
        features = features.to(device)
        
        # Compute TSM
        tsm = compute_tsm(features)
        
        # Generate pseudo-labels with selected method
        with torch.no_grad():
            if boundary_method == 'rtp':
                boundaries = rtp.parse_recursive(tsm, top_k=rtp_top_k)
            else:
                boundaries = find_boundaries_peaks(
                    tsm,
                    kernel_size=rtp_kernel_size,
                    distance=peaks_distance,
                    prominence=peaks_prominence,
                    max_boundaries=peaks_max_boundaries
                )
            boundaries = sorted(list(set(boundaries)))
        
        # Save per-epoch predictions (boundaries and times)
        boundary_times_epoch = [frame_indices[b] / video_fps for b in boundaries if b < len(frame_indices)]
        epoch_base = f"epoch_{epoch + 1:03d}"
        json_path = os.path.join(epoch_pred_dir, f"{epoch_base}.json")
        txt_path = os.path.join(epoch_pred_dir, f"{epoch_base}_times.txt")
        epoch_payload = {
            'epoch': epoch + 1,
            'n_boundaries': len(boundaries),
            'boundaries': boundaries,
            'boundary_times': boundary_times_epoch
        }
        with open(json_path, 'w') as f:
            json.dump(epoch_payload, f, indent=2)
        with open(txt_path, 'w') as f:
            for t in boundary_times_epoch:
                f.write(f"{t:.2f}\n")
        epoch_prediction_files.append({
            'epoch': epoch + 1,
            'json': json_path,
            'times_txt': txt_path,
            'n_boundaries': len(boundaries)
        })

        n_boundaries_history.append(len(boundaries))
        
        # Compute BoCo loss (with gradients)
        optimizer.zero_grad()
        features_train = feature_extractor.encode_features(resnet_features, requires_grad=True)
        tsm_train = compute_tsm(features_train)
        loss = compute_boco_loss(tsm_train, boundaries, gap=boco_gap)
        
        if loss.item() > 0:
            loss.backward()
            optimizer.step()
        
        loss_history.append(loss.item())
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Calculate ETA
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = n_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Progress display
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch + 1}/{n_epochs} [{(epoch+1)/n_epochs*100:.0f}%]")
        print(f"  Time: {epoch_time:.2f}s | ETA: {eta_str}")
        print(f"  Boundaries detected: {len(boundaries)}")
        print(f"  BoCo Loss: {loss.item():.4f}")
        
        if epoch > 0:
            delta_bounds = len(boundaries) - n_boundaries_history[-2]
            delta_loss = loss.item() - loss_history[-2]
            print(f"  Change: Δboundaries={delta_bounds:+d}, Δloss={delta_loss:+.4f}")
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average epoch time: {np.mean(epoch_times):.2f}s")
    
    # Final inference
    print(f"\n{'='*60}")
    print(f"STEP 4: FINAL BOUNDARY DETECTION")
    print(f"{'='*60}")
    
    with torch.no_grad():
        features_final = feature_extractor.encode_features(resnet_features, requires_grad=False)
        tsm_final = compute_tsm(features_final)
        if boundary_method == 'rtp':
            boundaries_final = rtp.parse_recursive(tsm_final, top_k=rtp_top_k)
        else:
            boundaries_final = find_boundaries_peaks(
                tsm_final,
                kernel_size=rtp_kernel_size,
                distance=peaks_distance,
                prominence=peaks_prominence,
                max_boundaries=peaks_max_boundaries
            )
        boundaries_final = sorted(list(set(boundaries_final)))
    
    boundary_times = [frame_indices[b] / video_fps for b in boundaries_final if b < len(frame_indices)]
    
    print(f"✓ Detected {len(boundaries_final)} event boundaries")
    print(f"\nBoundary times (seconds):")
    for i, t in enumerate(boundary_times, 1):
        print(f"  {i:2d}. {t:6.2f}s")
    
    results = {
        'boundaries': boundaries_final,
        'boundary_times': boundary_times,
        'n_boundaries': len(boundaries_final),
        'tsm': tsm_final.cpu().numpy(),
        'features': features_final.cpu().numpy(),
        'frame_indices': frame_indices,
        'loss_history': loss_history,
        'n_boundaries_history': n_boundaries_history,
        'training_time': total_time,
        'epoch_times': epoch_times
        ,
        'epoch_predictions': epoch_prediction_files
    }
    
    return results, video_fps


def visualize_uboco_results(
    results: Dict,
    frame_indices: List[int],
    video_fps: float,
    output_dir: str = 'outputs'
):
    """Visualize UBoCo results."""
    os.makedirs(output_dir, exist_ok=True)
    
    tsm = results['tsm']
    boundaries = results['boundaries']
    times = np.array(frame_indices) / video_fps
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. TSM with boundaries (large, top-left)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    im = ax1.imshow(tsm, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title('Temporal Self-Similarity Matrix (TSM)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frame', fontsize=11)
    ax1.set_ylabel('Frame', fontsize=11)
    
    for b in boundaries:
        if b < len(frame_indices):
            ax1.axhline(b, color='lime', alpha=0.6, linewidth=1.5)
            ax1.axvline(b, color='lime', alpha=0.6, linewidth=1.5)
    
    plt.colorbar(im, ax=ax1, label='Cosine Similarity', fraction=0.046)
    
    # 2. Training loss
    ax2 = fig.add_subplot(gs[0, 2])
    epochs = np.arange(1, len(results['loss_history']) + 1)
    ax2.plot(epochs, results['loss_history'], marker='o', linewidth=2, markersize=6, color='#e74c3c')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('BoCo Loss', fontsize=10)
    ax2.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    # 3. Number of boundaries
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(epochs, results['n_boundaries_history'], marker='s', linewidth=2, markersize=6, color='#3498db')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('# Boundaries', fontsize=10)
    ax3.set_title('Boundary Count', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(epochs)
    
    # 4. Boundary timeline (bottom, full width)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.eventplot(results['boundary_times'], colors='#e74c3c', linewidths=3)
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_title(f'Detected Event Boundaries (n={results["n_boundaries"]})', fontsize=12, fontweight='bold')
    ax4.set_ylim(0.5, 1.5)
    ax4.set_yticks([])
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add boundary time labels
    for i, t in enumerate(results['boundary_times']):
        if i % max(1, len(results['boundary_times']) // 20) == 0:  # Label every nth boundary
            ax4.text(t, 1.0, f'{t:.1f}s', rotation=45, ha='right', va='bottom', fontsize=8)
    
    plt.suptitle('UBoCo: Event Boundary Detection Results', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'{output_dir}/uboco_results.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_dir}/uboco_results.png")
    
    # Diagonal boundary scores
    rtp = RecursiveTSMParser()
    tsm_tensor = torch.tensor(tsm, dtype=torch.float32)
    scores = rtp.diagonal_conv(tsm_tensor).numpy()
    
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(times, scores, linewidth=1.5, alpha=0.8, color='#2c3e50')
    ax.fill_between(times, scores, alpha=0.2, color='#3498db')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Boundary Score', fontsize=12)
    ax.set_title('Diagonal Boundary Scores (RTP Contrastive Kernel Response)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for b in boundaries:
        if b < len(times):
            ax.axvline(times[b], color='#e74c3c', alpha=0.7, linestyle='--', linewidth=2)
            ax.text(times[b], ax.get_ylim()[1]*0.95, f'{times[b]:.1f}s', 
                   rotation=90, ha='right', va='top', fontsize=9, color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boundary_scores.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved boundary scores to {output_dir}/boundary_scores.png")


def save_uboco_results(
    results: Dict,
    frame_indices: List[int],
    video_path: str,
    video_fps: float,
    output_dir: str = 'outputs'
):
    """Save UBoCo results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        'video_path': video_path,
        'fps': video_fps,
        'n_frames_analyzed': len(frame_indices),
        'n_boundaries': results['n_boundaries'],
        'training_time_seconds': results['training_time'],
        'training_time_minutes': results['training_time'] / 60,
        'average_epoch_time': float(np.mean(results['epoch_times'])),
        'frame_indices': frame_indices,
        'boundaries': results['boundaries'],
        'boundary_times': results['boundary_times'],
        'loss_history': results['loss_history'],
        'n_boundaries_history': results['n_boundaries_history']
    }
    
    output_path = f'{output_dir}/uboco_boundaries.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Saved results to {output_path}")
    
    # Save boundary times as simple text
    boundary_file = f'{output_dir}/boundary_times.txt'
    with open(boundary_file, 'w') as f:
        f.write("# Event boundaries (seconds) - UBoCo\n")
        f.write(f"# Video: {video_path}\n")
        f.write(f"# Total boundaries: {len(results['boundary_times'])}\n")
        f.write("#\n")
        for i, t in enumerate(results['boundary_times'], 1):
            f.write(f"{t:.2f}\n")
    print(f"✓ Saved boundary times to {boundary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='UBoCo: Unsupervised Boundary Contrastive Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--fps_sample', type=float, default=25, help='Sample rate in fps')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for feature extraction')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Encoder output dimension')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--rtp_top_k', type=float, default=0.05, help='RTP top-k percentage for sampling')
    parser.add_argument('--boco_gap', type=int, default=20, help='BoCo local similarity gap')
    parser.add_argument('--output_dir', type=str, default='outputs_uboco', help='Output directory')
    parser.add_argument('--start_time', type=float, default=None, help='Start time in seconds')
    parser.add_argument('--end_time', type=float, default=None, help='End time in seconds')
    # Boundary method controls
    parser.add_argument('--boundary_method', type=str, default='peaks', choices=['rtp','peaks'], help='Boundary detection method')
    parser.add_argument('--rtp_kernel_size', type=int, default=5, help='RTP kernel size')
    parser.add_argument('--rtp_min_length', type=int, default=50, help='RTP minimum segment length to recurse')
    parser.add_argument('--rtp_threshold_diff', type=float, default=0.3, help='RTP min (max-mean) score to recurse')
    parser.add_argument('--rtp_max_depth', type=int, default=3, help='RTP maximum recursion depth')
    parser.add_argument('--rtp_max_boundaries', type=int, default=30, help='RTP global maximum number of boundaries')
    parser.add_argument('--peaks_distance', type=int, default=30, help='Peaks: minimum frames between boundaries')
    parser.add_argument('--peaks_prominence', type=float, default=0.6, help='Peaks: required prominence over baseline')
    parser.add_argument('--peaks_max_boundaries', type=int, default=25, help='Peaks: cap number of boundaries')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    
    print("\n" + "="*60)
    print("UBoCo: UNSUPERVISED BOUNDARY CONTRASTIVE LEARNING")
    print("="*60)
    print("Based on: Kang et al. (CVPR 2022)")
    print("Self-supervised with pseudo-labels from RTP algorithm")
    print("="*60)
    
    # Initialize
    feature_extractor = VideoFeatureExtractor(encoder_dim=args.encoder_dim)
    
    # Train
    overall_start = time.time()
    results, video_fps = train_uboco(
        feature_extractor,
        args.video_path,
        fps_sample=args.fps_sample,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rtp_top_k=args.rtp_top_k,
        boco_gap=args.boco_gap,
        start_time=args.start_time,
        end_time=args.end_time,
        output_dir=args.output_dir,
        boundary_method=args.boundary_method,
        rtp_kernel_size=args.rtp_kernel_size,
        rtp_min_length=args.rtp_min_length,
        rtp_threshold_diff=args.rtp_threshold_diff,
        rtp_max_depth=args.rtp_max_depth,
        rtp_max_boundaries=args.rtp_max_boundaries,
        peaks_distance=args.peaks_distance,
        peaks_prominence=args.peaks_prominence,
        peaks_max_boundaries=args.peaks_max_boundaries
    )
    overall_time = time.time() - overall_start
    
    # Save and visualize
    print(f"\n{'='*60}")
    print(f"STEP 5: SAVING RESULTS")
    print(f"{'='*60}")
    
    save_uboco_results(results, results['frame_indices'], args.video_path, video_fps, args.output_dir)
    visualize_uboco_results(results, results['frame_indices'], video_fps, args.output_dir)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"Training time: {results['training_time']:.1f}s ({results['training_time']/60:.1f} min)")
    print(f"Detected boundaries: {results['n_boundaries']}")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()