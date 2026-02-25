#!/usr/bin/env python3
"""
Efficient grid search over fps_sample, n_states, layer, and n_pca.

Key optimization: Extract features ONCE for all frames at all layers, 
then reuse them for all parameter combinations.

Strategy:
1. Extract features for ALL frames at ALL specified layers (one video pass per model)
2. For each layer:
   - For each n_pca: Compute PCA on all frames ONCE
   - For each fps_sample: Subsample the pre-computed features
   - For each n_states: Fit HMM (fast)
   - Evaluate boundaries

This is ~100x faster than the naive grid search.
"""

import argparse
import json
import os
import subprocess
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm


class VideoFeatureExtractor:
    """Extract vision features from video frames."""
    
    def __init__(self, model_name: str = 'facebook/dinov2-base', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        name_lower = model_name.lower()
        if 'clip' in name_lower:
            self.model_family = 'clip'
        else:
            self.model_family = 'vit'

        print(f"Loaded model '{model_name}' as {self.model_family.upper()} on {self.device}")
        
    def extract_all_layers(
        self, 
        video_path: str,
        layers: List[int],
        batch_size: int = 32,
        start_time: float = None,
        end_time: float = None,
        center_crop: bool = False,
        crop_size: int = None
    ) -> Tuple[Dict[int, np.ndarray], List[int], float]:
        """
        Extract features for ALL frames at multiple layers in one pass.
        
        Returns:
            features_by_layer: {layer_idx: features_array} where features_array is (n_frames, feat_dim)
            frame_indices: List of frame indices
            video_fps: Video FPS
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute start/end frames
        start_frame = 0 if start_time is None else max(0, int(start_time * video_fps))
        end_frame = (total_frames - 1) if end_time is None else min(total_frames - 1, int(end_time * video_fps))
        
        if end_frame < start_frame:
            raise ValueError("end_time must be greater than start_time")

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        segment_frames = end_frame - start_frame + 1

        print(f"\nVideo: {video_fps:.1f} fps, {total_frames} frames total")
        print(f"Processing segment: frames {start_frame} to {end_frame} ({segment_frames} frames)")
        print(f"Extracting features for all frames at layers: {layers}")

        # We'll extract ALL frames (no subsampling at this stage)
        frame_batch = []
        frame_indices = []
        features_by_layer = {layer: [] for layer in layers}
        
        global_frame_idx = start_frame
        pbar = tqdm(total=segment_frames, desc="Extracting features (all frames)")

        while True:
            if global_frame_idx > end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if center_crop:
                h, w = frame.shape[:2]
                crop_edge = min(h, w) if crop_size is None else min(crop_size, h, w)
                y0 = (h - crop_edge) // 2
                x0 = (w - crop_edge) // 2
                frame = frame[y0:y0+crop_edge, x0:x0+crop_edge]

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_batch.append(image)
            frame_indices.append(global_frame_idx)

            if len(frame_batch) >= batch_size:
                batch_features = self._process_batch_multilayer(frame_batch, layers)
                for layer in layers:
                    features_by_layer[layer].extend(batch_features[layer])
                frame_batch = []
                pbar.update(batch_size)

            global_frame_idx += 1

        # Process remaining frames
        if frame_batch:
            batch_features = self._process_batch_multilayer(frame_batch, layers)
            for layer in layers:
                features_by_layer[layer].extend(batch_features[layer])
            pbar.update(len(frame_batch))

        cap.release()
        pbar.close()
        
        # Convert to numpy arrays
        for layer in layers:
            features_by_layer[layer] = np.array(features_by_layer[layer])
            print(f"Layer {layer}: extracted {features_by_layer[layer].shape}")
        
        return features_by_layer, frame_indices, video_fps
    
    def _process_batch_multilayer(self, images: List[Image.Image], layers: List[int]) -> Dict[int, List[np.ndarray]]:
        """Process a batch and return features for multiple layers."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        batch_features = {layer: [] for layer in layers}

        with torch.no_grad():
            if self.model_family == 'clip':
                if hasattr(self.model, 'vision_model') and self.model.vision_model is not None:
                    vision_out = self.model.vision_model(
                        pixel_values=inputs['pixel_values'],
                        output_hidden_states=True
                    )
                    if getattr(vision_out, 'hidden_states', None) is not None:
                        for layer in layers:
                            hidden_states = vision_out.hidden_states[layer]
                            features_tensor = hidden_states.mean(dim=1)
                            batch_features[layer] = [f for f in features_tensor.cpu().numpy()]
                    else:
                        raise RuntimeError("CLIP model did not return hidden_states")
                else:
                    raise RuntimeError("CLIP model does not have vision_model")
            else:
                # ViT/DINO-like models
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    for layer in layers:
                        hidden_states = outputs.hidden_states[layer]
                        features_tensor = hidden_states.mean(dim=1)
                        batch_features[layer] = [f for f in features_tensor.cpu().numpy()]
                else:
                    raise RuntimeError("Model did not return hidden_states")

        return batch_features


class TimmVitFeatureExtractor:
    """Extract vision features from timm ViT backbones using forward hooks per transformer block."""
    def __init__(self, variant: str, ckpt_path: str, timm_model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.variant = variant
        self.timm_model_name = timm_model_name

        if variant == 'mocov3_vit':
            from mocov3_vit_loader import create_mocov3_vit_from_checkpoint
            self.model, missing, unexpected = create_mocov3_vit_from_checkpoint(
                ckpt_path,
                model_name=timm_model_name,
                global_pool='avg',
                device=torch.device(self.device),
                map_location='cpu',
                strict=False,
            )
        elif variant == 'dinov2_vit':
            from dinov2_vit_loader import create_dinov2_vit_from_checkpoint
            self.model, missing, unexpected = create_dinov2_vit_from_checkpoint(
                ckpt_path,
                model_name=timm_model_name,
                global_pool=None,
                device=torch.device(self.device),
                map_location='cpu',
                strict=False,
                img_size=224,
            )
        else:
            raise ValueError(f"Unsupported timm ViT variant: {variant}")

        # Resolve transforms using timm defaults, but enforce 224x224 resize/crop
        import timm
        from timm.data import resolve_data_config
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        self.data_config = resolve_data_config({}, model=self.model)
        mean = self.data_config.get('mean', (0.485, 0.456, 0.406))
        std = self.data_config.get('std', (0.229, 0.224, 0.225))
        target_size = 224
        self.transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Register forward hooks on each transformer block
        self._hook_outputs = []
        self._handles = []
        for blk in getattr(self.model, 'blocks'):
            handle = blk.register_forward_hook(self._hook_fn)
            self._handles.append(handle)

        print(f"Loaded TIMM model '{timm_model_name}' for variant {variant} on {self.device}")

    def _hook_fn(self, module, inputs, output):
        # output: (B, tokens, C)
        self._hook_outputs.append(output)

    def _process_batch_multilayer(self, images: List[Image.Image], layers: List[int]) -> Dict[int, List[np.ndarray]]:
        batch_features = {layer: [] for layer in layers}
        tensors = [self.transform(img).to(self.device) for img in images]
        pixel_values = torch.stack(tensors, dim=0)

        with torch.no_grad():
            self._hook_outputs = []
            try:
                _ = self.model.forward_features(pixel_values)
            except AttributeError:
                _ = self.model(pixel_values)

            num_blocks = len(self._hook_outputs)
            for layer in layers:
                idx = max(0, min(layer - 1, num_blocks - 1))
                feats = self._hook_outputs[idx]
                if feats.dim() == 3:
                    pooled = feats.mean(dim=1)
                else:
                    pooled = feats
                batch_features[layer] = [f for f in pooled.detach().cpu().numpy()]

        return batch_features

    def extract_all_layers(
        self,
        video_path: str,
        layers: List[int],
        batch_size: int = 32,
        start_time: float = None,
        end_time: float = None,
        center_crop: bool = False,
        crop_size: int = None
    ) -> Tuple[Dict[int, np.ndarray], List[int], float]:
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = 0 if start_time is None else max(0, int(start_time * video_fps))
        end_frame = (total_frames - 1) if end_time is None else min(total_frames - 1, int(end_time * video_fps))
        if end_frame < start_frame:
            raise ValueError("end_time must be greater than start_time")
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        segment_frames = end_frame - start_frame + 1
        print(f"\nVideo: {video_fps:.1f} fps, {total_frames} frames total")
        print(f"Processing segment: frames {start_frame} to {end_frame} ({segment_frames} frames)")
        print(f"Extracting features for all frames at layers: {layers}")

        frame_batch = []
        frame_indices = []
        features_by_layer = {layer: [] for layer in layers}

        global_frame_idx = start_frame
        pbar = tqdm(total=segment_frames, desc="Extracting features (all frames)")

        while True:
            if global_frame_idx > end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_batch.append(image)
            frame_indices.append(global_frame_idx)

            if len(frame_batch) >= batch_size:
                batch_features = self._process_batch_multilayer(frame_batch, layers)
                for layer in layers:
                    features_by_layer[layer].extend(batch_features[layer])
                frame_batch = []
                pbar.update(batch_size)

            global_frame_idx += 1

        if frame_batch:
            batch_features = self._process_batch_multilayer(frame_batch, layers)
            for layer in layers:
                features_by_layer[layer].extend(batch_features[layer])
            pbar.update(len(frame_batch))

        cap.release()
        pbar.close()

        for layer in layers:
            features_by_layer[layer] = np.array(features_by_layer[layer])
            print(f"Layer {layer}: extracted {features_by_layer[layer].shape}")

        return features_by_layer, frame_indices, video_fps


class VJepaFeatureExtractor:
    """Extract features from a V-JEPA (HF) model. Maps all requested layers to the same embedding."""
    def __init__(self, model_id: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        from vjepa_loader import init_vjepa
        self.model, self.processor = init_vjepa(model_id=model_id, device=torch.device(self.device))
        print(f"Loaded V-JEPA model '{model_id}' on {self.device}")

    def _process_batch_multilayer(self, images: List[Image.Image], layers: List[int]) -> Dict[int, List[np.ndarray]]:
        batch_features = {layer: [] for layer in layers}
        # Build a batch of single-frame videos expected by AutoVideoProcessor
        videos = [[img] for img in images]  # list of length B, each item is a list of 1 PIL image
        inputs = self.processor(videos=videos, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, 'get_vision_features'):
                feats = self.model.get_vision_features(**inputs)
            else:
                outputs = self.model(**inputs)
                feats = getattr(outputs, 'last_hidden_state', None)
                if feats is None:
                    feats = getattr(outputs, 'pooler_output', None)
                if feats is None:
                    raise RuntimeError('V-JEPA model did not return recognizable features')

            # Normalize features to (B, D)
            if feats.dim() == 5:  # (B, C, T, H, W) unlikely here
                feats = feats.mean(dim=(2,3,4))
            elif feats.dim() == 4:  # (B, T, tokens, C) or (B, T, H, W)
                feats = feats.mean(dim=(1,2))
            elif feats.dim() == 3:  # (B, tokens, C)
                feats = feats.mean(dim=1)
            # if (B, D) leave as is

            arr = feats.detach().cpu().numpy()
            for layer in layers:
                batch_features[layer] = [f for f in arr]
        return batch_features

    def extract_all_layers(
        self,
        video_path: str,
        layers: List[int],
        batch_size: int = 32,
        start_time: float = None,
        end_time: float = None,
        center_crop: bool = False,
        crop_size: int = None
    ) -> Tuple[Dict[int, np.ndarray], List[int], float]:
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = 0 if start_time is None else max(0, int(start_time * video_fps))
        end_frame = (total_frames - 1) if end_time is None else min(total_frames - 1, int(end_time * video_fps))
        if end_frame < start_frame:
            raise ValueError("end_time must be greater than start_time")
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        segment_frames = end_frame - start_frame + 1
        print(f"\nVideo: {video_fps:.1f} fps, {total_frames} frames total")
        print(f"Processing segment: frames {start_frame} to {end_frame} ({segment_frames} frames)")
        print(f"Extracting features for all frames at layers: {layers} (collapsed to single JEPA embedding)")

        frame_batch = []
        frame_indices = []
        features_by_layer = {layer: [] for layer in layers}

        global_frame_idx = start_frame
        pbar = tqdm(total=segment_frames, desc="Extracting features (all frames)")

        while True:
            if global_frame_idx > end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_batch.append(image)
            frame_indices.append(global_frame_idx)

            if len(frame_batch) >= batch_size:
                batch_features = self._process_batch_multilayer(frame_batch, layers)
                for layer in layers:
                    features_by_layer[layer].extend(batch_features[layer])
                frame_batch = []
                pbar.update(batch_size)

            global_frame_idx += 1

        if frame_batch:
            batch_features = self._process_batch_multilayer(frame_batch, layers)
            for layer in layers:
                features_by_layer[layer].extend(batch_features[layer])
            pbar.update(len(frame_batch))

        cap.release()
        pbar.close()

        for layer in layers:
            features_by_layer[layer] = np.array(features_by_layer[layer])
            print(f"Layer {layer}: extracted {features_by_layer[layer].shape}")

        return features_by_layer, frame_indices, video_fps

class ResNetFeatureExtractor:
    """Extract features from a ResNet backbone using forward hooks on bottleneck blocks."""
    def __init__(self, ckpt_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        from mocov3_resnet_loader import create_mocov3_resnet50_from_checkpoint
        self.model, missing, unexpected = create_mocov3_resnet50_from_checkpoint(
            ckpt_path,
            device=torch.device(self.device),
            map_location='cpu',
            strict=False,
        )
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Collect bottleneck modules in forward order
        self._bnecks = []
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.model, layer_name)
            for m in layer:
                self._bnecks.append(m)

        self._hook_outputs = []
        self._handles = []
        for m in self._bnecks:
            handle = m.register_forward_hook(self._hook_fn)
            self._handles.append(handle)
        print(f"Loaded ResNet50 backbone for MoCo v3 on {self.device}")

    def _hook_fn(self, module, inputs, output):
        self._hook_outputs.append(output)

    def _process_batch_multilayer(self, images: List[Image.Image], layers: List[int]) -> Dict[int, List[np.ndarray]]:
        import torch.nn.functional as F
        batch_features = {layer: [] for layer in layers}
        tensors = [self.transform(img).to(self.device) for img in images]
        x = torch.stack(tensors, dim=0)
        with torch.no_grad():
            self._hook_outputs = []
            _ = self.model(x)
            total = len(self._hook_outputs)
            for layer in layers:
                idx = max(0, min(layer - 1, total - 1))
                feat_map = self._hook_outputs[idx]  # (B, C, H, W)
                pooled = F.adaptive_avg_pool2d(feat_map, output_size=1).flatten(1)
                batch_features[layer] = [f for f in pooled.detach().cpu().numpy()]
        return batch_features

    def extract_all_layers(
        self,
        video_path: str,
        layers: List[int],
        batch_size: int = 32,
        start_time: float = None,
        end_time: float = None,
        center_crop: bool = False,
        crop_size: int = None
    ) -> Tuple[Dict[int, np.ndarray], List[int], float]:
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = 0 if start_time is None else max(0, int(start_time * video_fps))
        end_frame = (total_frames - 1) if end_time is None else min(total_frames - 1, int(end_time * video_fps))
        if end_frame < start_frame:
            raise ValueError("end_time must be greater than start_time")
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        segment_frames = end_frame - start_frame + 1
        print(f"\nVideo: {video_fps:.1f} fps, {total_frames} frames total")
        print(f"Processing segment: frames {start_frame} to {end_frame} ({segment_frames} frames)")
        print(f"Extracting features for all frames at layers: {layers}")

        frame_batch = []
        frame_indices = []
        features_by_layer = {layer: [] for layer in layers}

        global_frame_idx = start_frame
        pbar = tqdm(total=segment_frames, desc="Extracting features (all frames)")

        while True:
            if global_frame_idx > end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_batch.append(image)
            frame_indices.append(global_frame_idx)
            if len(frame_batch) >= batch_size:
                batch_features = self._process_batch_multilayer(frame_batch, layers)
                for layer in layers:
                    features_by_layer[layer].extend(batch_features[layer])
                frame_batch = []
                pbar.update(batch_size)
            global_frame_idx += 1

        if frame_batch:
            batch_features = self._process_batch_multilayer(frame_batch, layers)
            for layer in layers:
                features_by_layer[layer].extend(batch_features[layer])
            pbar.update(len(frame_batch))

        cap.release()
        pbar.close()

        for layer in layers:
            features_by_layer[layer] = np.array(features_by_layer[layer])
            print(f"Layer {layer}: extracted {features_by_layer[layer].shape}")

        return features_by_layer, frame_indices, video_fps


def subsample_features(features: np.ndarray, frame_indices: List[int], 
                       video_fps: float, fps_sample: float) -> Tuple[np.ndarray, List[int]]:
    """
    Subsample pre-extracted features to a target fps.
    
    Args:
        features: Full feature array (n_frames, feat_dim)
        frame_indices: Original frame indices
        video_fps: Original video FPS
        fps_sample: Target sampling rate
    
    Returns:
        subsampled_features, subsampled_frame_indices
    """
    if fps_sample <= 0:
        raise ValueError("fps_sample must be > 0")
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps_sample))
    
    # Subsample
    indices = np.arange(0, len(features), frame_interval)
    return features[indices], [frame_indices[i] for i in indices]


def fit_hmm_and_detect_boundaries(
    features: np.ndarray,
    n_states: int,
    n_pca: int,
    covariance_type: str = 'diag',
    n_iter: int = 100,
    min_covar: float = 1e-3
) -> Dict[str, Any]:
    """
    Fit HMM on features and detect boundaries.
    
    Args:
        features: Feature array (n_frames, feature_dim) - already PCA reduced
        n_states: Number of HMM states
        n_pca: Number of PCA components (for metadata)
        
    Returns:
        Dictionary with states, boundaries, etc.
    """
    # Fit HMM
    model = hmm.GaussianHMM(
        n_components=n_states, 
        covariance_type=covariance_type,
        n_iter=n_iter,
        verbose=False,
        min_covar=min_covar
    )
    try:
        model.fit(features)
    except ValueError as exc:
        raise RuntimeError(f"HMM fit failed: {exc}") from exc

    if hasattr(model, 'monitor_') and not getattr(model.monitor_, 'converged', True):
        raise RuntimeError("HMM failed to converge")

    # Ensure start probabilities are valid
    startprob = np.nan_to_num(model.startprob_, nan=0.0)
    startprob_sum = startprob.sum()
    if startprob_sum <= 0 or not np.all(np.isfinite(startprob)):
        raise RuntimeError("Invalid start probability vector")
    model.startprob_ = startprob / startprob_sum

    # Ensure transition matrix rows sum to 1 and contain no NaNs
    transmat = np.nan_to_num(model.transmat_, nan=0.0)
    row_sums = transmat.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() == 0
    if np.any(zero_rows):
        transmat[zero_rows] = 1.0 / model.n_components
        row_sums = transmat.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0) or not np.all(np.isfinite(transmat)):
        raise RuntimeError("Invalid transition matrix")
    model.transmat_ = transmat / row_sums
    
    # Get state sequence
    try:
        states = model.predict(features)
    except ValueError as exc:
        raise RuntimeError(f"HMM decoding failed: {exc}") from exc
    log_likelihood = model.score(features)
    
    # Find boundaries (where state changes)
    boundaries = np.where(np.diff(states) != 0)[0] + 1
    
    return {
        'states': states,
        'boundaries': boundaries,
        'n_boundaries': len(boundaries),
        'log_likelihood': log_likelihood,
    }


def run_evaluation(
    boundaries_json: str,
    ground_truth_path: str,
    video_fps: float,
    tolerances: List[int],
    det_offset_sec: float,
    gt_max_rows: int = None
) -> Dict[str, Any]:
    """
    Run evaluation using the evaluate_boundaries.py script.
    Returns metrics dictionary.
    """
    import tempfile
    
    # Create temp file for eval output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        eval_out = f.name
    
    try:
        eval_cmd = [
            'python', os.path.join(os.path.dirname(__file__), 'evaluate_boundaries.py'),
            boundaries_json,
            ground_truth_path,
            '--fps', str(video_fps),
            '--tolerances', *[str(t) for t in tolerances],
            '--det_offset_sec', str(det_offset_sec),
            '--output', eval_out
        ]
        if gt_max_rows is not None:
            eval_cmd.extend(['--gt_max_rows', str(gt_max_rows)])
        
        subprocess.run(eval_cmd, check=True, capture_output=True)
        
        with open(eval_out, 'r') as f:
            results = json.load(f)
        
        return results['metrics']
    finally:
        if os.path.exists(eval_out):
            os.remove(eval_out)


def save_boundaries_json(
    boundaries: np.ndarray,
    frame_indices: List[int],
    video_fps: float,
    output_path: str
):
    """Save boundaries to JSON format compatible with evaluate_boundaries.py"""
    boundary_times = [
        (frame_indices[idx] / video_fps) for idx in boundaries if idx < len(frame_indices)
    ]
    
    output = {
        'fps': video_fps,
        'n_boundaries': len(boundaries),
        'boundaries': boundaries.tolist(),
        'boundary_times': boundary_times,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Efficient grid search over parameters')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('ground_truth_path', type=str, help='Ground truth boundaries (XLSX/CSV/TXT)')
    parser.add_argument('--output_root', type=str, default='efficient_grid_runs', help='Root directory for runs')
    parser.add_argument('--fps_samples', nargs='+', type=int, default=[5, 15, 30])
    parser.add_argument('--n_states_list', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--layers', nargs='+', type=int, default=[1,2,3,4,5,6,7,8,9,10,11,12])
    parser.add_argument('--n_pca_list', nargs='+', type=int, default=[20, 30, 50])
    parser.add_argument('--tolerances', nargs='+', type=int, default=[10, 15, 25])
    parser.add_argument('--start_time', type=float, default=0)
    parser.add_argument('--end_time', type=float, default=1600)
    parser.add_argument('--model', type=str, default='facebook/dinov2-base')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Optional list of model names to sweep; overrides --model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gt_max_rows', type=int, default=482, help='Max rows to read from ground truth')
    parser.add_argument('--hmm_iter', type=int, default=100, help='HMM training iterations')
    parser.add_argument('--hmm_covariance_type', type=str, default='diag',
                        choices=['full', 'diag', 'tied', 'spherical'],
                        help='Covariance structure for GaussianHMM')
    parser.add_argument('--hmm_min_covar', type=float, default=1e-3,
                        help='Diagonal added to covariance matrices for stability')
    # Custom backbone options
    parser.add_argument('--backbone_type', type=str, default='hf',
                        choices=['hf', 'mocov3_vit', 'dinov2_vit', 'mocov3_resnet50', 'vjepa2'],
                        help='Backbone source: hf (transformers) or custom checkpoint loaders')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to checkpoint for custom backbones')
    parser.add_argument('--timm_model_name', type=str, default=None,
                        help='timm model name for ViT variants (e.g., vit_base_patch16_224, vit_large_patch14_224)')
    parser.add_argument('--vjepa_model_id', type=str, default='facebook/vjepa2-vitl-fpc64-256',
                        help='HF model id for V-JEPA (JEPA2)')
    parser.add_argument('--min_samples_per_state', type=float, default=5.0,
                        help='Minimum samples per state before fitting; combos below are skipped')

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    
    records: List[Dict[str, Any]] = []
    models_to_try = args.models if args.models else [args.model]

    for model_name in models_to_try:
        model_tag = model_name.replace('/', '-')
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}")
        
        # Step 1: Extract features for ALL frames at ALL layers (one pass!)
        if args.backbone_type == 'hf':
            extractor = VideoFeatureExtractor(model_name=model_name)
        elif args.backbone_type == 'mocov3_vit':
            timm_name = args.timm_model_name or 'vit_base_patch16_224'
            if not args.ckpt_path:
                raise ValueError('--ckpt_path is required for backbone_type=mocov3_vit')
            extractor = TimmVitFeatureExtractor('mocov3_vit', args.ckpt_path, timm_name)
        elif args.backbone_type == 'dinov2_vit':
            timm_name = args.timm_model_name or 'vit_large_patch14_224'
            if not args.ckpt_path:
                raise ValueError('--ckpt_path is required for backbone_type=dinov2_vit')
            extractor = TimmVitFeatureExtractor('dinov2_vit', args.ckpt_path, timm_name)
        elif args.backbone_type == 'mocov3_resnet50':
            if not args.ckpt_path:
                raise ValueError('--ckpt_path is required for backbone_type=mocov3_resnet50')
            extractor = ResNetFeatureExtractor(args.ckpt_path)
        elif args.backbone_type == 'vjepa2':
            extractor = VJepaFeatureExtractor(args.vjepa_model_id)
        else:
            raise ValueError(f"Unknown backbone_type: {args.backbone_type}")
        features_by_layer, frame_indices, video_fps = extractor.extract_all_layers(
            args.video_path,
            layers=args.layers,
            batch_size=args.batch_size,
            start_time=args.start_time,
            end_time=args.end_time
        )
        
        # Step 2: For each layer, compute PCA for all n_pca values
        print(f"\n{'='*80}")
        print("Computing PCA for all layer/n_pca combinations...")
        print(f"{'='*80}")
        
        # Store PCA-reduced features: pca_features[layer][n_pca] = reduced_features
        pca_features = {}
        
        for layer in args.layers:
            pca_features[layer] = {}
            features_full = features_by_layer[layer]
            
            # Standardize once per layer
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_full)
            
            for n_pca in args.n_pca_list:
                print(f"  Layer {layer}, PCA {n_pca} components...", end=' ')
                pca = PCA(n_components=n_pca)
                features_reduced = pca.fit_transform(features_scaled)
                pca_features[layer][n_pca] = features_reduced
                explained_var = pca.explained_variance_ratio_.sum()
                print(f"explained variance: {explained_var:.2%}")
        
        # Step 3: Grid search over fps_sample, n_states
        print(f"\n{'='*80}")
        print("Running grid search over fps_sample and n_states...")
        print(f"{'='*80}")
        
        total_combinations = len(args.layers) * len(args.n_pca_list) * len(args.fps_samples) * len(args.n_states_list)
        pbar = tqdm(total=total_combinations, desc="Grid search")
        
        for layer in args.layers:
            for n_pca in args.n_pca_list:
                # Get the full PCA-reduced features for this layer/n_pca
                features_full_pca = pca_features[layer][n_pca]
                
                for fps_sample in args.fps_samples:
                    # Subsample features (cheap operation)
                    features_sampled, frame_indices_sampled = subsample_features(
                        features_full_pca, frame_indices, video_fps, fps_sample
                    )
                    
                    for n_states in args.n_states_list:
                        # Create output directory
                        run_dir = os.path.join(
                            args.output_root, 
                            f"{model_tag}_fps{fps_sample}_ns{n_states}_ly{layer}_pca{n_pca}"
                        )
                        os.makedirs(run_dir, exist_ok=True)
                        
                        # Skip combinations that are too undersampled
                        samples_per_state = len(features_sampled) / max(n_states, 1)
                        if samples_per_state < args.min_samples_per_state:
                            print(
                                f"Skipping combo layer {layer}, PCA {n_pca}, fps {fps_sample}, states {n_states} "
                                f"(samples/state={samples_per_state:.2f} < {args.min_samples_per_state})"
                            )
                            pbar.update(1)
                            continue

                        # Fit HMM and detect boundaries (fast!)
                        try:
                            results = fit_hmm_and_detect_boundaries(
                                features_sampled,
                                n_states=n_states,
                                n_pca=n_pca,
                                covariance_type=args.hmm_covariance_type,
                                n_iter=args.hmm_iter,
                                min_covar=args.hmm_min_covar
                            )
                        except RuntimeError as exc:
                            print(
                                f"Skipping combo layer {layer}, PCA {n_pca}, fps {fps_sample}, states {n_states} "
                                f"due to HMM failure: {exc}"
                            )
                            pbar.update(1)
                            continue
                        
                        # Save boundaries
                        boundaries_json = os.path.join(run_dir, 'boundaries.json')
                        save_boundaries_json(
                            results['boundaries'],
                            frame_indices_sampled,
                            video_fps,
                            boundaries_json
                        )
                        
                        # Evaluate
                        metrics = run_evaluation(
                            boundaries_json,
                            args.ground_truth_path,
                            video_fps,
                            args.tolerances,
                            det_offset_sec=-args.start_time,
                            gt_max_rows=args.gt_max_rows
                        )
                        
                        # Record results
                        row: Dict[str, Any] = {
                            'model': model_tag,
                            'fps_sample': fps_sample,
                            'n_states': n_states,
                            'layer': layer,
                            'n_pca': n_pca,
                            'n_boundaries': results['n_boundaries'],
                            'log_likelihood': results['log_likelihood'],
                        }
                        for tol, vals in metrics.items():
                            row[f'{tol}_precision'] = vals['precision']
                            row[f'{tol}_recall'] = vals['recall']
                            row[f'{tol}_f1'] = vals['f1']
                        records.append(row)
                        
                        pbar.update(1)
        
        pbar.close()

    # Save CSV summary
    summary_csv = os.path.join(args.output_root, 'grid_summary.csv')
    df = pd.DataFrame.from_records(records)
    sort_cols = [c for c in ['model', 'fps_sample', 'n_states', 'layer', 'n_pca'] if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)
    df.to_csv(summary_csv, index=False)
    print(f"\n{'='*80}")
    print(f"✅ Grid search complete!")
    print(f"Saved summary to {summary_csv}")
    print(f"Total parameter combinations: {len(records)}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
