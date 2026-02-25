import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from typing import List, Dict, Tuple, Optional
from PIL import Image
import cv2

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
except ImportError:  # pragma: no cover - optional dependency for Qwen3 vision models
    apply_rotary_pos_emb_vision = None  # type: ignore

class QwenAttentionAnalyzer:
    """Extract and visualize attention patterns from Qwen2.5-VL for boundary detection analysis"""
    
    def __init__(self, segmenter):
        """
        Initialize with an existing QwenTemporalSegmenterFixed instance
        """
        self.segmenter = segmenter
        self.model = segmenter.model
        self.processor = segmenter.processor
        self.attention_weights = {}
        self.hooks = []
        
    def register_attention_hooks(self):
        """Register hooks to capture attention weights from vision encoder"""
        self.clear_hooks()
        self.attention_weights = {}
        self._qwen3_context = {}
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # Qwen2.5-VL uses different attention output formats
                # We need to handle both attention_weights and hidden_states
                if isinstance(output, tuple) and len(output) >= 2:
                    # output is typically (hidden_states, attention_weights)
                    if len(output) > 1 and output[1] is not None:
                        self.attention_weights[name] = output[1].detach().cpu()
                elif hasattr(output, 'attentions'):
                    self.attention_weights[name] = output.attentions.detach().cpu()
            return hook

        def get_qwen3_attention_hook(name):
            def hook(module, inputs, output):
                if apply_rotary_pos_emb_vision is None:
                    return
                if not inputs:
                    return

                hidden_states = inputs[0]
                if hidden_states is None or hidden_states.ndim != 2:
                    return

                # Retrieve cached kwargs captured by the pre-hook
                context = self._qwen3_context.pop(name, {})
                position_embeddings = context.get('position_embeddings')
                if position_embeddings is None:
                    return

                try:
                    qkv = module.qkv(hidden_states)
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"[WARN] Failed to access qkv output for {name}: {exc}")
                    return

                num_heads = getattr(module, 'num_heads', None)
                head_dim = getattr(module, 'head_dim', None)
                if not num_heads or not head_dim:
                    return

                seq_len = hidden_states.shape[0]
                try:
                    qkv = qkv.view(seq_len, 3, num_heads, head_dim)
                except RuntimeError as exc:  # pragma: no cover - unexpected shape
                    print(f"[WARN] Unexpected qkv shape for {name}: {qkv.shape} ({exc})")
                    return

                qkv = qkv.permute(1, 0, 2, 3)  # (3, seq_len, num_heads, head_dim)
                query_states, key_states, _ = qkv

                try:
                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb_vision(
                        query_states, key_states, cos, sin
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"[WARN] Failed to apply rotary embedding for {name}: {exc}")
                    return

                query_states = query_states.transpose(0, 1).unsqueeze(0)  # (1, num_heads, seq_len, head_dim)
                key_states = key_states.transpose(0, 1).unsqueeze(0)

                cu_seqlens = context.get('cu_seqlens')
                if cu_seqlens is not None:
                    cu = cu_seqlens.to(torch.long)
                else:
                    frame_count = getattr(self, "_current_window_frame_count", None) or 1
                    cu = torch.linspace(0, seq_len, steps=frame_count + 1, dtype=torch.long, device=query_states.device)

                if cu.numel() < 2:
                    return

                num_frames = int(cu.numel() - 1)
                frame_ids = torch.empty(seq_len, dtype=torch.long, device=query_states.device)
                for idx in range(num_frames):
                    frame_ids[cu[idx]:cu[idx + 1]] = idx

                frame_one_hot = F.one_hot(frame_ids, num_classes=num_frames).to(query_states.dtype)
                frame_counts = frame_one_hot.sum(dim=0).clamp_min(1.0)

                queries = query_states.squeeze(0)  # (num_heads, seq_len, head_dim)
                keys = key_states.squeeze(0)      # (num_heads, seq_len, head_dim)

                scaling = getattr(module, 'scaling', 1.0 / math.sqrt(head_dim))
                chunk_size = max(1, min(256, seq_len))

                head_frame_attention = torch.zeros(
                    queries.size(0), num_frames, num_frames,
                    dtype=queries.dtype, device=queries.device
                )

                for head_idx in range(queries.size(0)):
                    q = queries[head_idx]
                    k = keys[head_idx]
                    k_t = k.transpose(0, 1)  # (head_dim, seq_len)

                    for start in range(0, seq_len, chunk_size):
                        end = min(start + chunk_size, seq_len)
                        q_chunk = q[start:end]  # (chunk, head_dim)
                        scores = torch.matmul(q_chunk, k_t) * scaling  # (chunk, seq_len)
                        weights = F.softmax(scores, dim=-1)
                        frame_weights = weights @ frame_one_hot  # (chunk, num_frames)
                        src_frames = frame_ids[start:end]
                        head_frame_attention[head_idx].index_add_(0, src_frames, frame_weights)

                    head_frame_attention[head_idx] = head_frame_attention[head_idx] / frame_counts.unsqueeze(1)

                attn_weights = head_frame_attention.mean(dim=0, keepdim=True)
                self.attention_weights[name] = attn_weights.detach().cpu()

            return hook

        def get_qwen3_pre_hook(name):
            def pre_hook(module, inputs, kwargs):
                if kwargs is None:
                    return
                context = {}
                if 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
                    context['position_embeddings'] = kwargs['position_embeddings']
                if 'cu_seqlens' in kwargs and kwargs['cu_seqlens'] is not None:
                    context['cu_seqlens'] = kwargs['cu_seqlens']
                if context:
                    self._qwen3_context[name] = context
            return pre_hook
        
        # Find vision encoder attention layers
        # Qwen2.5-VL structure: model.visual (vision encoder)
        qwen3_attention_classes = {
            'Qwen3VLVisionAttention',
            'Qwen3VLMoeVisionAttention',
        }

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            vision_model = self.model.model.visual
            
            # Look for attention layers in the vision transformer
            for name, module in vision_model.named_modules():
                module_name = module.__class__.__name__
                name_lower = name.lower()

                if module_name in qwen3_attention_classes:
                    pre_hook = module.register_forward_pre_hook(get_qwen3_pre_hook(name), with_kwargs=True)
                    hook = module.register_forward_hook(get_qwen3_attention_hook(name), with_kwargs=False)
                    self.hooks.extend([pre_hook, hook])
                    print(f"[DEBUG] Registered Qwen3 attention hook for: {name}")
                elif (
                    'attn' in name_lower
                    and not any(part in name_lower for part in ('qkv', 'proj'))
                    and hasattr(module, 'forward')
                ):
                    hook = module.register_forward_hook(get_attention_hook(name))
                    self.hooks.append(hook)
                    print(f"[DEBUG] Registered hook for: {name}")
        
        # Also try to hook into the main model's vision processing
        if hasattr(self.model, 'vision_model'):
            for name, module in self.model.vision_model.named_modules():
                module_name = module.__class__.__name__
                name_lower = name.lower()

                if module_name in qwen3_attention_classes:
                    full_name = f"vision_{name}"
                    pre_hook = module.register_forward_pre_hook(get_qwen3_pre_hook(full_name), with_kwargs=True)
                    hook = module.register_forward_hook(get_qwen3_attention_hook(full_name), with_kwargs=False)
                    self.hooks.extend([pre_hook, hook])
                    print(f"[DEBUG] Registered Qwen3 attention hook for {full_name}")
                elif 'attention' in name_lower and not name_lower.endswith('.qkv') and hasattr(module, 'forward'):
                    hook = module.register_forward_hook(get_attention_hook(f"vision_{name}"))
                    self.hooks.append(hook)
                    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def extract_attention_for_window(self, frames: List[Image.Image], 
                                    left_idx: int, right_idx: int) -> Dict:
        """
        Extract attention patterns for a window of frames
        Returns attention maps and frame features
        """
        self._current_window_frame_count = len(frames)
        try:
            self.register_attention_hooks()

            # Process frames through model to get attention
            with torch.no_grad():
                # Run the boundary detection which triggers attention capture
                result = self.segmenter.ask_boundary_native(frames, left_idx, right_idx)

            # Collect captured attention weights
            attention_data = {
                'boundary_result': result,
                'attention_maps': dict(self.attention_weights),
                'n_frames': len(frames),
                'left_idx': left_idx,
                'right_idx': right_idx
            }

        finally:
            self.clear_hooks()
            if hasattr(self, '_current_window_frame_count'):
                delattr(self, '_current_window_frame_count')

        return attention_data
    
    def compute_frame_attention_matrix(self, attention_maps: Dict, 
                                      n_frames: int) -> Optional[np.ndarray]:
        """
        Compute frame-to-frame attention matrix from raw attention maps
        Returns: (n_frames, n_frames) attention matrix
        """
        if not attention_maps:
            return None
            
        # Get the last layer's attention (usually most semantic)
        last_layer_key = None
        for key in sorted(attention_maps.keys(), reverse=True):
            if attention_maps[key] is not None:
                last_layer_key = key
                break
                
        if last_layer_key is None:
            return None
            
        attn = attention_maps[last_layer_key]
        
        # Handle different attention tensor shapes
        # Typical shape: (batch, heads, seq_len, seq_len)
        if attn.dim() == 4:
            # Average over heads
            attn = attn.mean(dim=1)  # (batch, seq_len, seq_len)
            
        if attn.dim() == 3:
            attn = attn[0]  # Take first batch item
            
        # Now attn should be (seq_len, seq_len)
        # We need to extract frame-specific attention
        
        # Assuming patches are arranged as [CLS, frame1_patches, frame2_patches, ...]
        # We need to segment based on number of patches per frame
        
        seq_len = attn.shape[0]
        if seq_len == n_frames:
            return attn.numpy() if hasattr(attn, 'numpy') else attn

        # Estimate patches per frame (assuming square grid)
        patches_per_frame = (seq_len - 1) // n_frames if seq_len > n_frames else 1
        
        if patches_per_frame < 1:
            return None
            
        # Create frame-to-frame attention by averaging patch attentions
        frame_attn = np.zeros((n_frames, n_frames))
        
        for i in range(n_frames):
            for j in range(n_frames):
                start_i = 1 + i * patches_per_frame  # Skip CLS token
                end_i = min(start_i + patches_per_frame, seq_len)
                start_j = 1 + j * patches_per_frame
                end_j = min(start_j + patches_per_frame, seq_len)
                
                if start_i < seq_len and start_j < seq_len:
                    patch_attn = attn[start_i:end_i, start_j:end_j]
                    frame_attn[i, j] = patch_attn.mean().item()
                    
        return frame_attn
    
    def visualize_attention_pattern(self, frames: List[Image.Image],
                                   left_idx: int, right_idx: int,
                                   output_path: str = "attention_analysis.png",
                                   frame_numbers: Optional[List[int]] = None):
        """
        Create comprehensive attention visualization for boundary detection
        """
        # Extract attention
        attn_data = self.extract_attention_for_window(frames, left_idx, right_idx)
        
        # Compute frame attention matrix
        frame_attn = self.compute_frame_attention_matrix(
            attn_data['attention_maps'], 
            attn_data['n_frames']
        )
        
        if frame_attn is None:
            print("[WARN] Could not extract frame attention matrix")
            return None

        if isinstance(frame_attn, torch.Tensor):
            frame_attn = frame_attn.detach().cpu().numpy()
            
        n_frames = len(frames)
        frame_labels = frame_numbers if frame_numbers else list(range(1, n_frames + 1))

        fig = plt.figure(figsize=(18, 10))
        outer = gridspec.GridSpec(2, 1, height_ratios=[1.1, 1.7], hspace=0.35)

        frame_grid = gridspec.GridSpecFromSubplotSpec(
            1, n_frames, subplot_spec=outer[0], wspace=0.05
        )

        for i in range(n_frames):
            ax = fig.add_subplot(frame_grid[0, i])
            ax.imshow(frames[i])
            ax.axis('off')
            label = frame_labels[i]
            if isinstance(label, float):
                label = f"{label:.2f}"
            title = f"Frame {label}"
            if i == left_idx:
                ax.set_title(f"{title}\n(Left)", color='blue', fontweight='bold')
            elif i == right_idx:
                ax.set_title(f"{title}\n(Right)", color='red', fontweight='bold')
            else:
                ax.set_title(title)

        summary_grid = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[1], width_ratios=[2.5, 1.0], wspace=0.3
        )

        ax_full = fig.add_subplot(summary_grid[0, 0])
        vmax = float(frame_attn.max()) if np.isfinite(frame_attn).all() else None
        sns.heatmap(
            frame_attn,
            cmap='YlOrRd',
            cbar=True,
            square=True,
            xticklabels=frame_labels,
            yticklabels=frame_labels,
            ax=ax_full,
            vmin=0,
            vmax=vmax if vmax and vmax > 0 else None
        )
        ax_full.set_title("Frame-to-Frame Attention")
        ax_full.set_xlabel("Target Frame")
        ax_full.set_ylabel("Source Frame")

        ax_contrast = fig.add_subplot(summary_grid[0, 1])
        
        # Compute contrastive scores
        # High score = strong within-group attention, weak across-group attention
        left_group = list(range(0, left_idx + 1))
        right_group = list(range(right_idx, n_frames))
        
        # Within-group attention
        left_within = frame_attn[np.ix_(left_group, left_group)].mean() if left_group else 0
        right_within = frame_attn[np.ix_(right_group, right_group)].mean() if right_group else 0
        
        # Cross-group attention  
        if left_group and right_group:
            left_to_right = frame_attn[np.ix_(left_group, right_group)].mean()
            right_to_left = frame_attn[np.ix_(right_group, left_group)].mean()
            cross_group = (left_to_right + right_to_left) / 2
        else:
            cross_group = 0
            
        # Contrastive score
        within_avg = (left_within + right_within) / 2 if (left_within + right_within) > 0 else 1
        contrastive_score = (within_avg - cross_group) / within_avg if within_avg > 0 else 0
        
        # Plot bars
        categories = ['Within\nLeft', 'Within\nRight', 'Cross\nGroups']
        values = [left_within, right_within, cross_group]
        colors = ['blue', 'red', 'gray']
        bars = ax_contrast.bar(categories, values, color=colors, alpha=0.7)
        ax_contrast.set_ylabel('Mean Attention')
        ax_contrast.set_title(f'Contrastive Analysis\nScore: {contrastive_score:.3f}')
        ax_contrast.set_ylim([0, max(values) * 1.2 if values else 1])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax_contrast.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom')
        
        # Add result info
        result = attn_data['boundary_result']
        fig.suptitle(
            f"Attention Analysis - Boundary: {result['boundary']} "
            f"(Confidence: {result.get('confidence', 0):.3f})",
            fontsize=14, fontweight='bold'
        )
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Saved attention visualization to {output_path}")
        
        return {
            'frame_attention': frame_attn,
            'contrastive_score': contrastive_score,
            'within_left': left_within,
            'within_right': right_within,
            'cross_group': cross_group,
            'boundary_result': result
        }
    
    def analyze_video_attention_patterns(self, video_path: str,
                                        sample_fps: float = 2.0,
                                        window_size: int = 8,
                                        n_samples: int = 5,
                                        output_dir: str = "attention_analysis"):
        """
        Analyze attention patterns at multiple points in the video
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample frames
        frames, frame_indices, video_fps, _ = self.segmenter.sample_frames(
            video_path, sample_fps
        )
        
        n_frames = len(frames)
        if n_frames < window_size:
            print(f"[ERROR] Video too short for analysis")
            return
            
        # Sample windows throughout the video
        sample_positions = np.linspace(0, n_frames - window_size, n_samples, dtype=int)
        
        results = []
        for i, start_idx in enumerate(sample_positions):
            window_frames = frames[start_idx:start_idx + window_size]
            
            # Center frames for boundary
            right_idx = window_size // 2
            left_idx = right_idx - 1
            window_frame_indices = frame_indices[start_idx:start_idx + window_size]
            
            print(f"\n[INFO] Analyzing window {i+1}/{n_samples} (frames {start_idx}-{start_idx+window_size-1})")
            
            # Visualize attention
            output_path = os.path.join(output_dir, f"attention_window_{i+1}.png")
            result = self.visualize_attention_pattern(
                window_frames,
                left_idx,
                right_idx,
                output_path=output_path,
                frame_numbers=window_frame_indices
            )
            
            if result:
                result['window_start'] = start_idx
                result['time_seconds'] = frame_indices[start_idx] / video_fps
                results.append(result)
                
        # Summary plot
        if results:
            self._create_summary_plot(results, os.path.join(output_dir, "attention_summary.png"))
            
        return results
    
    def _create_summary_plot(self, results: List[Dict], output_path: str):
        """Create a summary plot of contrastive scores across the video"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        times = [r['time_seconds'] for r in results]
        contrastive_scores = [r['contrastive_score'] for r in results]
        boundaries = [r['boundary_result']['boundary'] for r in results]
        confidences = [r['boundary_result'].get('confidence', 0) for r in results]
        
        # Plot 1: Contrastive scores
        ax1 = axes[0]
        colors = ['red' if b else 'blue' for b in boundaries]
        bars = ax1.bar(range(len(times)), contrastive_scores, color=colors, alpha=0.7)
        ax1.set_xlabel('Window Sample')
        ax1.set_ylabel('Contrastive Score')
        ax1.set_title('Attention Contrastive Scores Across Video')
        ax1.set_xticks(range(len(times)))
        ax1.set_xticklabels([f"{t:.1f}s" for t in times], rotation=45)
        
        # Add boundary labels
        for i, (bar, boundary) in enumerate(zip(bars, boundaries)):
            label = "B" if boundary else "N"
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    label, ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Attention components
        ax2 = axes[1]
        within_left = [r['within_left'] for r in results]
        within_right = [r['within_right'] for r in results]
        cross_group = [r['cross_group'] for r in results]
        
        x = np.arange(len(times))
        width = 0.25
        ax2.bar(x - width, within_left, width, label='Within Left', alpha=0.7)
        ax2.bar(x, within_right, width, label='Within Right', alpha=0.7)
        ax2.bar(x + width, cross_group, width, label='Cross Group', alpha=0.7)
        
        ax2.set_xlabel('Window Sample')
        ax2.set_ylabel('Mean Attention')
        ax2.set_title('Attention Components')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{t:.1f}s" for t in times], rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Saved summary plot to {output_path}")

# Usage example
def analyze_attention_for_boundaries(video_path: str, 
                                    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                                    sample_fps: float = 2.0):
    """
    Main function to analyze attention patterns for boundary detection
    """
    from qwen_temporal_segmentation_fixed import QwenTemporalSegmenterFixed
    
    # Initialize segmenter
    segmenter = QwenTemporalSegmenterFixed(
        model_name=model_name,
        response_mode="binary"  # Binary mode as requested
    )
    
    # Create attention analyzer
    analyzer = QwenAttentionAnalyzer(segmenter)
    
    # Analyze attention patterns at multiple windows
    results = analyzer.analyze_video_attention_patterns(
        video_path=video_path,
        sample_fps=sample_fps,
        window_size=8,
        n_samples=10,  # Sample 10 windows across video
        output_dir="attention_analysis"
    )
    
    # Print summary statistics
    if results:
        boundaries_found = sum(1 for r in results if r['boundary_result']['boundary'])
        avg_contrastive = np.mean([r['contrastive_score'] for r in results])
        
        print("\n" + "="*50)
        print("ATTENTION ANALYSIS SUMMARY")
        print("="*50)
        print(f"Windows analyzed: {len(results)}")
        print(f"Boundaries detected: {boundaries_found}/{len(results)}")
        print(f"Average contrastive score: {avg_contrastive:.3f}")
        
        # Compare contrastive scores for boundaries vs non-boundaries
        boundary_scores = [r['contrastive_score'] for r in results 
                          if r['boundary_result']['boundary']]
        non_boundary_scores = [r['contrastive_score'] for r in results 
                              if not r['boundary_result']['boundary']]
        
        if boundary_scores:
            print(f"Avg contrastive (boundaries): {np.mean(boundary_scores):.3f}")
        if non_boundary_scores:
            print(f"Avg contrastive (non-boundaries): {np.mean(non_boundary_scores):.3f}")
    
    return results