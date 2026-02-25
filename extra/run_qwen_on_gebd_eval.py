#!/usr/bin/env python3
"""
run_qwen_on_gebd_eval.py
Run Qwen predictor on GEBD evaluation set and save in correct format
"""

import os
import pickle
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re

# Import your Qwen segmenter (use the updated file with GEBD-aligned prompt)
from qwen_temporal_segmentation_gebd import QwenTemporalSegmenterFixed


VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}


def _norm_key(name):
    """Create normalization variants for indexing and lookup."""
    if not name:
        return []
    base = os.path.basename(str(name))
    lower = base.lower()
    stem, ext = os.path.splitext(lower)
    # also consider prefix before first underscore (strip clip timecodes)
    prefix = stem.split('_')[0]
    # slug: alnum only
    slug = re.sub(r"[^a-z0-9]", "", stem)
    slug_prefix = re.sub(r"[^a-z0-9]", "", prefix)
    keys = {lower, stem, slug, prefix, slug_prefix}
    return list(keys)


def build_video_index(root_dir, allowed_exts=VIDEO_EXTS):
    """Recursively index videos under root_dir by several normalized keys.

    Returns: dict key -> list(paths)
    """
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in allowed_exts:
                continue
            path = os.path.join(dirpath, fname)
            for key in _norm_key(fname):
                index.setdefault(key, []).append(path)
    return index


def _pick_best(paths):
    """Pick a stable best path among candidates: prefer .mp4 and shorter path."""
    if not paths:
        return None
    def score(p):
        ext = os.path.splitext(p)[1].lower()
        ext_score = 0 if ext == '.mp4' else 1
        return (ext_score, len(p))
    return sorted(paths, key=score)[0]


def resolve_video_path(vid_id, vid_info, video_dir, index, allowed_exts=VIDEO_EXTS):
    """Resolve a video file path using GT hints and recursive index."""
    # 1) Direct path from GT (relative to video_dir)
    video_filename = vid_info.get('path_video')
    if video_filename:
        direct = os.path.join(video_dir, video_filename)
        if os.path.exists(direct):
            return direct
        # try basename only
        base = os.path.basename(video_filename)
        for key in _norm_key(base):
            if key in index:
                return _pick_best(index[key])

    # 2) Try vid_id.mp4 (and other exts) under index
    for ext in allowed_exts:
        for key in _norm_key(f"{vid_id}{ext}"):
            if key in index:
                return _pick_best(index[key])

    # 3) Try vid_id raw keys
    for key in _norm_key(vid_id):
        if key in index:
            return _pick_best(index[key])

    return None


def load_gt_dict(gt_path):
    """Load the GT dictionary to get list of videos"""
    with open(gt_path, 'rb') as f:
        gt_dict = pickle.load(f)
    print(f"[INFO] Loaded {len(gt_dict)} videos from GT file")
    return gt_dict


def convert_timestamps_to_frames(boundary_times, fps, downsample=3):
    """
    Convert boundary timestamps to downsampled frame indices
    
    Args:
        boundary_times: List of boundary times in seconds
        fps: Video FPS
        downsample: Downsampling factor (default 3 for GEBD)
    
    Returns:
        List of frame indices (already accounting for downsampling)
    """
    frame_indices = []
    for t in boundary_times:
        # Convert time to original frame index
        frame_idx = int(np.round(t * fps))
        # Convert to downsampled index
        ds_idx = frame_idx // downsample
        frame_indices.append(ds_idx)
    
    return frame_indices


def run_prediction_on_video(segmenter, video_path, video_info, args):
    """
    Run Qwen predictor on a single video
    
    Args:
        segmenter: QwenTemporalSegmenterFixed instance
        video_path: Path to video file
        video_info: Dictionary with video metadata from GT
        args: Command line arguments
    
    Returns:
        List of boundary frame indices (downsampled)
    """
    try:
        # Run sliding window detection
        results = segmenter.sliding_window_detection(
            video_path=video_path,
            sample_fps=args.sample_fps,
            start_time=None,  # Use full video
            end_time=None,
            window_size=args.window_size,
            stride=args.stride,
            vote_threshold=args.vote_threshold,
            merge_distance_sec=args.merge_distance,
            output_dir=args.temp_output_dir,
            debug=getattr(args, 'debug', False),
            debug_dir=getattr(args, 'debug_dir', None),
            debug_tag=None,
            debug_save_images=True
        )
        
        # Convert timestamps to downsampled frame indices
        boundary_times = results.get("boundary_times", [])
        fps = video_info['fps']
        frame_indices = convert_timestamps_to_frames(
            boundary_times, 
            fps, 
            downsample=args.downsample
        )
        
        return frame_indices
        
    except Exception as e:
        print(f"[ERROR] Failed on video {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_predictions(predictions, output_dir):
    """
    Save predictions in GEBD eval format
    
    Each video gets saved as: {output_dir}/{vid_id}.pkl
    Format: {'bdy_idx_list_smt': [frame_idx1, frame_idx2, ...]}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for vid_id, frame_indices in predictions.items():
        output_path = os.path.join(output_dir, f"{vid_id}.pkl")
        
        # Save in the format expected by eval script
        pred_data = {
            'bdy_idx_list_smt': frame_indices
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(pred_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"[INFO] Saved {len(predictions)} prediction files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen predictor on GEBD evaluation set"
    )
    
    # Input/output paths
    parser.add_argument(
        "--gt-path", 
        type=str, 
        default="./k400_mr345_val_min_change_duration0.3.pkl",
        help="Path to GT pickle file"
    )
    parser.add_argument(
        "--video-dir", 
        type=str, 
        required=True,
        help="Directory containing video files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs/exp_k400/detect_seg",
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--temp-output-dir",
        type=str,
        default="./temp_outputs",
        help="Temporary output directory for Qwen results"
    )
    
    # Model settings
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen model name"
    )
    parser.add_argument(
        "--prompt-type", 
        type=str, 
        choices=["small", "narrative", "context", "semantic", "gebd_aligned", "aggressive"], 
        default="gebd_aligned",
        help="Prompt style (add 'aggressive' to encourage more boundary hypotheses)"
    )
    
    # Detection parameters
    parser.add_argument(
        "--sample-fps", 
        type=float, 
        default=25,
        help="Frame sampling rate"
    )
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=8,
        help="Window size for detection"
    )
    parser.add_argument(
        "--stride", 
        type=int, 
        default=1,
        help="Stride for sliding window"
    )
    parser.add_argument(
        "--vote-threshold", 
        type=int, 
        default=1,
        help="Minimum votes to confirm boundary"
    )
    parser.add_argument(
        "--merge-distance", 
        type=float, 
        default=0.5,
        help="Merge boundaries within this many seconds"
    )
    parser.add_argument(
        "--downsample", 
        type=int, 
        default=3,
        help="Downsampling factor for frame indices (default 3 for GEBD)"
    )
    
    # Processing options
    parser.add_argument(
        "--max-videos", 
        type=int, 
        default=None,
        help="Maximum number of videos to process (for testing)"
    )
    parser.add_argument(
        "--min-f1-filter",
        type=float,
        default=0.3,
        help="Filter videos with f1_consis_avg below this threshold (default 0.3, same as eval script)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have predictions"
    )
    # Debug/testing options
    parser.add_argument(
        "--only-video",
        type=str,
        default=None,
        help="Process only this specific vid_id (after filtering)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save per-window frames and outputs"
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Directory to save debug dumps (defaults to <temp-output-dir>/debug_windows)"
    )
    # Sharding options
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards to split the workload across"
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This process's shard id (0-indexed)"
    )
    
    args = parser.parse_args()
    
    # Load GT dictionary
    gt_dict = load_gt_dict(args.gt_path)
    
    # Filter by f1_consis_avg
    filtered_vids = {}
    for vid_id, vid_info in gt_dict.items():
        if vid_info.get('f1_consis_avg', 0) >= args.min_f1_filter:
            filtered_vids[vid_id] = vid_info
    
    print(f"[INFO] After f1 filtering: {len(filtered_vids)} videos")
    
    # Limit number of videos if specified
    if args.max_videos:
        video_ids = list(filtered_vids.keys())[:args.max_videos]
        filtered_vids = {vid: filtered_vids[vid] for vid in video_ids}
        print(f"[INFO] Limited to {args.max_videos} videos for testing")

    # Restrict to a single video if requested
    if args.only_video is not None:
        if args.only_video in filtered_vids:
            filtered_vids = {args.only_video: filtered_vids[args.only_video]}
            print(f"[INFO] Restricting run to only_video={args.only_video}")
        else:
            print(f"[WARN] only_video={args.only_video} not present after filtering; nothing to do.")
            filtered_vids = {}

    # Apply deterministic sharding if requested
    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError("shard-id must be in [0, num-shards-1]")
        all_ids = sorted(filtered_vids.keys())
        shard_ids = all_ids[args.shard_id::args.num_shards]
        filtered_vids = {vid: filtered_vids[vid] for vid in shard_ids}
        print(f"[INFO] Sharding enabled: shard {args.shard_id}/{args.num_shards} -> {len(filtered_vids)} videos")
    
    # Initialize Qwen segmenter
    print(f"[INFO] Initializing Qwen model: {args.model}")
    segmenter = QwenTemporalSegmenterFixed(
        model_name=args.model,
        prompt_type=args.prompt_type
    )
    
    # Build video index once for robust lookup
    print(f"[INFO] Building video index for {args.video_dir} (recursive)...")
    video_index = build_video_index(args.video_dir)
    print(f"[INFO] Indexed {sum(len(v) for v in video_index.values())} files (unique keys: {len(video_index)})")

    # Process each video
    predictions = {}
    os.makedirs(args.temp_output_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    for vid_id, vid_info in tqdm(filtered_vids.items(), desc="Processing videos"):
        # Check if already processed
        output_path = os.path.join(args.output_dir, f"{vid_id}.pkl")
        if args.skip_existing and os.path.exists(output_path):
            print(f"[SKIP] {vid_id} already processed")
            continue
        
        # Resolve video path robustly
        video_path = resolve_video_path(vid_id, vid_info, args.video_dir, video_index)
        if not video_path:
            print(f"[WARN] Video not found for vid_id={vid_id}. Tried GT path and recursive search.")
            continue
        
        print(f"\n[INFO] Processing {vid_id}")
        print(f"       Video: {video_path}")
        print(f"       FPS: {vid_info['fps']}, Frames: {vid_info['num_frames']}")
        
        # Run prediction
        frame_indices = run_prediction_on_video(
            segmenter, 
            video_path, 
            vid_info, 
            args
        )
        
        predictions[vid_id] = frame_indices
        print(f"       Found {len(frame_indices)} boundaries")

        # Save per-video prediction immediately
        try:
            output_path = os.path.join(args.output_dir, f"{vid_id}.pkl")
            pred_data = { 'bdy_idx_list_smt': frame_indices }
            with open(output_path, 'wb') as f:
                pickle.dump(pred_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"       Saved prediction -> {output_path}")
        except Exception as e:
            print(f"[WARN] Failed to save {vid_id} prediction: {e}")
    
    # Save all predictions
    print("\n[INFO] Saving predictions...")
    save_predictions(predictions, args.output_dir)
    
    # Save summary
    summary = {
        "model": args.model,
        "prompt_type": args.prompt_type,
        "total_videos": len(predictions),
        "videos_processed": list(predictions.keys()),
        "parameters": {
            "sample_fps": args.sample_fps,
            "window_size": args.window_size,
            "stride": args.stride,
            "vote_threshold": args.vote_threshold,
            "merge_distance": args.merge_distance,
            "downsample": args.downsample
        }
    }
    
    summary_path = os.path.join(args.output_dir, "prediction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[DONE] Processed {len(predictions)} videos")
    print(f"[INFO] Predictions saved to: {args.output_dir}/")
    print(f"[INFO] Summary saved to: {summary_path}")
    print(f"\nTo run evaluation:")
    print(f"python eval_script.py")


if __name__ == "__main__":
    main()