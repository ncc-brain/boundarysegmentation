#!/usr/bin/env python3
"""
run_uboco_on_gebd_eval.py
Batch runner to apply UBoCo on the GEBD evaluation set efficiently.

Features:
- Robust video path resolution via recursive indexing
- Shared ResNet-50 backbone across videos (reset trainable encoder per video)
- Saves predictions in GEBD eval format (.pkl with 'bdy_idx_list_smt')
- Sharding, filtering, skip-existing, and detailed summary
"""

import os
import re
import pickle
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm
import torch

# Reuse UBoCo components
from uboco_gebd import VideoFeatureExtractor, train_uboco


VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}


def _norm_key(name: str):
    """Create normalization variants for indexing and lookup."""
    if not name:
        return []
    base = os.path.basename(str(name))
    lower = base.lower()
    stem, _ = os.path.splitext(lower)
    prefix = stem.split('_')[0]
    slug = re.sub(r"[^a-z0-9]", "", stem)
    slug_prefix = re.sub(r"[^a-z0-9]", "", prefix)
    keys = {lower, stem, slug, prefix, slug_prefix}
    return list(keys)


def build_video_index(root_dir: str, allowed_exts=VIDEO_EXTS) -> Dict[str, list]:
    """Recursively index videos under root_dir by several normalized keys."""
    index: Dict[str, list] = {}
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
    video_filename = vid_info.get('path_video')
    if video_filename:
        direct = os.path.join(video_dir, video_filename)
        if os.path.exists(direct):
            return direct
        base = os.path.basename(video_filename)
        for key in _norm_key(base):
            if key in index:
                return _pick_best(index[key])

    for ext in allowed_exts:
        for key in _norm_key(f"{vid_id}{ext}"):
            if key in index:
                return _pick_best(index[key])

    for key in _norm_key(vid_id):
        if key in index:
            return _pick_best(index[key])

    return None


def load_gt_dict(gt_path):
    with open(gt_path, 'rb') as f:
        gt_dict = pickle.load(f)
    print(f"[INFO] Loaded {len(gt_dict)} videos from GT file")
    return gt_dict


def convert_timestamps_to_frames(boundary_times, fps, downsample=3):
    """Convert boundary timestamps to downsampled frame indices (GEBD uses 3x)."""
    frame_indices = []
    for t in boundary_times:
        frame_idx = int(np.round(t * fps))
        ds_idx = frame_idx // downsample
        frame_indices.append(ds_idx)
    return frame_indices


def reset_encoder_weights(model: VideoFeatureExtractor):
    """Reset trainable encoder parameters in-place while keeping the backbone in memory."""
    for module in model.encoder.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()


def run_uboco_on_video(feature_extractor: VideoFeatureExtractor, video_path: str, vid_info: dict, args) -> list:
    """Train+infer UBoCo on a single video and return downsampled frame indices."""
    # Per-video output directory to avoid collisions
    per_video_out = os.path.join(args.uboco_output_root, str(vid_info.get('vid', 'video')))
    os.makedirs(per_video_out, exist_ok=True)

    results, video_fps = train_uboco(
        feature_extractor=feature_extractor,
        video_path=video_path,
        fps_sample=args.sample_fps,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rtp_top_k=args.rtp_top_k,
        boco_gap=args.boco_gap,
        start_time=None,
        end_time=None,
        output_dir=per_video_out,
        boundary_method=args.boundary_method,
        rtp_kernel_size=args.rtp_kernel_size,
        rtp_min_length=args.rtp_min_length,
        rtp_threshold_diff=args.rtp_threshold_diff,
        rtp_max_depth=args.rtp_max_depth,
        rtp_max_boundaries=args.rtp_max_boundaries,
        peaks_distance=args.peaks_distance,
        peaks_prominence=args.peaks_prominence,
        peaks_max_boundaries=args.peaks_max_boundaries,
    )

    # Convert using GT fps for evaluation compatibility
    gt_fps = vid_info['fps']
    ds_frame_indices = convert_timestamps_to_frames(
        results.get('boundary_times', []),
        fps=gt_fps,
        downsample=args.downsample,
    )

    # Free GPU memory between videos if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ds_frame_indices


def save_predictions(predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for vid_id, frame_indices in predictions.items():
        output_path = os.path.join(output_dir, f"{vid_id}.pkl")
        payload = {'bdy_idx_list_smt': frame_indices}
        with open(output_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved {len(predictions)} prediction files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run UBoCo on GEBD evaluation set")

    # IO
    parser.add_argument("--gt-path", type=str, default="./k400_mr345_val_min_change_duration0.3.pkl", help="Path to GT pickle file")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, default="./outputs/exp_k400/detect_seg", help="Output directory for predictions (.pkl)")
    parser.add_argument("--uboco-output-root", type=str, default="./outputs_uboco_eval", help="Root directory to store per-video UBoCo artifacts")

    # Model/training params (forwarded to UBoCo)
    parser.add_argument("--sample-fps", type=float, default=25, help="Frame sampling rate for UBoCo")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--encoder-dim", type=int, default=512, help="Encoder output dimension")
    parser.add_argument("--n-epochs", type=int, default=5, help="Number of training epochs per video")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for encoder")
    parser.add_argument("--rtp-top-k", type=float, default=0.2, help="RTP top-k percentage for sampling")
    parser.add_argument("--boco-gap", type=int, default=20, help="BoCo local similarity gap")

    # Boundary method controls
    parser.add_argument("--boundary-method", type=str, default='rtp', choices=['rtp', 'peaks'], help="Boundary detection method")
    parser.add_argument("--rtp-kernel-size", type=int, default=5, help="RTP kernel size")
    parser.add_argument("--rtp-min-length", type=int, default=50, help="RTP minimum segment length to recurse")
    parser.add_argument("--rtp-threshold-diff", type=float, default=0.3, help="RTP min (max-mean) score to recurse")
    parser.add_argument("--rtp-max-depth", type=int, default=None, help="RTP maximum recursion depth")
    parser.add_argument("--rtp-max-boundaries", type=int, default=None, help="RTP global maximum number of boundaries")
    parser.add_argument("--peaks-distance", type=int, default=20, help="Peaks: minimum frames between boundaries")
    parser.add_argument("--peaks-prominence", type=float, default=0.5, help="Peaks: required prominence over baseline")
    parser.add_argument("--peaks-max-boundaries", type=int, default=None, help="Peaks: cap number of boundaries")

    # Evaluation/export options
    parser.add_argument("--downsample", type=int, default=3, help="Downsampling factor for frame indices (default 3 for GEBD)")

    # Processing options
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process (for testing)")
    parser.add_argument("--min-f1-filter", type=float, default=0.3, help="Filter GT videos with f1_consis_avg below this threshold")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos that already have predictions")
    parser.add_argument("--only-video", type=str, default=None, help="Process only this specific vid_id (after filtering)")

    # Sharding
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards to split workload across")
    parser.add_argument("--shard-id", type=int, default=0, help="This process shard id (0-indexed)")

    args = parser.parse_args()

    # Load GT dict and filter
    gt_dict = load_gt_dict(args.gt_path)
    filtered_vids = {}
    for vid_id, vid_info in gt_dict.items():
        if vid_info.get('f1_consis_avg', 0) >= args.min_f1_filter:
            filtered_vids[vid_id] = vid_info

    print(f"[INFO] After f1 filtering: {len(filtered_vids)} videos")

    if args.max_videos:
        video_ids = list(filtered_vids.keys())[: args.max_videos]
        filtered_vids = {vid: filtered_vids[vid] for vid in video_ids}
        print(f"[INFO] Limited to {args.max_videos} videos for testing")

    if args.only_video is not None:
        if args.only_video in filtered_vids:
            filtered_vids = {args.only_video: filtered_vids[args.only_video]}
            print(f"[INFO] Restricting run to only_video={args.only_video}")
        else:
            print(f"[WARN] only_video={args.only_video} not present after filtering; nothing to do.")
            filtered_vids = {}

    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError("shard-id must be in [0, num-shards-1]")
        all_ids = sorted(filtered_vids.keys())
        shard_ids = all_ids[args.shard_id :: args.num_shards]
        filtered_vids = {vid: filtered_vids[vid] for vid in shard_ids}
        print(f"[INFO] Sharding enabled: shard {args.shard_id}/{args.num_shards} -> {len(filtered_vids)} videos")

    # Initialize shared feature extractor once (reuse backbone, reset encoder per video)
    print(f"[INFO] Initializing UBoCo feature extractor (encoder_dim={args.encoder_dim})")
    feature_extractor = VideoFeatureExtractor(encoder_dim=args.encoder_dim)

    # Build video index once for robust lookup
    print(f"[INFO] Building video index for {args.video_dir} (recursive)...")
    video_index = build_video_index(args.video_dir)
    print(f"[INFO] Indexed {sum(len(v) for v in video_index.values())} files (unique keys: {len(video_index)})")

    predictions = {}
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.uboco_output_root, exist_ok=True)

    for vid_id, vid_info in tqdm(filtered_vids.items(), desc="Processing videos"):
        out_path = os.path.join(args.output_dir, f"{vid_id}.pkl")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {vid_id} already processed")
            continue

        video_path = resolve_video_path(vid_id, vid_info, args.video_dir, video_index)
        if not video_path:
            print(f"[WARN] Video not found for vid_id={vid_id}. Tried GT path and recursive search.")
            continue

        print(f"\n[INFO] Processing {vid_id}")
        print(f"       Video: {video_path}")
        print(f"       FPS: {vid_info['fps']}, Frames: {vid_info['num_frames']}")

        # Reset encoder so each video is trained independently
        reset_encoder_weights(feature_extractor)

        try:
            ds_indices = run_uboco_on_video(
                feature_extractor=feature_extractor,
                video_path=video_path,
                vid_info={**vid_info, 'vid': vid_id},
                args=args,
            )
        except Exception as e:
            print(f"[ERROR] Failed on video {vid_id}: {e}")
            import traceback
            traceback.print_exc()
            ds_indices = []

        predictions[vid_id] = ds_indices
        print(f"       Found {len(ds_indices)} boundaries (downsampled)")

        # Save per video immediately
        try:
            pred_data = {'bdy_idx_list_smt': ds_indices}
            with open(out_path, 'wb') as f:
                pickle.dump(pred_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"       Saved prediction -> {out_path}")
        except Exception as e:
            print(f"[WARN] Failed to save {vid_id} prediction: {e}")

    # Save aggregate predictions and summary
    print("\n[INFO] Saving predictions...")
    save_predictions(predictions, args.output_dir)

    summary = {
        "runner": "uboco",
        "total_videos": len(predictions),
        "videos_processed": list(predictions.keys()),
        "parameters": {
            "sample_fps": args.sample_fps,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "encoder_dim": args.encoder_dim,
            "lr": args.lr,
            "rtp_top_k": args.rtp_top_k,
            "boco_gap": args.boco_gap,
            "boundary_method": args.boundary_method,
            "rtp_kernel_size": args.rtp_kernel_size,
            "rtp_min_length": args.rtp_min_length,
            "rtp_threshold_diff": args.rtp_threshold_diff,
            "rtp_max_depth": args.rtp_max_depth,
            "rtp_max_boundaries": args.rtp_max_boundaries,
            "peaks_distance": args.peaks_distance,
            "peaks_prominence": args.peaks_prominence,
            "peaks_max_boundaries": args.peaks_max_boundaries,
            "downsample": args.downsample,
        },
    }

    summary_path = os.path.join(args.output_dir, "prediction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Processed {len(predictions)} videos")
    print(f"[INFO] Predictions saved to: {args.output_dir}/")
    print(f"[INFO] Summary saved to: {summary_path}")
    print("\nExample eval invocation:")
    print("python eval_script.py")


if __name__ == "__main__":
    main()


