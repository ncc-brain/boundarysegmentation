#!/usr/bin/env python3
"""
report_predictions_vs_gt.py
Print predictions and corresponding ground-truths for a handful of videos.

- Reads prediction_summary.json from the prediction directory to get processed IDs
- Loads each {vid_id}.pkl (format: {'bdy_idx_list_smt': [...]})
- Loads GT pickle and prints GT boundary frame indices per rater
"""

import os
import json
import pickle
import argparse
from typing import List, Dict


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def format_list(vals: List[int], limit: int = 40) -> str:
    if len(vals) <= limit:
        return str(vals)
    head = vals[:limit]
    return f"{head} ... (+{len(vals) - limit} more)"


def compute_f1(preds: List[int], gts: List[int], tolerance: int) -> float:
    """
    Greedy one-to-one matching within +/- tolerance frames.
    Returns F1 score.
    """
    if not preds and not gts:
        return 1.0
    if not preds or not gts:
        return 0.0

    preds_sorted = sorted(preds)
    gts_sorted = sorted(gts)

    i = 0
    j = 0
    matches = 0
    while i < len(preds_sorted) and j < len(gts_sorted):
        p = preds_sorted[i]
        g = gts_sorted[j]
        if abs(p - g) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif p < g - tolerance:
            i += 1
        else:
            j += 1

    precision = matches / len(preds_sorted) if preds_sorted else 0.0
    recall = matches / len(gts_sorted) if gts_sorted else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_ratio_list(min_ratio: float, max_ratio: float, step: float) -> List[float]:
    """Create an inclusive list of ratios from min to max with given step.
    Ratios are rounded to 3 decimals to avoid FP artifacts in printing/keys.
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if max_ratio < min_ratio:
        raise ValueError("max_ratio must be >= min_ratio")
    count = int(round((max_ratio - min_ratio) / step)) + 1
    return [round(min_ratio + i * step, 3) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description='Report predictions vs GT for a few videos')
    parser.add_argument('--gt-path', type=str,default="boundry_segmentation/k400_mr345_val_min_change_duration0.3.pkl", help='Path to GT pickle file')
    parser.add_argument('--pred-dir', type=str, required=True, help='Directory with prediction .pkl files and prediction_summary.json')
    parser.add_argument('--limit', type=int, default=50000, help='How many videos to report')
    parser.add_argument('--downsample', type=int, default=1, help='Downsampling factor used in predictions')
    parser.add_argument('--show', type=int, default=40, help='Max number of indices to print per list')
    parser.add_argument('--tolerance', type=int, default=None, help='[Deprecated] Absolute matching tolerance in frames; if set, used for a single-F1 print alongside relative sweep')
    parser.add_argument('--rel-tol-min', type=float, default=0.05, help='Minimum relative tolerance (fraction of video length)')
    parser.add_argument('--rel-tol-max', type=float, default=0.5, help='Maximum relative tolerance (fraction of video length)')
    parser.add_argument('--rel-tol-step', type=float, default=0.05, help='Step for relative tolerance sweep')
    parser.add_argument('--ignore-summary', action='store_true', help='Ignore prediction_summary.json and scan directory for .pkl files')
    args = parser.parse_args()

    summary_path = os.path.join(args.pred_dir, 'prediction_summary.json')
    processed_ids = []
    
    if not args.ignore_summary and os.path.exists(summary_path):
        summary = load_json(summary_path)
        processed_ids = summary.get('videos_processed', [])
    
    if not processed_ids:
        if args.ignore_summary:
            print('[INFO] Ignoring prediction_summary.json (--ignore-summary flag set); scanning directory for .pkl files...')
        else:
            print('[WARN] No prediction_summary.json or videos_processed list; scanning directory for .pkl files...')
        processed_ids = [os.path.splitext(f)[0] for f in os.listdir(args.pred_dir) if f.endswith('.pkl')]
        processed_ids.sort()

    gt_dict = load_pickle(args.gt_path)

    count = 0
    print(f"[INFO] Reporting up to {args.limit} videos")
    # For overall aggregation across videos: tolerance_ratio -> list of best F1
    overall_best_f1_by_ratio: Dict[float, List[float]] = {}
    best_f1s_single_abs: List[float] = []
    tolerance_ratios = build_ratio_list(args.rel_tol_min, args.rel_tol_max, args.rel_tol_step)
    print(f"[INFO] Relative tolerance sweep: {tolerance_ratios}")

    best_f1s: List[float] = []  # kept for backward-compatible average of per-video best (from absolute tol if provided)
    for vid_id in processed_ids:
        if count >= args.limit:
            break

        pred_path = os.path.join(args.pred_dir, f"{vid_id}.pkl")
        if not os.path.exists(pred_path):
            continue

        if vid_id not in gt_dict:
            print(f"\n[VID {vid_id}] (no GT found)")
            pred = load_pickle(pred_path)
            preds_ds = list(pred.get('bdy_idx_list_smt', []))
            print(f"  preds (downsampled idx, n={len(preds_ds)}): {format_list(preds_ds, args.show)}")
            count += 1
            continue

        vid_info = gt_dict[vid_id]
        fps = vid_info.get('fps')
        num_frames = vid_info.get('num_frames')

        pred = load_pickle(pred_path)
        preds_ds = list(pred.get('bdy_idx_list_smt', []))
        preds_orig = [p * args.downsample for p in preds_ds]

        print(f"\n[VID {vid_id}] fps={fps}, frames={num_frames}")
        print(f"  preds (downsampled idx, n={len(preds_ds)}): {format_list(preds_ds, args.show)}")
        print(f"  preds (orig frames, n={len(preds_orig)}): {format_list(preds_orig, args.show)}")

        gt_all_raters: List[List[int]] = vid_info.get('substages_myframeidx', [])
        if not gt_all_raters:
            print("  [WARN] No GT substages_myframeidx present")
        else:
            for r_idx, gt_frames in enumerate(gt_all_raters):
                print(f"  gt[rater {r_idx}] (n={len(gt_frames)}): {format_list(gt_frames, args.show)}")

            # Optional absolute tolerance single-F1 report if provided
            if args.tolerance is not None:
                best_f1_abs = 0.0
                best_rater_abs = -1
                for r_idx, gt_frames in enumerate(gt_all_raters):
                    f1 = compute_f1(preds_orig, list(gt_frames), args.tolerance)
                    if f1 > best_f1_abs:
                        best_f1_abs = f1
                        best_rater_abs = r_idx
                print(f"  [ABS] best_f1={best_f1_abs:.3f} (tol_frames={args.tolerance}, rater {best_rater_abs})")
                best_f1s_single_abs.append(best_f1_abs)

            # Relative tolerance sweep
            if num_frames is None:
                print("  [WARN] num_frames is None; cannot compute relative tolerances")
            else:
                print("  [REL] best F1 by tolerance ratio:")
                for ratio in tolerance_ratios:
                    tol_frames = int(round(ratio * num_frames))
                    best_f1_for_ratio = 0.0
                    best_rater_idx_for_ratio = -1
                    for r_idx, gt_frames in enumerate(gt_all_raters):
                        f1 = compute_f1(preds_orig, list(gt_frames), tol_frames)
                        if f1 > best_f1_for_ratio:
                            best_f1_for_ratio = f1
                            best_rater_idx_for_ratio = r_idx
                    print(f"    ratio={ratio:.3f} (tol_frames={tol_frames}): best_f1={best_f1_for_ratio:.3f} (rater {best_rater_idx_for_ratio})")
                    overall_best_f1_by_ratio.setdefault(ratio, []).append(best_f1_for_ratio)

        count += 1

    if count == 0:
        print('[INFO] Nothing reported - no matching prediction files found')
    # Overall aggregates
    if overall_best_f1_by_ratio:
        print("\n[INFO] Overall average of per-video best F1 by relative tolerance:")
        for ratio in sorted(overall_best_f1_by_ratio.keys()):
            vals = overall_best_f1_by_ratio[ratio]
            avg_val = sum(vals) / len(vals)
            print(f"  ratio={ratio:.3f}: avg_best_f1={avg_val:.3f} over {len(vals)} videos")
    if best_f1s_single_abs:
        avg_best_f1_abs = sum(best_f1s_single_abs) / len(best_f1s_single_abs)
        print(f"\n[INFO] Average of per-video best F1 (absolute tol) over {len(best_f1s_single_abs)} videos: {avg_best_f1_abs:.3f}")


if __name__ == '__main__':
    main()