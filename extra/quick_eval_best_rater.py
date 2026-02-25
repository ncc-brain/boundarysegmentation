#!/usr/bin/env python3
"""
quick_eval_best_rater.py

Intermediate evaluation: for each video with predictions, compare against all
available GT raters, pick the rater yielding the highest F1 at a given
tolerance (as a fraction of video length, same as the official eval), then
report per-video and overall metrics.
"""

import os
import argparse
import pickle
import numpy as np
from typing import List, Tuple, Dict


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def match_and_score(pred_frames: List[int], gt_frames: List[int], tol_frames: int) -> Tuple[int, int, int, float, float, float]:
    """Greedy one-to-one matching within tolerance, return tp, fp, fn and prec/rec/f1.
    pred_frames and gt_frames are lists of frame indices.
    """
    if len(gt_frames) == 0:
        # If no positives, recall is defined as 1 by the official eval, precision by dets
        tp = 0
        fp = len(pred_frames)
        fn = 0
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 1.0
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        return tp, fp, fn, prec, rec, f1

    # Build distance matrix (|gt - pred|)
    if len(pred_frames) == 0:
        return 0, 0, len(gt_frames), 0.0, 0.0, 0.0

    dist = np.abs(np.subtract.outer(np.array(gt_frames, dtype=np.int64), np.array(pred_frames, dtype=np.int64)))
    tp = 0
    used_pred = set()
    # Greedy: for each gt in order, take nearest pred if within tol
    for gi in range(dist.shape[0]):
        if dist.shape[1] == 0:
            break
        # mask out used preds by setting large distance
        masked = dist[gi, :].copy()
        if used_pred:
            for pi in used_pred:
                if 0 <= pi < masked.shape[0]:
                    masked[pi] = tol_frames + 1
        min_pi = int(np.argmin(masked))
        if masked[min_pi] <= tol_frames and min_pi not in used_pred:
            tp += 1
            used_pred.add(min_pi)

    fp = len(pred_frames) - tp
    fn = len(gt_frames) - tp
    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return tp, fp, fn, prec, rec, f1


def main():
    parser = argparse.ArgumentParser(description="Quick eval (best rater per video) at a fixed tolerance")
    parser.add_argument('--gt-path', type=str, required=True, help='Path to GT pickle file')
    parser.add_argument('--pred-dir', type=str, required=True, help='Directory with prediction .pkl files')
    parser.add_argument('--downsample', type=int, default=3, help='Downsample factor used in predictions')
    parser.add_argument('--min-f1-filter', type=float, default=0.3, help='Filter videos with f1_consis_avg below this')
    parser.add_argument('--tol', type=float, default=0.3, help='Tolerance as fraction of video length (0-1)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of videos to evaluate')
    args = parser.parse_args()

    gt_dict: Dict = load_pickle(args.gt_path)

    # Collect prediction files intersecting GT after filter
    vid_ids = [vid for vid, info in gt_dict.items() if info.get('f1_consis_avg', 0) >= args.min_f1_filter]
    if args.limit:
        vid_ids = vid_ids[:args.limit]

    preds = {}
    for vid in vid_ids:
        pred_path = os.path.join(args.pred_dir, f"{vid}.pkl")
        if os.path.exists(pred_path):
            preds[vid] = pred_path

    if not preds:
        print("[WARN] No predictions found overlapping filtered GT set.")
        return

    per_video = []
    tp_all = 0
    fp_all = 0
    fn_all = 0

    for vid, pred_path in preds.items():
        info = gt_dict[vid]
        nframes = int(info['num_frames'])
        tol_frames = int(round(args.tol * (nframes - 1)))

        pred_obj = load_pickle(pred_path)
        pred_ds = list(pred_obj.get('bdy_idx_list_smt', []))
        pred_orig = [int(p) * args.downsample for p in pred_ds]
        # Clip to valid range
        pred_orig = [min(max(0, p), nframes - 1) for p in pred_orig]

        best = {'f1': -1.0, 'prec': 0.0, 'rec': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'rater': -1}
        raters: List[List[int]] = info.get('substages_myframeidx', [])
        for r_idx, gt_frames in enumerate(raters):
            gt_list = [int(x) for x in gt_frames]
            tp, fp, fn, prec, rec, f1 = match_and_score(pred_orig, gt_list, tol_frames)
            if f1 > best['f1']:
                best = {'f1': f1, 'prec': prec, 'rec': rec, 'tp': tp, 'fp': fp, 'fn': fn, 'rater': r_idx}

        per_video.append((vid, best))
        tp_all += best['tp']
        fp_all += best['fp']
        fn_all += best['fn']

    # Micro-averaged metrics
    prec_all = 0.0 if (tp_all + fp_all) == 0 else tp_all / (tp_all + fp_all)
    rec_all = 0.0 if (tp_all + fn_all) == 0 else tp_all / (tp_all + fn_all)
    f1_all = 0.0 if (prec_all + rec_all) == 0 else 2 * prec_all * rec_all / (prec_all + rec_all)

    # Macro-averaged F1 across videos
    macro_f1 = float(np.mean([v[1]['f1'] for v in per_video])) if per_video else 0.0

    print(f"[SUMMARY] videos={len(per_video)} tol={args.tol} downsample={args.downsample}")
    print(f"  micro: prec={prec_all:.3f} rec={rec_all:.3f} f1={f1_all:.3f}")
    print(f"  macro: f1={macro_f1:.3f}")

    # Print first 10 per-video lines
    for vid, best in per_video[:10]:
        print(f"  {vid}: f1={best['f1']:.3f} (p={best['prec']:.3f}, r={best['rec']:.3f}) rater={best['rater']} tp={best['tp']} fp={best['fp']} fn={best['fn']}")


if __name__ == '__main__':
    main()



