#!/usr/bin/env python3
"""
Evaluate detected boundaries against ground truth with frame tolerances.

Loads detected boundaries from JSON or TXT and ground truth from
Excel/CSV/TXT. Computes precision/recall/F1 at tolerances in frames
(1..5) given a reference FPS.
"""

import argparse
import json
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


def load_detected(det_path: str, offset_sec: float = 0.0, max_rows: int | None = None) -> List[float]:
    if det_path.lower().endswith('.json'):
        with open(det_path, 'r') as f:
            data = json.load(f)
        if 'boundary_times' in data:
            times = [float(t) for t in data['boundary_times']]
            if max_rows is not None:
                times = times[:max_rows]
            if offset_sec:
                times = [t + offset_sec for t in times]
            return times
        raise KeyError("JSON missing 'boundary_times'")
    # TXT
    times: List[float] = []
    with open(det_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            try:
                times.append(float(s))
            except ValueError:
                continue
    if max_rows is not None:
        times = times[:max_rows]
    if offset_sec:
        times = [t + offset_sec for t in times]
    return times


def load_ground_truth(
    gt_path: str,
    offset_sec: float = 0.0,
    max_rows: int | None = None,
    preferred_column: Optional[str] = None,
    coarse_scenes: bool = False,
) -> List[float]:
    lower = gt_path.lower()
    if lower.endswith('.xlsx') or lower.endswith('.xls'):
        df = pd.read_excel(gt_path)
        if coarse_scenes:
            # Apply row cap before deriving coarse scene boundaries
            if max_rows is not None:
                df = df.head(max_rows)
            # Derive times: when 'Scene Segments' has an entry, take previous row's 'End Time (s) '
            scene_cols = ['Scene Segments', 'Scene Segments ']
            end_candidates = ['End Time (s) ', 'End Time (s)', 'end', 'End', 'end_time', 'End Time', 'EndTime', 'end_sec', 'EndSeconds']
            scene_col = next((c for c in scene_cols if c in df.columns), None)
            end_col = next((c for c in end_candidates if c in df.columns), None)
            if scene_col is None or end_col is None:
                raise ValueError("Coarse scenes requires columns 'Scene Segments' and an end-time column")
            vals: List[float] = []
            scene_values = df[scene_col].tolist()
            for i, val in enumerate(scene_values):
                if pd.notna(val) and str(val).strip() != '':
                    prev_idx = i - 1
                    if prev_idx >= 0:
                        prev_end = df[end_col].iloc[prev_idx]
                        if pd.notna(prev_end):
                            vals.append(float(prev_end))
            if max_rows is not None:
                vals = vals[:max_rows]
            if offset_sec:
                vals = [t + offset_sec for t in vals]
            return vals
        # If a preferred column is specified, use it strictly if present
        if preferred_column and preferred_column in df.columns:
            vals = [float(x) for x in df[preferred_column].dropna().tolist()]
            if max_rows is not None:
                vals = vals[:max_rows]
            if offset_sec:
                vals = [t + offset_sec for t in vals]
            return vals
        # Prefer end times explicitly (note: check for trailing spaces in column names)
        end_candidates = ['End Time (s) ', 'End Time (s)', 'end', 'End', 'end_time', 'End Time', 'EndTime', 'end_sec', 'EndSeconds']
        for col in end_candidates:
            if col in df.columns:
                vals = [float(x) for x in df[col].dropna().tolist()]
                if max_rows is not None:
                    vals = vals[:max_rows]
                if offset_sec:
                    vals = [t + offset_sec for t in vals]
                return vals
        # Otherwise try common generic names
        for col in ['time', 'Time', 'seconds', 'Seconds', 'boundary_time', 'BoundaryTime']:
            if col in df.columns:
                vals = [float(x) for x in df[col].dropna().tolist()]
                if max_rows is not None:
                    vals = vals[:max_rows]
                if offset_sec:
                    vals = [t + offset_sec for t in vals]
                return vals
        # Fallback: first numeric column
        num_cols = df.select_dtypes(include=[float, int]).columns
        if len(num_cols) == 0:
            raise ValueError('No usable time column found in Excel ground truth')
        vals = [float(x) for x in df[num_cols[0]].dropna().tolist()]
        if max_rows is not None:
            vals = vals[:max_rows]
        if offset_sec:
            vals = [t + offset_sec for t in vals]
        return vals
    if lower.endswith('.csv'):
        df = pd.read_csv(gt_path)
        if coarse_scenes:
            # Apply row cap before deriving coarse scene boundaries
            if max_rows is not None:
                df = df.head(max_rows)
            scene_cols = ['Scene Segments', 'Scene Segments ']
            end_candidates = ['End Time (s) ', 'End Time (s)', 'end', 'End', 'end_time', 'End Time', 'EndTime', 'end_sec', 'EndSeconds']
            scene_col = next((c for c in scene_cols if c in df.columns), None)
            end_col = next((c for c in end_candidates if c in df.columns), None)
            if scene_col is None or end_col is None:
                raise ValueError("Coarse scenes requires columns 'Scene Segments' and an end-time column")
            vals: List[float] = []
            scene_values = df[scene_col].tolist()
            for i, val in enumerate(scene_values):
                if pd.notna(val) and str(val).strip() != '':
                    prev_idx = i - 1
                    if prev_idx >= 0:
                        prev_end = df[end_col].iloc[prev_idx]
                        if pd.notna(prev_end):
                            vals.append(float(prev_end))
            if max_rows is not None:
                vals = vals[:max_rows]
            if offset_sec:
                vals = [t + offset_sec for t in vals]
            return vals
        if preferred_column and preferred_column in df.columns:
            vals = [float(x) for x in df[preferred_column].dropna().tolist()]
            if max_rows is not None:
                vals = vals[:max_rows]
            if offset_sec:
                vals = [t + offset_sec for t in vals]
            return vals
        end_candidates = ['End Time (s) ', 'End Time (s)', 'end', 'End', 'end_time', 'End Time', 'EndTime', 'end_sec', 'EndSeconds']
        for col in end_candidates:
            if col in df.columns:
                vals = [float(x) for x in df[col].dropna().tolist()]
                if max_rows is not None:
                    vals = vals[:max_rows]
                if offset_sec:
                    vals = [t + offset_sec for t in vals]
                return vals
        for col in ['time', 'Time', 'seconds', 'Seconds', 'boundary_time', 'BoundaryTime']:
            if col in df.columns:
                vals = [float(x) for x in df[col].dropna().tolist()]
                if max_rows is not None:
                    vals = vals[:max_rows]
                if offset_sec:
                    vals = [t + offset_sec for t in vals]
                return vals
        num_cols = df.select_dtypes(include=[float, int]).columns
        if len(num_cols) == 0:
            raise ValueError('No usable time column found in CSV ground truth')
        vals = [float(x) for x in df[num_cols[0]].dropna().tolist()]
        if max_rows is not None:
            vals = vals[:max_rows]
        if offset_sec:
            vals = [t + offset_sec for t in vals]
        return vals
    # TXT
    times: List[float] = []
    with open(gt_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            try:
                times.append(float(s))
            except ValueError:
                continue
    if max_rows is not None:
        times = times[:max_rows]
    if offset_sec:
        times = [t + offset_sec for t in times]
    return times


def match_boundaries(det_times: List[float], gt_times: List[float], fps: float, tol_frames: int) -> Dict[str, float]:
    det_times = sorted(det_times)
    gt_times = sorted(gt_times)
    tol_sec = tol_frames / fps

    used_gt = np.zeros(len(gt_times), dtype=bool)
    tp = 0
    fp = 0
    for t in det_times:
        # find nearest unmatched gt within tol
        best_idx = -1
        best_dist = 1e9
        for i, g in enumerate(gt_times):
            if used_gt[i]:
                continue
            d = abs(t - g)
            if d <= tol_sec and d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            tp += 1
            used_gt[best_idx] = True
        else:
            fp += 1
    fn = int((~used_gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'tp': float(tp), 'fp': float(fp), 'fn': float(fn),
        'precision': precision, 'recall': recall, 'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate detected boundaries against ground truth')
    parser.add_argument('detected_path', type=str, help='Detected boundaries (JSON with boundary_times or TXT)')
    parser.add_argument('ground_truth_path', type=str, help='Ground truth boundaries (XLSX/CSV/TXT)')
    parser.add_argument('--fps', type=float, required=True, help='Reference FPS for frame tolerances')
    parser.add_argument('--tolerances', nargs='+', type=int, default=[5, 10, 15, 25, 50],
                        help='Frame tolerances to evaluate (e.g., 1 3 5)')
    parser.add_argument('--output', type=str, default='outputs/eval_metrics.json', help='Where to save metrics')
    parser.add_argument('--gt_column', type=str, default=None, help="Exact column name to read from ground truth (e.g., 'End Time (s)')")
    parser.add_argument('--coarse-scenes', action='store_true',
                        help="If set, derive ground truth from coarse scene markers: whenever 'Scene Segments' has an entry, take previous row's End Time (s)")
    # Offsets and row limits
    parser.add_argument('--det_offset_sec', type=float, default=0.0,
                        help='Offset to add to detected times (use negative to subtract cutoff)')
    parser.add_argument('--gt_offset_sec', type=float, default=0.0,
                        help='Offset to add to ground-truth times (use negative to subtract cutoff)')
    parser.add_argument('--gt_max_rows', type=int, default=482,
                        help='Only use first N rows from ground-truth (default 482)')
    parser.add_argument('--det_max_rows', type=int, default=None,
                        help='Only use first N detected times (optional)')

    args = parser.parse_args()

    det = load_detected(args.detected_path, offset_sec=args.det_offset_sec, max_rows=args.det_max_rows)
    gt = load_ground_truth(
        args.ground_truth_path,
        offset_sec=args.gt_offset_sec,
        max_rows=args.gt_max_rows,
        preferred_column=args.gt_column,
        coarse_scenes=args.coarse_scenes,
    )

    print(f"Detected times: {det}")
    print(f"Ground truth times: {gt}")

    metrics = {}
    for tol in args.tolerances:
        metrics[f'frames_{tol}'] = match_boundaries(det, gt, args.fps, tol)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'fps': args.fps, 'metrics': metrics}, f, indent=2)
    print(f"Saved evaluation to {args.output}")


if __name__ == '__main__':
    main()
