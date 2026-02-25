#!/usr/bin/env python3
"""
Run a parameter grid over fps_sample, n_states, and DINO layer; evaluate results.

For each combo, runs extract_boundries.py with:
  --start_time 123 --end_time 1553
and writes outputs to an informative folder name. Then evaluates using
evaluate_boundaries.py with the correct video FPS, tolerances, and
detected offset -123 (to remove the cutoff).

Produces a CSV table of metrics across tolerances for all combos.
"""

import argparse
import json
import os
import subprocess
from typing import List, Dict, Any

import cv2
import pandas as pd


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def read_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"Could not read FPS from video: {video_path}")
    return float(fps)


def main():
    parser = argparse.ArgumentParser(description='Grid search over fps_sample, n_states, layer and evaluate')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('ground_truth_path', type=str, help='Ground truth boundaries (XLSX/CSV/TXT)')
    parser.add_argument('--output_root', type=str, default='grid_runs', help='Root directory for runs')
    parser.add_argument('--fps_samples', nargs='+', type=int, default=[5, 15, 30])
    parser.add_argument('--n_states_list', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 5, 9, 11])
    parser.add_argument('--n_pca_list', nargs='+', type=int, default=[20, 30, 50])
    parser.add_argument('--tolerances', nargs='+', type=int, default=[1, 3, 5, 10, 15, 30, 60])
    parser.add_argument('--start_time', type=float, default=123.0)
    parser.add_argument('--end_time', type=float, default=1553.0)
    parser.add_argument('--model', type=str, default='facebook/dinov2-base')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Optional list of model names to sweep; overrides --model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--force', action='store_true', help='Re-run extraction even if outputs exist')

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    video_fps = read_video_fps(args.video_path)

    records: List[Dict[str, Any]] = []

    models_to_try = args.models if args.models else [args.model]

    for model_name in models_to_try:
        model_tag = model_name.replace('/', '-')
        for fps_sample in args.fps_samples:
            for n_states in args.n_states_list:
                for layer in args.layers:
                    for n_pca in args.n_pca_list:
                        run_dir = os.path.join(
                            args.output_root, f"{model_tag}_fps{fps_sample}_ns{n_states}_ly{layer}_pca{n_pca}"
                        )
                        os.makedirs(run_dir, exist_ok=True)

                        boundaries_json = os.path.join(run_dir, 'boundaries.json')

                        need_extract = args.force or (not os.path.exists(boundaries_json))
                        if need_extract:
                            extract_cmd = [
                                'python', os.path.join(os.path.dirname(__file__), 'extract_boundries.py'),
                                args.video_path,
                                '--n_states', str(n_states),
                                '--layer', str(layer),
                                '--fps_sample', str(fps_sample),
                                '--n_pca', str(n_pca),
                                '--batch_size', str(args.batch_size),
                                '--output_dir', run_dir,
                                '--model', model_name,
                                '--start_time', str(args.start_time),
                                '--end_time', str(args.end_time),
                                '--smooth', 'none',
                                '--min_segment_seconds', '0'
                            ]
                            print('Running extraction:', ' '.join(extract_cmd))
                            run_cmd(extract_cmd)
                        else:
                            print(f"Skipping extraction (exists): {boundaries_json}")

                        # Evaluate
                        eval_out = os.path.join(run_dir, 'eval_metrics.json')
                        eval_cmd = [
                            'python', os.path.join(os.path.dirname(__file__), 'evaluate_boundaries.py'),
                            boundaries_json,
                            args.ground_truth_path,
                            '--fps', str(video_fps),
                            '--tolerances', *[str(t) for t in args.tolerances],
                            '--det_offset_sec', str(-args.start_time),
                            '--gt_max_rows', '482',
                            '--output', eval_out
                        ]
                        print('Running eval:', ' '.join(eval_cmd))
                        run_cmd(eval_cmd)

                        # Read metrics and accumulate
                        with open(eval_out, 'r') as f:
                            metrics = json.load(f)['metrics']

                        row: Dict[str, Any] = {
                            'model': model_tag,
                            'fps_sample': fps_sample,
                            'n_states': n_states,
                            'layer': layer,
                            'n_pca': n_pca,
                        }
                        for tol, vals in metrics.items():
                            # tol like 'frames_5'
                            row[f'{tol}_precision'] = vals['precision']
                            row[f'{tol}_recall'] = vals['recall']
                            row[f'{tol}_f1'] = vals['f1']
                        records.append(row)

    # Save CSV summary
    summary_csv = os.path.join(args.output_root, 'grid_summary.csv')
    df = pd.DataFrame.from_records(records)
    sort_cols = [c for c in ['model', 'fps_sample', 'n_states', 'layer', 'n_pca'] if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)
    df.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")


if __name__ == '__main__':
    main()


