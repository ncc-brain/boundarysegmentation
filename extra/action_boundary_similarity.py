#!/usr/bin/env python3
"""
Compute action-model clip similarity around candidate boundary times.

Given a video and a list of boundary times (seconds), extract two clips
around each boundary (before and after) and compute cosine similarity
between their video embeddings using a pretrained action model
(torchvision r3d_18 by default). Lower similarity indicates a boundary.

Outputs a CSV with time and similarity; optionally a JSON.
"""

import argparse
import csv
import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


def read_times_from_txt(path: str) -> List[float]:
    times: List[float] = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            try:
                times.append(float(s))
            except ValueError:
                continue
    return times


def read_times_from_json(path: str) -> List[float]:
    with open(path, 'r') as f:
        data = json.load(f)
    if 'boundary_times' in data:
        return [float(x) for x in data['boundary_times']]
    raise KeyError("JSON missing 'boundary_times'")


def load_model(device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()
    return model, preprocess, device


def read_clip_frames(cap: cv2.VideoCapture, frame_indices: List[int]) -> np.ndarray:
    frames = []
    for idx in frame_indices:
        if idx < 0:
            idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)  # (T, H, W, C)


def make_clip_indices(center_time: float, fps: float, clip_len: int, direction: str) -> List[int]:
    center_idx = int(round(center_time * fps))
    if direction == 'before':
        start = center_idx - clip_len
        return list(range(start, center_idx))
    elif direction == 'after':
        end = center_idx + clip_len
        return list(range(center_idx, end))
    else:
        raise ValueError("direction must be 'before' or 'after'")


def video_to_tensor(frames_thwc: np.ndarray, preprocess) -> torch.Tensor:
    # frames_thwc: (T, H, W, C) uint8
    if frames_thwc.size == 0:
        return torch.empty(0)
    # Torchvision video models expect (C, T, H, W) float tensor
    video = torch.from_numpy(frames_thwc).permute(3, 0, 1, 2).float() / 255.0
    video = preprocess(video)  # (C, T, H, W)
    return video


def compute_embedding(model, device, video_tensor: torch.Tensor) -> np.ndarray:
    if video_tensor.numel() == 0:
        return np.zeros((512,), dtype=np.float32)
    with torch.no_grad():
        x = video_tensor.unsqueeze(0).to(device)  # (1, C, T, H, W)
        feat = model(x)  # (1, D)
        emb = torch.nn.functional.normalize(feat, dim=1)
        return emb.squeeze(0).cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def main():
    parser = argparse.ArgumentParser(description='Action model similarity around boundary times')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('times_path', type=str, help='Path to boundary_times.txt or boundaries.json')
    parser.add_argument('--delta_sec', type=float, default=1.5, help='Offset from boundary for clip centers')
    parser.add_argument('--clip_len', type=int, default=12, help='Frames per clip (before/after)')
    parser.add_argument('--output_csv', type=str, default='outputs/action_similarity.csv', help='Output CSV path')
    parser.add_argument('--output_json', type=str, default=None, help='Optional JSON output path')
    parser.add_argument('--start_offset_sec', type=float, default=0.0, help='Offset to add to TXT times')

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    # Load times
    if args.times_path.lower().endswith('.txt'):
        times = [t + args.start_offset_sec for t in read_times_from_txt(args.times_path)]
    else:
        times = read_times_from_json(args.times_path)

    # Init video and model
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model, preprocess, device = load_model()

    results: List[Tuple[float, float]] = []

    for t in times:
        pre_center = max(0.0, t - args.delta_sec)
        post_center = t + args.delta_sec

        pre_idx = make_clip_indices(pre_center, fps, args.clip_len, direction='before')
        post_idx = make_clip_indices(post_center, fps, args.clip_len, direction='after')

        # Bounds clamp
        pre_idx = [min(max(0, i), total_frames - 1) for i in pre_idx]
        post_idx = [min(max(0, i), total_frames - 1) for i in post_idx]

        pre_frames = read_clip_frames(cap, pre_idx)
        post_frames = read_clip_frames(cap, post_idx)

        pre_video = video_to_tensor(pre_frames, preprocess)
        post_video = video_to_tensor(post_frames, preprocess)

        pre_emb = compute_embedding(model, device, pre_video)
        post_emb = compute_embedding(model, device, post_video)

        sim = cosine_similarity(pre_emb, post_emb)
        results.append((t, sim))

    cap.release()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_sec', 'similarity'])
        writer.writerows(results)
    print(f"Saved similarity scores to {args.output_csv}")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({'times': [t for t, _ in results], 'similarity': [s for _, s in results]}, f, indent=2)
        print(f"Saved JSON to {args.output_json}")


if __name__ == '__main__':
    main()



