#!/usr/bin/env python3
"""
Visualize boundary frame transitions ex post from results JSON.

For each detected boundary, extracts the two frames around the transition
from the original video and composes a paginated grid of before/after thumbnails.
"""

import argparse
import json
import math
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_results(results_path: str) -> dict:
    with open(results_path, 'r') as f:
        data = json.load(f)
    required_keys = ['video_path', 'frame_indices', 'boundaries']
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing '{k}' in results file: {results_path}")
    return data


def read_boundary_times_txt(times_path: str) -> List[float]:
    times: List[float] = []
    with open(times_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            try:
                times.append(float(s))
            except ValueError:
                continue
    return times


def center_crop_square(image_bgr: np.ndarray, crop_size: Optional[int] = None) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    edge = min(h, w) if crop_size is None else min(crop_size, h, w)
    y0 = (h - edge) // 2
    x0 = (w - edge) // 2
    return image_bgr[y0:y0 + edge, x0:x0 + edge]


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    if frame_idx < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def make_thumbnail(image_bgr: np.ndarray, thumb_size: int, center_crop: bool, crop_size: Optional[int]) -> Image.Image:
    if center_crop:
        image_bgr = center_crop_square(image_bgr, crop_size)
    # Convert to RGB
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    img = img.resize((thumb_size, thumb_size), Image.BILINEAR)
    return img


def compose_pages(
    pairs: List[Tuple[Image.Image, Image.Image, float, int, int]],
    output_dir: str,
    max_per_page: int,
    thumb_size: int
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    num_pairs = len(pairs)
    num_pages = math.ceil(num_pairs / max_per_page) if num_pairs > 0 else 1

    for page in range(num_pages):
        start = page * max_per_page
        end = min(num_pairs, (page + 1) * max_per_page)
        page_pairs = pairs[start:end]

        rows = len(page_pairs)
        cols = 2
        canvas_w = cols * thumb_size
        canvas_h = rows * thumb_size
        canvas = Image.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for r, (pre_img, post_img, t_sec, pre_idx, post_idx) in enumerate(page_pairs):
            y = r * thumb_size
            # Pre (left)
            canvas.paste(pre_img, (0, y))
            # Post (right)
            canvas.paste(post_img, (thumb_size, y))

            # Labels
            label_left = f"t={t_sec:.2f}s  pre={pre_idx}"
            label_right = f"t={t_sec:.2f}s  post={post_idx}"
            draw.rectangle([0, y, thumb_size - 1, y + 18], fill=(0, 0, 0))
            draw.rectangle([thumb_size, y, 2 * thumb_size - 1, y + 18], fill=(0, 0, 0))
            draw.text((4, y + 2), label_left, fill=(255, 255, 255), font=font)
            draw.text((thumb_size + 4, y + 2), label_right, fill=(255, 255, 255), font=font)

        out_path = os.path.join(output_dir, f"transitions_page_{page + 1:03d}.png")
        canvas.save(out_path)
        print(f"Saved {out_path}")


def compose_separate(
    pairs: List[Tuple[Image.Image, Image.Image, float, int, int]],
    output_dir: str,
    thumb_size: int
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    for idx, (pre_img, post_img, t_sec, pre_idx, post_idx) in enumerate(pairs):
        canvas_w = 2 * thumb_size
        canvas_h = thumb_size + 22
        canvas = Image.new('RGB', (canvas_w, canvas_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # Paste images
        canvas.paste(pre_img, (0, 22))
        canvas.paste(post_img, (thumb_size, 22))

        # Labels on top bar
        label_left = f"pre {pre_idx}"
        if t_sec >= 0:
            label_left = f"t={t_sec:.2f}s  pre={pre_idx}"
        label_right = f"post {post_idx}"
        if t_sec >= 0:
            label_right = f"t={t_sec:.2f}s  post={post_idx}"
        draw.text((6, 3), label_left, fill=(255, 255, 255), font=font)
        draw.text((thumb_size + 6, 3), label_right, fill=(255, 255, 255), font=font)

        # Filename includes boundary ordinal and time
        t_tag = f"t{t_sec:.2f}" if t_sec >= 0 else "tNA"
        out_path = os.path.join(output_dir, f"transition_{idx+1:05d}_{t_tag}.png")
        canvas.save(out_path)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize boundary transitions from results (JSON or TXT)')
    parser.add_argument('results_path', type=str, help='Path to boundaries.json or boundary_times.txt')
    parser.add_argument('--video_path', type=str, default=None, help='Override video path in results')
    parser.add_argument('--output_dir', type=str, default='outputs/transitions', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['separate','pages'], default='separate',
                        help='Output one image per boundary (separate) or paginated sheets (pages)')
    parser.add_argument('--max_per_page', type=int, default=30, help='Max pairs per page (pages mode)')
    parser.add_argument('--thumb_size', type=int, default=256, help='Thumbnail square size in pixels')
    parser.add_argument('--center_crop', action='store_true', help='Center-crop frames before resizing')
    parser.add_argument('--crop_size', type=int, default=None, help='Optional crop edge length before resizing')
    parser.add_argument('--start_at', type=int, default=0, help='Start at boundary index (inclusive)')
    parser.add_argument('--end_at', type=int, default=None, help='End at boundary index (exclusive)')
    parser.add_argument('--start_offset_sec', type=float, default=0.0,
                        help='Add this offset to TXT times (e.g., segment starts at 123s)')
    parser.add_argument('--delta_sec', type=float, default=1.5,
                        help='Offset in seconds before/after boundary for pre/post frames')

    args = parser.parse_args()

    pairs: List[Tuple[Image.Image, Image.Image, float, int, int]] = []

    # Decide mode: JSON vs TXT
    if args.results_path.lower().endswith('.txt'):
        # Simple mode from boundary_times.txt
        if not args.video_path:
            raise ValueError("--video_path is required when using a .txt times file")
        video_path = args.video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        times_all = read_boundary_times_txt(args.results_path)
        # Subset times
        end_at = args.end_at if args.end_at is not None else len(times_all)
        start_at = max(0, int(args.start_at))
        end_at = min(len(times_all), int(end_at))
        times_slice = times_all[start_at:end_at]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        delta = max(0.0, float(args.delta_sec))
        for t_rel in times_slice:
            t_abs = float(t_rel) + float(args.start_offset_sec)
            pre_time = max(0.0, t_abs - delta)
            post_time = max(pre_time, t_abs + delta)
            pre_frame_idx = int(round(pre_time * fps))
            post_frame_idx = int(round(post_time * fps))
            # Ensure different indices and within bounds
            if post_frame_idx <= pre_frame_idx:
                post_frame_idx = min(pre_frame_idx + 1, total_frames - 1)
            pre_frame_idx = min(max(0, pre_frame_idx), total_frames - 2)
            post_frame_idx = min(max(1, post_frame_idx), total_frames - 1)

            pre_frame = read_frame(cap, pre_frame_idx)
            post_frame = read_frame(cap, post_frame_idx)
            if pre_frame is None or post_frame is None:
                continue

            pre_img = make_thumbnail(pre_frame, args.thumb_size, args.center_crop, args.crop_size)
            post_img = make_thumbnail(post_frame, args.thumb_size, args.center_crop, args.crop_size)
            pairs.append((pre_img, post_img, t_abs, pre_frame_idx, post_frame_idx))

        cap.release()
    else:
        # JSON mode (existing behavior)
        data = read_results(args.results_path)
        video_path = args.video_path or data['video_path']
        boundaries: List[int] = data['boundaries']
        frame_indices: List[int] = data['frame_indices']
        boundary_times: Optional[List[float]] = data.get('boundary_times')

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Subset boundaries (use actual boundary positions, not ordinal indices)
        end_at = args.end_at if args.end_at is not None else len(boundaries)
        start_at = max(0, int(args.start_at))
        end_at = min(len(boundaries), int(end_at))
        boundary_positions = boundaries[start_at:end_at]
        times_slice = boundary_times[start_at:end_at] if boundary_times else [None] * (end_at - start_at)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delta = max(0.0, float(args.delta_sec))
        for b_idx, b_pos in enumerate(boundary_positions):
            if b_pos <= 0 or b_pos >= len(frame_indices):
                continue
            t_val = times_slice[b_idx]
            if t_val is not None:
                # Prefer time-based sampling around boundary time
                pre_time = max(0.0, float(t_val) - delta)
                post_time = max(pre_time, float(t_val) + delta)
                pre_frame_idx = int(round(pre_time * fps))
                post_frame_idx = int(round(post_time * fps))
            else:
                # Fallback to adjacent sampled indices
                pre_sample_idx = int(b_pos) - 1
                post_sample_idx = int(b_pos)
                pre_frame_idx = int(frame_indices[pre_sample_idx])
                post_frame_idx = int(frame_indices[post_sample_idx])

            pre_frame = read_frame(cap, pre_frame_idx)
            post_frame = read_frame(cap, post_frame_idx)
            if pre_frame is None or post_frame is None:
                continue

            pre_img = make_thumbnail(pre_frame, args.thumb_size, args.center_crop, args.crop_size)
            post_img = make_thumbnail(post_frame, args.thumb_size, args.center_crop, args.crop_size)

            t_sec = float(t_val) if t_val is not None else -1.0
            pairs.append((pre_img, post_img, t_sec, pre_frame_idx, post_frame_idx))

        cap.release()

    if not pairs:
        print("No pairs to visualize. Check your boundaries or indices.")
        return

    if args.mode == 'pages':
        compose_pages(pairs, args.output_dir, args.max_per_page, args.thumb_size)
    else:
        compose_separate(pairs, args.output_dir, args.thumb_size)


if __name__ == '__main__':
    main()


