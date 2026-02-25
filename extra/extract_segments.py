#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")


def sanitize_name(text: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in text).strip('_')


def ffmpeg_select_codec(copy: bool) -> list[str]:
    if copy:
        return [
            '-c:v', 'copy',
            '-c:a', 'copy'
        ]
    return [
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k'
    ]


def cut_clip(ffmpeg_bin: str, video_path: str, start: float, duration: float, out_path: str, reencode: bool) -> None:
    args = [
        ffmpeg_bin,
        '-hide_banner', '-loglevel', 'error',
        '-ss', f"{start:.3f}",
        '-i', video_path,
        '-t', f"{duration:.3f}",
        '-avoid_negative_ts', 'make_zero',
    ] + ffmpeg_select_codec(copy=not reencode) + [
        '-y', out_path
    ]
    run_cmd(args)


def extract_frames(ffmpeg_bin: str, clip_path: str, frames_dir: str, fps: float | None) -> None:
    os.makedirs(frames_dir, exist_ok=True)
    args = [
        ffmpeg_bin,
        '-hide_banner', '-loglevel', 'error',
        '-i', clip_path,
    ]
    if fps and fps > 0:
        args += ['-vf', f"fps={fps}"]
    args += [
        '-qscale:v', '2',
        os.path.join(frames_dir, 'frame_%06d.jpg')
    ]
    run_cmd(args)


def extract_frames_window(
    ffmpeg_bin: str,
    source_video: str,
    frames_dir: str,
    window_start: float,
    window_size: int,
    fps: float,
) -> None:
    os.makedirs(frames_dir, exist_ok=True)
    # Extract exactly window_size frames at fps within [window_start, window_start + window_size/fps]
    duration = max(0.0, window_size / fps)
    args = [
        ffmpeg_bin,
        '-hide_banner', '-loglevel', 'error',
        '-ss', f"{window_start:.3f}",
        '-i', source_video,
        '-t', f"{duration:.3f}",
        '-vf', f"fps={fps}",
        '-frames:v', str(window_size),
        '-qscale:v', '2',
        os.path.join(frames_dir, 'frame_%06d.jpg')
    ]
    run_cmd(args)


def extract_audio(ffmpeg_bin: str, clip_path: str, out_wav: str) -> None:
    args = [
        ffmpeg_bin,
        '-hide_banner', '-loglevel', 'error',
        '-i', clip_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '2',
        '-y', out_wav
    ]
    run_cmd(args)


def load_boundaries(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def find_boundary_by_time(data: dict, target_time: float, tolerance: float = 0.1) -> dict | None:
    best = None
    best_dt = float('inf')
    for b in data.get('boundaries', []):
        t = b.get('time')
        if t is None:
            continue
        dt = abs(t - target_time)
        if dt < best_dt:
            best_dt = dt
            best = b
    if best is None:
        return None
    if best_dt <= tolerance:
        return best
    return None


def parse_times_arg(times_arg: str) -> list[float]:
    values: list[float] = []
    for part in times_arg.split(','):
        part = part.strip()
        if not part:
            continue
        if part.endswith('s'):
            part = part[:-1]
        values.append(float(part))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract visualization segments (video, frames, audio, metadata) from a video using boundary JSON.')
    parser.add_argument('--json', required=True, help='Path to av_boundaries.json')
    parser.add_argument('--video', required=False, help='Override video path (if different from JSON)')
    parser.add_argument('--times', required=True, help='Comma-separated list of times in seconds (e.g., 77.44,93.73,84.12,1.07,78.51)')
    parser.add_argument('--pre', type=float, default=1.0, help='Seconds before boundary to include')
    parser.add_argument('--post', type=float, default=2.0, help='Seconds after boundary to include')
    parser.add_argument('--frames-fps', type=float, default=8.0, help='FPS for extracted frames (0 to keep source fps)')
    parser.add_argument('--frames-mode', choices=['clip_span', 'boundary_window'], default='clip_span', help='clip_span: frames from whole clip; boundary_window: exactly model window (e.g., 8 frames @ sample_fps)')
    parser.add_argument('--out-dir', required=True, help='Output directory root')
    parser.add_argument('--reencode', action='store_true', help='Re-encode instead of stream-copy when cutting clips')
    parser.add_argument('--ffmpeg-bin', default='ffmpeg', help='Path to ffmpeg binary')
    args = parser.parse_args()

    data = load_boundaries(args.json)
    source_video = args.video if args.video else data.get('video_path')
    if not source_video:
        print('Error: video path not provided and not present in JSON.', file=sys.stderr)
        sys.exit(2)
    if not Path(source_video).exists():
        print(f'Error: video not found at {source_video}', file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    times = parse_times_arg(args.times)
    fps_json = data.get('fps')
    sample_fps = data.get('sample_fps', 8.0)
    window_size_json = data.get('window_size', 8)

    for t in times:
        boundary = find_boundary_by_time(data, t, tolerance=0.3)
        boundary_time = boundary['time'] if boundary else t
        start = max(0.0, boundary_time - args.pre)
        duration = args.pre + args.post

        label_parts = [f"t{boundary_time:.2f}"]
        if boundary and 'frame_index' in boundary:
            label_parts.append(f"f{boundary['frame_index']}")
        seg_name = '_'.join(label_parts)
        seg_dir = os.path.join(args.out_dir, seg_name)
        os.makedirs(seg_dir, exist_ok=True)

        clip_mp4 = os.path.join(seg_dir, 'clip.mp4')
        frames_dir = os.path.join(seg_dir, 'frames')
        audio_wav = os.path.join(seg_dir, 'audio.wav')
        info_json = os.path.join(seg_dir, 'info.json')

        cut_clip(args.ffmpeg_bin, source_video, start, duration, clip_mp4, reencode=args.reencode)
        if args.frames_mode == 'boundary_window':
            # Compute window centered at boundary_time using model window_size and sample_fps
            win_size = int(window_size_json) if window_size_json else 8
            fps_use = float(sample_fps) if sample_fps else (args.frames_fps if args.frames_fps > 0 else 8.0)
            half_win_sec = (win_size / fps_use) / 2.0
            frames_start = max(0.0, boundary_time - half_win_sec)
            extract_frames_window(args.ffmpeg_bin, source_video, frames_dir, frames_start, win_size, fps_use)
        else:
            extract_frames(args.ffmpeg_bin, clip_mp4, frames_dir, fps=args.frames_fps if args.frames_fps > 0 else None)
        extract_audio(args.ffmpeg_bin, clip_mp4, audio_wav)

        segment_info = {
            'requested_time': t,
            'matched_boundary': boundary if boundary else None,
            'used_time': boundary_time,
            'window': {'start': start, 'duration': duration},
            'json_source': os.path.abspath(args.json),
            'video_source': os.path.abspath(source_video),
            'project_fps': fps_json,
            'frames_fps': (sample_fps if args.frames_mode == 'boundary_window' else (args.frames_fps if args.frames_fps > 0 else fps_json)),
            'frames_mode': args.frames_mode,
            'model_window_size': window_size_json,
            'ffmpeg': args.ffmpeg_bin,
        }
        with open(info_json, 'w') as f:
            json.dump(segment_info, f, indent=2)

    print(f"Done. Segments written to: {args.out_dir}")


if __name__ == '__main__':
    main()




