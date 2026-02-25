#!/usr/bin/env python3
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Boundary:
    time: float
    frame_index: int
    confidence: Optional[float]
    votes: Optional[int]
    visual_cues: Optional[List[str]]
    audio_cues: Optional[List[str]]
    raw: Dict[str, Any]


def load_metadata(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)


def parse_boundaries(meta: Dict[str, Any]) -> List[Boundary]:
    boundaries: List[Boundary] = []
    for b in meta.get("boundaries", []):
        boundaries.append(
            Boundary(
                time=float(b.get("time")),
                frame_index=int(b.get("frame_index")),
                confidence=b.get("confidence"),
                votes=b.get("votes"),
                visual_cues=b.get("visual_cues"),
                audio_cues=b.get("audio_cues"),
                raw=b,
            )
        )
    return boundaries


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
        raise e


def find_nearest_boundary_by_time(boundaries: List[Boundary], t: float) -> Boundary:
    return min(boundaries, key=lambda b: abs(b.time - t))


def find_nearest_boundary_by_frame(boundaries: List[Boundary], frame_index: int) -> Boundary:
    return min(boundaries, key=lambda b: abs(b.frame_index - frame_index))


def format_segment_dir_name(boundary: Boundary) -> str:
    safe_time = f"{boundary.time:.2f}".replace(".", "_")
    return f"time_{safe_time}_frame_{boundary.frame_index}"


def clip_time_window(center_time: float, pre: float, post: float, video_duration: Optional[float]) -> Tuple[float, float]:
    start = max(0.0, center_time - pre)
    end = center_time + post
    if video_duration is not None:
        end = min(end, video_duration)
    if end <= start:
        end = min(center_time + max(pre, post), (video_duration or (center_time + max(pre, post))))
    return start, end


def probe_duration(video_path: str) -> Optional[float]:
    try:
        # Use ffprobe to get duration in seconds
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        duration_str = result.stdout.strip()
        return float(duration_str) if duration_str else None
    except Exception:
        return None


def extract_clip(video_path: str, start: float, end: float, out_clip_path: str) -> None:
    duration = max(0.0, end - start)
    # Re-encode for accurate cuts across non-keyframes
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        out_clip_path,
    ]
    run(cmd)


def extract_audio(clip_path: str, out_audio_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        clip_path,
        "-vn",
        "-acodec",
        "pcm_s16le" if out_audio_path.lower().endswith(".wav") else "aac",
        out_audio_path,
    ]
    run(cmd)


def extract_frames(clip_path: str, frames_dir: str, fps: float) -> None:
    ensure_dir(frames_dir)
    # Zero-padded frame files
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        clip_path,
        "-r",
        f"{fps}",
        os.path.join(frames_dir, "frame_%05d.jpg"),
    ]
    run(cmd)


def write_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract visualization bundles for selected boundaries.")
    parser.add_argument("--json", required=True, help="Path to av_boundaries.json")
    parser.add_argument("--video", required=False, default=None, help="Override video path; defaults to json's video_path")
    parser.add_argument("--outdir", required=True, help="Output directory for visualization bundles")
    parser.add_argument("--times", required=False, default=None, help="Comma-separated seconds for centers (e.g., 77.44,93.73)")
    parser.add_argument("--frames", required=False, default=None, help="Comma-separated frame indices for centers (e.g., 580,702)")
    parser.add_argument("--pre_sec", type=float, default=2.0, help="Seconds before center time")
    parser.add_argument("--post_sec", type=float, default=2.0, help="Seconds after center time")
    parser.add_argument("--sample_fps", type=float, default=None, help="FPS for extracted frames; default=meta.sample_fps or 8")

    args = parser.parse_args()

    meta = load_metadata(args.__dict__["json"])  # avoid shadowing builtins
    boundaries = parse_boundaries(meta)
    if not boundaries:
        print("No boundaries found in JSON.", file=sys.stderr)
        sys.exit(1)

    video_path = args.video or meta.get("video_path")
    if not video_path:
        print("Video path not provided and not present in JSON.", file=sys.stderr)
        sys.exit(1)

    sample_fps = args.sample_fps or meta.get("sample_fps") or 8.0
    ensure_dir(args.outdir)

    # Save top-level metadata for reference
    write_json(os.path.join(args.outdir, "metadata.json"), {
        k: v for k, v in meta.items() if k != "boundaries"
    })

    # Determine selections
    selections: List[Boundary] = []
    if args.times:
        for t_str in args.times.split(","):
            t = float(t_str.strip())
            selections.append(find_nearest_boundary_by_time(boundaries, t))
    if args.frames:
        for f_str in args.frames.split(","):
            fi = int(f_str.strip())
            selections.append(find_nearest_boundary_by_frame(boundaries, fi))

    # De-duplicate selections by (time, frame_index)
    seen: set = set()
    unique_selections: List[Boundary] = []
    for s in selections:
        key = (round(s.time, 3), s.frame_index)
        if key not in seen:
            seen.add(key)
            unique_selections.append(s)

    if not unique_selections:
        print("No selections provided via --times or --frames.", file=sys.stderr)
        sys.exit(1)

    # Probe duration for clipping windows
    duration = probe_duration(video_path)

    # Process each selection
    manifest: List[Dict[str, Any]] = []
    for idx, boundary in enumerate(unique_selections, start=1):
        seg_dir = os.path.join(args.outdir, f"seg_{idx:02d}_" + format_segment_dir_name(boundary))
        ensure_dir(seg_dir)

        start_sec, end_sec = clip_time_window(boundary.time, args.pre_sec, args.post_sec, duration)
        clip_path = os.path.join(seg_dir, "clip.mp4")
        audio_path = os.path.join(seg_dir, "audio.wav")
        frames_dir = os.path.join(seg_dir, "frames")

        extract_clip(video_path, start_sec, end_sec, clip_path)
        extract_audio(clip_path, audio_path)
        extract_frames(clip_path, frames_dir, float(sample_fps))

        # Save model/boundary snippet
        write_json(os.path.join(seg_dir, "model_output.json"), boundary.raw)

        # Save per-segment manifest
        seg_info = {
            "segment_index": idx,
            "boundary_time": boundary.time,
            "boundary_frame_index": boundary.frame_index,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": max(0.0, end_sec - start_sec),
            "clip_path": clip_path,
            "audio_path": audio_path,
            "frames_dir": frames_dir,
        }
        write_json(os.path.join(seg_dir, "segment.json"), seg_info)
        manifest.append(seg_info)

    write_json(os.path.join(args.outdir, "manifest.json"), {
        "video_path": video_path,
        "selections": manifest,
    })

    print(f"Wrote {len(unique_selections)} segments to {args.outdir}")


if __name__ == "__main__":
    main()





