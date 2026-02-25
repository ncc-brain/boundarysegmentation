#!/usr/bin/env python3
"""
check_videos_against_gt.py
Scan a video directory recursively and report which GT videos can be resolved.
"""

import os
import argparse
import pickle
import json
import re
from tqdm import tqdm

VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}


def _norm_key(name):
    if not name:
        return []
    base = os.path.basename(str(name))
    lower = base.lower()
    stem, ext = os.path.splitext(lower)
    prefix = stem.split('_')[0]
    slug = re.sub(r"[^a-z0-9]", "", stem)
    slug_prefix = re.sub(r"[^a-z0-9]", "", prefix)
    return list({lower, stem, slug, prefix, slug_prefix})


def build_video_index(root_dir, allowed_exts=VIDEO_EXTS):
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
    if not paths:
        return None
    def score(p):
        ext = os.path.splitext(p)[1].lower()
        ext_score = 0 if ext == '.mp4' else 1
        return (ext_score, len(p))
    return sorted(paths, key=score)[0]


def resolve_video_path(vid_id, vid_info, video_dir, index):
    video_filename = vid_info.get('path_video')
    if video_filename:
        direct = os.path.join(video_dir, video_filename)
        if os.path.exists(direct):
            return direct
        base = os.path.basename(video_filename)
        for key in _norm_key(base):
            if key in index:
                return _pick_best(index[key])
    for ext in VIDEO_EXTS:
        for key in _norm_key(f"{vid_id}{ext}"):
            if key in index:
                return _pick_best(index[key])
    for key in _norm_key(vid_id):
        if key in index:
            return _pick_best(index[key])
    return None


def main():
    parser = argparse.ArgumentParser(description="Check GT videos against a directory recursively")
    parser.add_argument('--gt-path', type=str, required=True, help='Path to GT pickle file')
    parser.add_argument('--video-dir', type=str, required=True, help='Root directory of videos')
    parser.add_argument('--max-videos', type=int, default=None, help='Limit number of GT entries to check')
    parser.add_argument('--output-json', type=str, default=None, help='Optional path to save JSON report')
    args = parser.parse_args()

    with open(args.gt_path, 'rb') as f:
        gt_dict = pickle.load(f)
    vid_items = list(gt_dict.items())
    if args.max_videos:
        vid_items = vid_items[:args.max_videos]

    print(f"[INFO] Building video index for {args.video_dir} (recursive)...")
    index = build_video_index(args.video_dir)
    total_files = sum(len(v) for v in index.values())
    print(f"[INFO] Indexed {total_files} files (unique keys: {len(index)})")

    found = {}
    missing = {}
    for vid_id, vid_info in tqdm(vid_items, desc='Resolving'):
        path = resolve_video_path(vid_id, vid_info, args.video_dir, index)
        if path:
            found[vid_id] = path
        else:
            missing[vid_id] = {
                'hint': vid_info.get('path_video')
            }

    print(f"\n[REPORT] Found: {len(found)} | Missing: {len(missing)} | Total checked: {len(vid_items)}")
    if missing:
        sample_missing = list(missing.items())[:10]
        print("[SAMPLE MISSING] (up to 10)")
        for vid_id, info in sample_missing:
            print(f"  - {vid_id} (hint: {info['hint']})")

    if args.output_json:
        report = {
            'found': found,
            'missing': missing,
            'stats': {
                'found': len(found),
                'missing': len(missing),
                'total_checked': len(vid_items),
                'indexed_files': total_files
            }
        }
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Report saved to {args.output_json}")


if __name__ == '__main__':
    main()


