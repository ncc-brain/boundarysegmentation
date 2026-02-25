# Boundary Segmentation

Temporal event boundary detection for video using UBoCo (contrastive kernel), Qwen3-VL segmenter, and Qwen3-Omni describer.

> **Note:** The folder name is intentionally spelled `boundry_segmentation` (not "boundary").
>
> For a fast handoff flow, start with `QUICKSTART.md`.
>
> For canonical commands only, see `PIPELINES.md`.

## Project Overview

This project provides tools for detecting event boundaries in videos:

- **UBoCo** – Unsupervised Boundary Contrastive Learning (Kang et al., CVPR 2022) with RTP contrastive kernel boundary detection
- **Qwen segment** – Qwen3-VL sliding-window binary boundary prediction
- **Qwen describer** – Qwen3-Omni detailed video understanding (scene descriptions, audio transcription)

Outputs can be evaluated against reference boundaries using `evaluate_boundaries.py`.

## Environment

**Collaborators must create their own environment.** The verified setup below was run from a local env at `../envs/bseg` (outside this repo).

### Verified environment

Exact versions observed:

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| pip | 25.2 |
| torch | 2.8.0 |
| torchvision | 0.23.0 |
| transformers | 4.57.0 |
| opencv-python-headless | 4.12.0.88 |
| numpy | 2.2.6 |
| pandas | 2.3.2 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| tqdm | 4.67.1 |
| hmmlearn | 0.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.6 |
| seaborn | 0.13.2 |

Additional: `PIL`, `qwen-vl-utils` (for Qwen3-VL), `qwen_omni_utils` (for Qwen3-Omni). **Optional:** `ffmpeg` for audio/video extraction.

## Required Data

### Sherlock clip

Place the Sherlock intro clip at:

```
boundry_segmentation/sherlock.mp4
```

(or use an absolute path when invoking scripts). The clip is used for sanity checks and short eval runs.

### Where to run scripts

Run commands either **from inside** `boundry_segmentation/` (e.g. `python uboco_gebd.py sherlock.mp4 ...`) or **from the parent directory** with a prefixed path (e.g. `python boundry_segmentation/uboco_gebd.py boundry_segmentation/sherlock.mp4 ...`).

## Forward-Pass Commands

### a) UBoCo contrastive kernel boundary detection (short mode)

```bash
python uboco_gebd.py sherlock.mp4 \
  --boundary_method peaks \
  --rtp_kernel_size 5 \
  --rtp_min_length 50 \
  --rtp_threshold_diff 0.3 \
  --rtp_max_depth 3 \
  --rtp_max_boundaries 30 \
  --peaks_distance 30 \
  --peaks_prominence 0.6 \
  --peaks_max_boundaries 25 \
  --n_epochs 2 \
  --end_time 60 \
  --output_dir outputs/sanity_uboco
```

### b) Qwen segment (short mode, binary)

```bash
python qwen.py sherlock.mp4 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --response-mode binary \
  --end-time 10 \
  --output-dir outputs/sanity_qwen_segment
```

Quick sanity alternative (smaller model):

```bash
python qwen.py sherlock.mp4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --response-mode binary \
  --end-time 10 \
  --output-dir outputs/sanity_qwen_segment
```

### c) Qwen describer mode (short mode, debug-first-window)

```bash
python qwen_omni_describer.py sherlock.mp4 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --debug-save outputs/sanity_qwen_describer/debug_first_window \
  --end-time 8 \
  --window-size 4 \
  --stride 4 \
  --sample-fps 1 \
  --output-dir outputs/sanity_qwen_describer
```

The describer can be slow on a first run because Qwen3-Omni-30B has a heavy cold load plus generation.

## Short Eval Workflow

**Note:** Sherlock ground truth is not included in this repo. The sanity eval uses **reference boundaries derived from ubeco_sherlock output** (not human GT). The default reference file is `references/sherlock_reference_boundaries_from_ubeco.txt`. If that file is missing, `run_sherlock_sanity.sh` falls back to extracting from `outputs/captions_stride12/transfer boundaries/ubeco_sherlock.json`.

1. Reference boundaries: use `references/sherlock_reference_boundaries_from_ubeco.txt` (included in repo), or extract manually if needed:
   ```bash
   python -c "
   import json
   with open('outputs/captions_stride12/transfer boundaries/ubeco_sherlock.json') as f:
       d = json.load(f)
   for t in d['boundary_times']:
       print(t)
   " > outputs/sanity_eval/sherlock_reference_boundaries.txt
   ```

2. Evaluate UBoCo and Qwen outputs (using default reference in `references/`):
   ```bash
   python evaluate_boundaries.py outputs/sanity_uboco/boundary_times.txt references/sherlock_reference_boundaries_from_ubeco.txt --fps 25 --tolerances 5 10 15 --output outputs/sanity_eval/uboco_vs_reference.json
   python evaluate_boundaries.py outputs/sanity_qwen_segment/boundaries.json references/sherlock_reference_boundaries_from_ubeco.txt --fps 25 --tolerances 5 10 15 --output outputs/sanity_eval/qwen_vs_reference.json
   ```

Or run the full sanity script: `bash run_sherlock_sanity.sh`

By default the sanity script uses `Qwen/Qwen3-VL-2B-Instruct` for segmentation and skips describer (`RUN_DESCRIBER=0`).  
Enable describer explicitly with: `RUN_DESCRIBER=1 bash run_sherlock_sanity.sh`

## Verified on This Machine

| Step | Runtime (approx) | Output path |
|------|------------------|-------------|
| UBoCo short (60s, 2 epochs) | ~22 s | `outputs/sanity_uboco/uboco_boundaries.json`, `boundary_times.txt` |
| Qwen3 segment short (10s) | ~40 s | `outputs/sanity_qwen_segment/boundaries.json`, `boundary_times.txt` |
| Qwen3 describer debug (1 window) | ~25 min (cold load + first generation) | `outputs/sanity_qwen_describer/debug_first_window/window_00000/` |
| Reference (default) | — | `references/sherlock_reference_boundaries_from_ubeco.txt` |
| evaluate_boundaries | &lt;1 s | `outputs/sanity_eval/uboco_vs_reference.json`, `qwen_vs_reference.json` |

## Recorded UBECO / Qwen3 Reference Metrics

The repository includes recorded metric snapshots for traceability:

- `references/ubeco_eval_metrics_scenes.json`
  - `frames_10` F1 = `0.1509433962264151`
  - `frames_100` F1 = `0.18867924528301885`
- `references/qwen3_eval_metrics_ws12_prompt.json`
  - `frames_50` F1 = `0.3018867924528302`
  - `frames_100` F1 = `0.49056603773584906`
  - `frames_125` F1 = `0.5283018867924528`

## Known-Good Sherlock Parameter Sets

From `references/uboco_sherlock_params.txt` and existing outputs:

**UBoCo (peaks):**
- `--boundary_method peaks`
- `--rtp_kernel_size 5` `--rtp_min_length 50` `--rtp_threshold_diff 0.3`
- `--rtp_max_depth 3` `--rtp_max_boundaries 30`
- `--peaks_distance 30` `--peaks_prominence 0.6` `--peaks_max_boundaries 25`

**UBoCo (RTP):**
- `--boundary_method rtp` with same RTP params above

## GEBD Short Eval Examples

For full GEBD evaluation (requires GT pickle and video dir):

```bash
# UBoCo: limit to 2 videos
python run_uboco_on_gebd_eval.py --video-dir /path/to/videos --max-videos 2

# UBoCo: single video by ID
python run_uboco_on_gebd_eval.py --video-dir /path/to/videos --only-video <vid_id>

# Qwen: limit to 2 videos
python run_qwen_on_gebd_eval.py --video-dir /path/to/videos --max-videos 2

# Qwen: single video
python run_qwen_on_gebd_eval.py --video-dir /path/to/videos --only-video <vid_id>
```

## Repo Init / Push (without actually pushing)

```bash
# Initialize git repo
git init

# Add remote (example)
git remote add origin <your-repo-url>

# Stage and commit
git add .
git commit -m "Initial boundary segmentation project"

# Push (when ready)
# git push -u origin main
```
