# Canonical Pipelines

This file lists the three supported pipelines for handoff and reproducibility.

## A) UBoCo Contrastive Boundary Pipeline

Script: `uboco_gebd.py`

Known-good Sherlock params (from `references/uboco_sherlock_params.txt`):

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
  --end_time 60 \
  --output_dir outputs/sanity_uboco
```

Expected outputs:

- `outputs/sanity_uboco/uboco_boundaries.json`
- `outputs/sanity_uboco/boundary_times.txt`

Primary reference metrics:

- `references/ubeco_eval_metrics_scenes.json`

## B) Qwen3 Segment Pipeline

Script: `qwen.py` (binary mode)

Final/high-capacity configuration (Qwen3-VL 30B A3B):

```bash
python qwen.py sherlock.mp4 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --response-mode binary \
  --sample-fps 25 \
  --window-size 8 \
  --stride 2 \
  --end-time 50 \
  --output-dir outputs/qwen3_30b_a3b_sherlock_tokenconfidence
```

Fast sanity alternative (Qwen3-VL 2B):

```bash
python qwen.py sherlock.mp4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --response-mode binary \
  --sample-fps 25 \
  --window-size 8 \
  --stride 1 \
  --end-time 50 \
  --output-dir outputs/qwen3_2b_sherlock_tokenconfidence
```

Expected outputs:

- `outputs/.../boundaries.json`
- `outputs/.../boundary_times.txt`

Primary reference metrics:

- `references/qwen3_eval_metrics_ws12_prompt.json`

## C) Qwen3 Describer Pipeline

Script: `qwen_omni_describer.py`

Debug-first-window command:

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

Expected outputs:

- `outputs/sanity_qwen_describer/video_understanding.json`
- `outputs/sanity_qwen_describer/debug_first_window/window_00000/` (debug assets)

Performance note:

- Qwen3-Omni-30B cold start is heavy (model load + first generation). A first debug window can take many minutes on shared hardware.

## Combined Sanity Run

Use:

```bash
bash run_sherlock_sanity.sh
```

Defaults:

- Qwen segment model defaults to `Qwen/Qwen3-VL-2B-Instruct` for speed.
- Describer is skipped by default (`RUN_DESCRIBER=0`).

Enable describer:

```bash
RUN_DESCRIBER=1 bash run_sherlock_sanity.sh
```
