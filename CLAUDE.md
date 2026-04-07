# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This repo is the `boundry_segmentation` package (intentionally misspelled). When imported from the parent directory it is `from boundry_segmentation.qwen import ...`. The `__init__.py` at the repo root marks it as a package. All runnable scripts sit at the root; `extra/` holds non-core research scripts.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If Qwen models are gated:
huggingface-cli login
```

Verified stack: Python 3.12, torch 2.8, transformers 4.57. The environment lives **outside** the repo (e.g. `../envs/bseg`).

Required data: place `sherlock.mp4` in the repo root (not tracked by git).

## Running the Three Pipelines

Run commands from inside the repo root.

**UBoCo (contrastive kernel):**
```bash
python uboco_gebd.py sherlock.mp4 \
  --boundary_method peaks \
  --rtp_kernel_size 5 --rtp_min_length 50 --rtp_threshold_diff 0.3 \
  --rtp_max_depth 3 --rtp_max_boundaries 30 \
  --peaks_distance 30 --peaks_prominence 0.6 --peaks_max_boundaries 25 \
  --n_epochs 2 --end_time 60 --output_dir outputs/sanity_uboco
```

**Qwen3-VL segment (binary, fast):**
```bash
python qwen.py sherlock.mp4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --response-mode binary --end-time 10 \
  --output-dir outputs/sanity_qwen_segment
```

**Qwen3-Omni describer (first-window debug):**
```bash
python qwen_omni_describer.py sherlock.mp4 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --debug-save outputs/sanity_qwen_describer/debug_first_window \
  --end-time 8 --window-size 4 --stride 4 --sample-fps 1 \
  --output-dir outputs/sanity_qwen_describer
```

**Full sanity run (UBoCo + Qwen segment + eval):**
```bash
bash run_sherlock_sanity.sh
# Enable describer:
RUN_DESCRIBER=1 bash run_sherlock_sanity.sh
# Override model or interpreter:
QWEN_SEG_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct bash run_sherlock_sanity.sh
PYTHON=/path/to/python bash run_sherlock_sanity.sh
```

## Evaluation

```bash
python evaluate_boundaries.py <detected> <reference> --fps 25 --tolerances 5 10 15 --output out.json
```

- `<detected>`: JSON with `boundary_times` key, or plain TXT (one timestamp per line)
- `<reference>`: same formats, or `.xlsx`/`.xls` (use `--gt_column` to pick column; `--coarse-scenes` for scene-level GT)
- Pre-computed reference: `references/sherlock_reference_boundaries_from_ubeco.txt`
- Pre-computed metrics: `references/ubeco_eval_metrics_scenes.json`, `references/qwen3_eval_metrics_ws12_prompt.json`

## Tests

```bash
python -m pytest tests/
# or single file:
python -m unittest tests/test_qwen_binary_mode.py
```

The test in `tests/test_qwen_binary_mode.py` mocks the Qwen model/processor but requires the `Qwen/Qwen2.5-VL-7B-Instruct` **tokenizer** to be available (downloaded from Hugging Face). Tests skip automatically if unavailable. `test_video_samples_plot` also requires `sherlock.mp4`.

## Architecture

### Pipeline A — UBoCo (`uboco_gebd.py`)
Frozen ResNet-50 backbone extracts 2048-d frame features; a trainable 1D-CNN encoder (2048→512) is trained with contrastive loss over short windows. After training, pairwise cosine similarities are computed across the video and boundary detection is applied in one of two modes:
- `peaks`: scipy `find_peaks` on the similarity signal
- `rtp`: recursive temporal partitioning (depth-limited binary split)

Outputs: `boundary_times.txt` + `uboco_boundaries.json` + diagnostic plots.

### Pipeline B — Qwen3-VL segment (`qwen.py`)
Sliding-window inference with `AutoModelForVision2Seq`. Each window is sent as a binary question ("is there a boundary?"). In `binary` response mode the model returns **token-level logits** for `true`/`false` tokens; confidence = softmax probability of the `true` token. Metrics stored per window: `prob_true`, `prob_false`, `logit_margin`, `entropy`.

The class `QwenTemporalSegmenterFixed` owns model loading, frame sampling, and window iteration. `qwen_vl_utils.py` provides `process_vision_info` required by Qwen3-VL.

### Pipeline C — Qwen3-Omni describer (`qwen_omni_describer.py`)
Audio-visual understanding using `Qwen3OmniMoeForConditionalGeneration`. Extracts frames (OpenCV) and audio (librosa/soundfile, optionally via ffmpeg). Produces per-window scene descriptions + audio transcriptions in `video_understanding.json`. `qwen_omni_utils.py` provides `process_mm_info`.

### Evaluation (`evaluate_boundaries.py`)
Computes precision/recall/F1 at configurable frame tolerances. Supports detected boundaries from JSON or TXT and ground-truth from Excel, CSV, or TXT. The Excel path handles both fine-grained segment boundaries and coarse scene boundaries.
