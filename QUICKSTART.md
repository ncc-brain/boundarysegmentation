# Quickstart

Use this for the fastest colleague handoff.

## 1) Clone and enter the project

```bash
git clone https://github.com/Cappl1/boundarysegmentation.git
cd boundarysegmentation/boundry_segmentation
```

> The folder is intentionally spelled `boundry_segmentation`.

## 2) Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Qwen model access is gated, authenticate with Hugging Face first:

```bash
huggingface-cli login
```

## 3) Add the Sherlock clip

Place the video at:

```bash
boundry_segmentation/sherlock.mp4
```

From inside `boundry_segmentation/`, this should exist:

```bash
ls sherlock.mp4
```

## 4) Run the one-command sanity pipeline

```bash
bash run_sherlock_sanity.sh
```

Defaults:

- Qwen segment uses `Qwen/Qwen3-VL-2B-Instruct` (fast sanity).
- Qwen describer is skipped (`RUN_DESCRIBER=0`).

To run describer too:

```bash
RUN_DESCRIBER=1 bash run_sherlock_sanity.sh
```

To use the final high-capacity Qwen3 segment model:

```bash
QWEN_SEG_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct bash run_sherlock_sanity.sh
```

If you want to force a specific Python interpreter:

```bash
PYTHON=/path/to/python bash run_sherlock_sanity.sh
```

## 5) Expected runtime (verified on this machine)

- UBoCo short run: ~22s
- Qwen3 segment short run: ~40s
- Qwen3 describer debug-first-window: ~25 min (cold load + first generation)

## 6) Where outputs go

- `outputs/sanity_uboco/`
- `outputs/sanity_qwen_segment/`
- `outputs/sanity_qwen_describer/`
- `outputs/sanity_eval/`

## 7) Optional ultra-fast smoke test

If you only want to test inference plumbing quickly:

```bash
python uboco_gebd.py sherlock.mp4 --end_time 30 --n_epochs 1 --output_dir outputs/smoke_uboco
python qwen.py sherlock.mp4 --model Qwen/Qwen3-VL-2B-Instruct --response-mode binary --end-time 10 --output-dir outputs/smoke_qwen
```

## Notes

- Sherlock spreadsheet annotations are included as `Sherlock_Segments_1000_NN_2017.xlsx`.
- Sanity eval uses `references/sherlock_reference_boundaries_from_ubeco.txt` by default for fast reproducibility.
- Full details and GEBD batch examples are in `README.md`.
