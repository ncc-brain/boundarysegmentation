# Quickstart

Use this to setup this segmentation project on a RUB HPC

## 1) Login into an HPC Login node

Don't know how? Start on [our slab here](https://brainsinthewild.slab.com/posts/%F0%9F%9F%A8-0-introduction-to-hpc-how-to-get-access-2jtvo0ca).

All further steps are done on an HPC login node unless otherwise stated.

## 2) Load HPC modules

```bash
rub-deploy-spack-configs-2026
module load spack/2026
spack install miniforge3
```

## 3) Download this project onto HPC

```bash
cd ~/
mkdir projects
cd projects
git clone https://github.com/ncc-brain/boundarysegmentation.git
cd boundarysegmentation
```

## 4) Setup the environment

```bash
conda create -n boundarysegmentation python=3.12.12
conda init
```

**Relogin into your HPC** and continue with the following

```bash
conda activate boundarysegmentation
cd ~/projects/boundarysegmentation

pip install -r requirements.txt
pip install pillow qwen-vl-utils qwen_omni_utils ffmpeg
```

## 3) Transfer video files to an HPC

**From your local machine** run the following to upload a file to HPC

```bash
scp -i ~/.ssh/elysium -r -C local_path your_rub_id@login001.elysium.hpc.ruhr-uni-bochum.de:remote_path
```

Here local_path is a path to a file of interest on your local machine
remote_path is a path to a Folder in which this file will be stored on a remote machine

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
