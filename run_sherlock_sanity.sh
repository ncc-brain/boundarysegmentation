#!/usr/bin/env bash
# Run Sherlock sanity check: UBoCo + Qwen segment + Qwen describer, then evaluate against reference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
VIDEO="${VIDEO:-sherlock.mp4}"

# Output dirs (relative to project root)
UBOCO_OUT="outputs/sanity_uboco"
QWEN_SEG_OUT="outputs/sanity_qwen_segment"
QWEN_DESC_OUT="outputs/sanity_qwen_describer"
REF_DIR="outputs/sanity_eval"
REF_TXT_PREFERRED="references/sherlock_reference_boundaries_from_ubeco.txt"
REF_JSON="${REF_JSON:-outputs/captions_stride12/transfer boundaries/ubeco_sherlock.json}"
REF_TXT="$REF_DIR/sherlock_reference_boundaries.txt"

if [[ ! -f "$VIDEO" ]]; then
  echo "[ERROR] Video not found: $VIDEO"
  echo "Place sherlock.mp4 in $SCRIPT_DIR or set VIDEO=/path/to/sherlock.mp4"
  exit 1
fi

echo "=== 1. UBoCo short run ==="
"$PYTHON" uboco_gebd.py "$VIDEO" \
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
  --output_dir "$UBOCO_OUT"

echo ""
echo "=== 2. Qwen segment short run (binary) ==="
"$PYTHON" qwen.py "$VIDEO" \
  --response-mode binary \
  --end-time 10 \
  --output-dir "$QWEN_SEG_OUT"

echo ""
echo "=== 3. Qwen describer short debug run (first window only) ==="
"$PYTHON" qwen_omni_describer.py "$VIDEO" \
  --debug-save "$QWEN_DESC_OUT/debug_first_window" \
  --end-time 8 \
  --window-size 4 \
  --stride 4 \
  --sample-fps 1 \
  --output-dir "$QWEN_DESC_OUT"

echo ""
echo "=== 4. Reference boundaries for eval ==="
if [[ -f "$REF_TXT_PREFERRED" ]]; then
  echo "Using $REF_TXT_PREFERRED"
  REF_TXT="$REF_TXT_PREFERRED"
else
  echo "Extracting from $REF_JSON (fallback)"
  mkdir -p "$REF_DIR"
  if [[ ! -f "$REF_JSON" ]]; then
    echo "[ERROR] Reference file not found: $REF_TXT_PREFERRED"
    echo "        and fallback JSON not found: $REF_JSON"
    exit 1
  fi
  "$PYTHON" -c "
import json
with open('$REF_JSON') as f:
    d = json.load(f)
for t in d['boundary_times']:
    print(t)
" > "$REF_TXT"
fi

echo ""
echo "=== 5. Evaluate UBoCo and Qwen against reference ==="
mkdir -p "$REF_DIR"
"$PYTHON" evaluate_boundaries.py \
  "$UBOCO_OUT/boundary_times.txt" \
  "$REF_TXT" \
  --fps 25 \
  --tolerances 5 10 15 \
  --output "$REF_DIR/uboco_vs_reference.json"

"$PYTHON" evaluate_boundaries.py \
  "$QWEN_SEG_OUT/boundaries.json" \
  "$REF_TXT" \
  --fps 25 \
  --tolerances 5 10 15 \
  --output "$REF_DIR/qwen_vs_reference.json"

echo ""
echo "=== Done. Results in $REF_DIR ==="
