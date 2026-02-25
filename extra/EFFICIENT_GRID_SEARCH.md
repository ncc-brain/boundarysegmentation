# Efficient Grid Search

## Overview

`efficient_grid_search_eval.py` is a highly optimized version of the grid search that extracts video features only once and reuses them across all parameter combinations.

## Performance Comparison

### Old `grid_search_eval.py`:
```
For 4 layers × 3 n_pca × 3 fps × 3 states = 108 combinations:
- 108 video passes (SLOW!)
- 108 feature extractions
- 108 PCA computations
- 108 HMM fits

Estimated time: ~2-10 hours depending on video length
```

### New `efficient_grid_search_eval.py`:
```
For the same 108 combinations:
- 1 video pass (extract all frames, all layers)
- 12 PCA computations (4 layers × 3 n_pca values)
- 108 HMM fits (only part that varies with n_states)

Estimated time: ~10-30 minutes (10-100x faster!)
```

## Usage

The command-line interface is identical to the old version:

```bash
python efficient_grid_search_eval.py \
    path/to/video.mp4 \
    path/to/ground_truth.xlsx \
    --output_root efficient_grid_runs \
    --fps_samples 5 15 30 \
    --n_states_list 5 10 20 \
    --layers 1 5 9 11 \
    --n_pca_list 20 30 50 \
    --tolerances 1 3 5 10 15 30 60 \
    --start_time 123.0 \
    --end_time 1553.0 \
    --model facebook/dinov2-base \
    --batch_size 16
```

## How It Works

1. **Phase 1: Feature Extraction (one-time cost)**
   - Reads video once
   - Extracts features for ALL frames at ALL specified layers
   - Stores features in memory

2. **Phase 2: PCA (computed once per layer/n_pca combination)**
   - For each layer:
     - For each n_pca value:
       - Compute PCA on all frames
       - Store reduced features

3. **Phase 3: Fast Grid Search**
   - For each layer/n_pca/fps_sample/n_states combination:
     - Subsample pre-computed PCA features (instant)
     - Fit HMM (fast, ~1-5 seconds)
     - Detect boundaries
     - Evaluate metrics

## Output

Produces the same outputs as the old version:
- `grid_summary.csv` - CSV with all metrics across all parameter combinations
- Individual run folders with `boundaries.json` for each combination

## Memory Requirements

This script loads all features into memory, so it uses more RAM than the old version:
- Raw features: ~(n_frames × feature_dim × n_layers × 4 bytes)
- For a 1500-second video at 30 fps with 4 layers and 768-dim features: ~700 MB
- PCA features add: ~(n_frames × max_n_pca × n_layers × 4 bytes) ~50 MB

Most modern machines should handle this easily, but if you run into memory issues, you can:
- Reduce the number of layers tested at once
- Reduce batch_size during extraction (doesn't affect memory of stored features)
- Process one model at a time

## Recommended Workflow

1. Use `efficient_grid_search_eval.py` for large-scale grid searches
2. Use `grid_search_eval.py` only if memory is extremely limited
3. Use `extract_boundries.py` directly for single experiments with custom parameters
