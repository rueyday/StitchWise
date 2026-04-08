# StitchWise

StitchWise is the image stitching module for the **Disaster View** course project.
It focuses on building a reliable aerial stitching pipeline for low-altitude, downward-facing drone imagery.

This repository currently implements:
- pairwise matching and stitching with OpenCV (SIFT + KNN + Lowe ratio + RANSAC)
- batch pairwise diagnostics and parameter sweep
- pair connectivity graph construction with automatic edge rejection
- global pose initialization from accepted pair graph edges (no bundle adjustment)
- global mosaic rendering and validation reports

## Scope and Status

Implemented in this repo:
- Pairwise stitching and debugging tools
- Pair graph generation and filtering
- Global no-BA stitching (single mosaic output)

Not implemented yet:
- Bundle adjustment
- Lens distortion calibration/correction from known intrinsics
- Advanced seam finding / exposure compensation / multiband blending

## Repository Structure

```text
StitchWise/
├─ configs/
│  └─ stitching.yaml
├─ data/raw/Aerial234/
├─ scripts/
│  ├─ check_dataset.py
│  ├─ run_pairwise.py
│  ├─ run_pairwise_batch.py
│  ├─ run_pairwise_sweep.py
│  ├─ build_pair_graph.py
│  ├─ solve_global_no_ba.py
│  ├─ render_global_no_ba.py
│  ├─ validate_global_no_ba.py
│  └─ run_global_no_ba.py
├─ src/stitchwise/
│  ├─ config.py
│  ├─ io_utils.py
│  ├─ features.py
│  ├─ matching.py
│  ├─ geometry.py
│  ├─ warping.py
│  ├─ blending.py
│  ├─ debug_viz.py
│  └─ pipeline_pairwise.py
└─ outputs/
```

## Environment Setup

```bash
python -m pip install -r requirements.txt
```

## Step-by-Step Workflow

### 1) Dataset inspection

Check numeric ordering, missing indices, and neighboring candidates:

```bash
python scripts/check_dataset.py
```

### 2) Single pairwise run

Run one pair and save debug outputs:

```bash
python scripts/run_pairwise.py --image1 010.JPG --image2 011.JPG
```

Outputs go to:
- `outputs/pairwise/<image1_stem>_<image2_stem>/`
- includes `raw_matches.jpg`, `good_matches.jpg`, `inlier_matches.jpg`, `stitched.jpg`, `stats.json`

### 3) Batch pairwise over neighboring offsets

Evaluate local pairs in bulk (supports `(i, i+k)`):

```bash
python scripts/run_pairwise_batch.py --start-index 1 --end-index 236 --neighbor-offset 1
python scripts/run_pairwise_batch.py --start-index 1 --end-index 236 --neighbor-offset 2
python scripts/run_pairwise_batch.py --start-index 1 --end-index 236 --neighbor-offset 3
```

Outputs go to:
- `outputs/pairwise_batch/offset_1/`
- `outputs/pairwise_batch/offset_2/`
- `outputs/pairwise_batch/offset_3/`

Each offset folder includes `summary.csv` and `summary.json`.

### 4) Parameter sweep on a difficult pair

Use a small grid to test robustness:

```bash
python scripts/run_pairwise_sweep.py --image1 003.JPG --image2 004.JPG
```

Outputs:
- `outputs/pairwise_sweep/summary.csv`
- `outputs/pairwise_sweep/summary.json`
- per-combination debug folders

### 5) Build pair graph from local candidates

Build graph edges for offsets 1,2,3 with rejection rules:

```bash
python scripts/build_pair_graph.py --neighbor-offsets 1,2,3 --output-dir outputs/pair_graph
```

Main outputs:
- `outputs/pair_graph/pair_graph_edges.csv|json`
- `outputs/pair_graph/accepted_edges.csv|json`
- `outputs/pair_graph/rejected_edges.csv|json`
- `outputs/pair_graph/connected_components.json`
- `outputs/pair_graph/graph_summary.md`
- `outputs/pair_graph/main_component_nodes.json`
- `outputs/pair_graph/accepted_edges_main_component_h.json`

### 6) Solve global transforms (no BA)

Compute global image poses from accepted edges in the main component:

```bash
python scripts/solve_global_no_ba.py \
  --pair-graph-dir outputs/pair_graph \
  --output-dir outputs/global_no_ba
```

Outputs:
- `outputs/global_no_ba/global_poses.json`
- `outputs/global_no_ba/tree_edges.csv`
- `outputs/global_no_ba/dropped_edges.csv`
- `outputs/global_no_ba/dropped_nodes.csv`

### 7) Render global mosaic (no BA)

Render one global stitched image from solved poses:

```bash
python scripts/render_global_no_ba.py \
  --config configs/stitching.yaml \
  --poses-json outputs/global_no_ba/global_poses.json \
  --output-dir outputs/global_no_ba
```

Outputs:
- `outputs/global_no_ba/mosaic_no_ba.tif`
- `outputs/global_no_ba/mosaic_no_ba_preview.jpg`
- `outputs/global_no_ba/mosaic_alpha.png`
- `outputs/global_no_ba/render_manifest.json`

### 8) Validate global no-BA result

Generate quality diagnostics:

```bash
python scripts/validate_global_no_ba.py \
  --pair-graph-dir outputs/pair_graph \
  --global-dir outputs/global_no_ba
```

Outputs:
- `outputs/global_no_ba/loop_residuals.csv`
- `outputs/global_no_ba/validation_report.json`
- `outputs/global_no_ba/validation_report.md`

### 9) One-command run (recommended)

If pair graph is already built:

```bash
python scripts/run_global_no_ba.py
```

If you want to rebuild pair graph first:

```bash
python scripts/run_global_no_ba.py --run-build-graph --neighbor-offsets 1,2,3
```

## Rejection and Safety Rules

In graph building, an edge is rejected if:
- `inlier_count` is too low
- `inlier_ratio` is too low
- `good_match_count` is too low
- homography/canvas is clearly invalid (degenerate or exploding warp)

In global rendering, the script automatically rescales canvas when needed to prevent memory explosion.

## Practical Notes

- Aerial234 in this setup has one missing index (`118.JPG`), which is expected in the local copy.
- The current graph typically forms one large connected component and one small isolated early segment.
- The no-BA global result is a practical baseline for integration with downstream detection/segmentation modules.

## Next Improvements (Optional)

- Add bundle adjustment for global consistency
- Add intrinsic-based distortion correction
- Add stronger blending/seam handling for visual quality
