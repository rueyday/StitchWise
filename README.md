# StitchWise
**Metric-Accurate Disaster Region Mapping from Drone Imagery**

StitchWise takes a sequence of nadir (top-down) drone images, detects and segments disaster regions using a fine-tuned YOLOv8 model + SAM, stitches all images into a growing georeferenced orthomosaic, and overlays disaster regions with metric scale — so you can instantly measure exact ground distances.

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Setup](#3-setup)
4. [Live GUI — Quick Start](#4-live-gui--quick-start)
5. [Live GUI — Full Guide](#5-live-gui--full-guide)
6. [Streamlit GUI](#6-streamlit-gui)
7. [CLI Pipeline](#7-cli-pipeline)
8. [Pipeline Stages in Detail](#8-pipeline-stages-in-detail)
9. [Configuration & Options](#9-configuration--options)
10. [Project Structure](#10-project-structure)
11. [Test Run Results](#11-test-run-results)
12. [Troubleshooting](#12-troubleshooting)
13. [Branch History](#13-branch-history)

---

## 1. System Overview

```
Drone images (JPG/PNG)
        │
        ├─ Stage 1: Metric Scale ──── GSD per image (m/px) from EXIF altitude + focal length
        │
        ├─ Stage 2: Detection ──────── YOLOv8 tiled inference → bounding boxes (4 classes)
        │         + Segmentation ───── SAM refines boxes → pixel-perfect masks
        │
        ├─ Stage 3: Stitching ──────── SIFT + RANSAC → global spanning-tree pose → orthomosaic
        │
        └─ Stage 4: Overlay ────────── Masks projected onto mosaic canvas
                                       + class colour overlay + live GUI
                                       + click-to-measure distance in metres
```

**Detected disaster classes (Raphael's YOLOv8, trained on RescueNet):**
| Class | Colour |
|-------|--------|
| Water / Flooding | Blue |
| Building Damaged | Red |
| Road Blocked | Orange |
| Vehicle | Green |

---

## 2. Pipeline Architecture

```
live_view.py             Standalone live GUI (recommended entry point)
run_pipeline.py          Batch pipeline orchestrator (all 4 stages)
app.py                   Streamlit GUI (browse results after pipeline run)

src/
  metric_scale.py        GSD estimation (EXIF altitude + focal length)
  exif_extractor.py      DJI XMP / standard EXIF parser
  depth_model.py         Depth Anything V2 fallback
  stitchwise/            Image stitching library (Zhaochen)
    features.py          SIFT keypoint detection
    matching.py          KNN + Lowe ratio filter
    geometry.py          RANSAC homography
    warping.py           Perspective warp
    blending.py          Feather blend

detection/
  predict.py             Tiled orthomosaic inference (Raphael)
  train.py               YOLOv8 two-phase fine-tuning
  evaluate.py            Per-class test metrics
  prepare_rescuenet.py   RescueNet dataset prep + YOLO label conversion
  model/best.pt          Final trained weights (rescuenet_v2, 49.6 MB)

segmentation/
  segment_sam.py         YOLO bbox → SAM pixel mask (Kane)

scripts/
  build_pair_graph.py    SIFT pairwise matching + graph construction
  solve_global_no_ba.py  Spanning-tree global pose solving
  render_global_no_ba.py Warp + blend all images onto global canvas
```

---

## 3. Setup

### Prerequisites
- Python 3.10 – 3.13
- Windows / Linux / macOS
- GPU recommended for SAM (CPU works but is slow — ~5-15s per frame)

> **Windows users:** The Microsoft Store Python has a MAX_PATH (260 char) limit that
> breaks PyTorch installation. Create your venv at a short path:
> ```cmd
> python -m venv C:\sw_env
> C:\sw_env\Scripts\activate
> ```

### Install dependencies

```bash
# CPU only (no CUDA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# With CUDA GPU (10-20x faster SAM + YOLO inference)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Model weights

The trained YOLOv8 weights are already included at `detection/model/best.pt` (49.6 MB, `rescuenet_v2` final).

**SAM weights** (`mobile_sam.pt`, ~39 MB) are downloaded automatically on first run from Ultralytics.

---

## 4. Live GUI — Quick Start

```bash
# Activate venv (Windows)
C:\sw_env\Scripts\activate

# Run with your image directory
python live_view.py --image-dir data/rescuenet_seq --ext .jpg

# Reuse cached stitching on repeat runs (much faster)
python live_view.py --image-dir data/rescuenet_seq --ext .jpg --reuse
```

The GUI window opens immediately. The map grows frame-by-frame as images are stitched in.

---

## 5. Live GUI — Full Guide

### What it does

`live_view.py` is a standalone Python popup that runs the complete pipeline in a background thread and shows results live as they come in:

1. **Estimates GSD** for each image from EXIF (altitude + focal length)
2. **Builds the stitch graph** — SIFT matches every image against its neighbours
3. **Solves global poses** — spanning-tree homographies to a shared canvas
4. For each frame added to the mosaic:
   - Runs **YOLO tiled detection** (Raphael's model) → bounding boxes
   - Passes boxes to **SAM** (Kane's model) → pixel-precise masks
   - **Warps** the image and its masks onto the growing mosaic canvas
   - **Composites** the disaster colour overlay at 45% opacity
   - Draws a **green outline** around the newest frame
   - Sends the updated mosaic to the GUI

### Running it

```bash
python live_view.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--image-dir` | *(required)* | Directory containing your input images |
| `--ext` | `.jpg` | File extension to look for (case-insensitive) |
| `--n-frames` | all | Limit to first N images |
| `--output-dir` | `outputs/live_view` | Where to cache stitching results |
| `--yolo-weights` | `detection/model/best.pt` | Path to YOLO weights |
| `--conf` | `0.25` | Detection confidence threshold (lower = more detections) |
| `--offsets` | `1,2,3` | Compare each frame against its ±N neighbours for stitching |
| `--reuse` | off | Skip rebuilding pair graph if it already exists |
| `--no-seg` | off | Skip YOLO + SAM (plain stitching only, much faster) |

**Examples:**
```bash
# First run — builds pair graph (slow), then shows map live
python live_view.py --image-dir data/rescuenet_seq --ext .jpg

# Repeat run — reuses cached poses, starts rendering immediately
python live_view.py --image-dir data/rescuenet_seq --ext .jpg --reuse

# Test with first 10 frames only
python live_view.py --image-dir data/rescuenet_seq --n-frames 10 --ext .jpg --reuse

# Plain stitching, no detection (fastest)
python live_view.py --image-dir data/rescuenet_seq --ext .jpg --no-seg --reuse

# Custom confidence + more neighbours
python live_view.py --image-dir data/rescuenet_seq --ext .jpg \
  --conf 0.3 --offsets 1,2,3,4,5 --reuse
```

### GUI controls

| Action | How |
|--------|-----|
| **Pan** | Left-click and drag |
| **Zoom in / out** | Mouse scroll wheel (zoom anchored at cursor) |
| **Fit to window** | Double-click anywhere, or click **Fit** button |
| **Measure distance** | Left-click point A, then left-click point B |
| **Clear measurement** | Click **Clear pts** button |

### Distance measurement

1. Left-click a point on the map — labelled **A** (green dot)
2. Left-click a second point — labelled **B** (red dot), yellow line drawn between them
3. Distance shown in the right panel in **metres**
4. Also shows pixel coordinates and cm/px value
5. Click anywhere again to start a new measurement (A resets)

### Right panel

- **Detected Classes** legend — colour swatches matching the overlay
- **Distance Tool** — current measurement result
- **GSD** — ground sampling distance in cm/px (shown at bottom)
- **Zoom %** — current zoom level

### Status bar (bottom)

Shows the current pipeline stage:
- `Estimating metric scale…` → computing GSD from EXIF
- `Building image pair graph…` → SIFT matching (slowest step, ~3-8 min for 15 large images without GPU)
- `Solving global poses…` → homography computation
- `Reusing existing global poses…` → cached, skipping straight to render
- `Loading YOLO model…` / `Loading SAM model…`
- `Frame N/M  <filename>` → actively rendering
- `✓ Done — N/M frames` → complete

### Tips

- **First run is slow** because SIFT matching runs on all image pairs. Use `--reuse` on every subsequent run — it skips straight to rendering frames.
- **Images must overlap** (~30%+) with their neighbours. Non-overlapping images are dropped from the mosaic.
- **Large images (4000×3000)** are resized to 1600px max dimension before matching and stitching. This is automatic.
- **No GPU?** Use `--no-seg` for plain stitching, or `mobile_sam.pt` is already selected automatically (fastest SAM variant).
- The **rescuenet_seq dataset** (`data/rescuenet_seq/`) is a verified 20-image consecutive sequence from the RescueNet train split — confirmed to stitch correctly (72+ SIFT matches between consecutive frames).

---

## 6. Streamlit GUI

The Streamlit GUI is for browsing results *after* a batch pipeline run.

```bash
C:\sw_env\Scripts\activate
streamlit run app.py
```

Browser opens at `http://localhost:8501`. Tabs:
- **Mosaic** — interactive mosaic with disaster overlay and distance tool
- **Live Stitching** — replay the mosaic build frame by frame
- **Per Image** — browse original + segmentation mask side by side
- **GSD Stats** — GSD timeline chart + CSV download

---

## 7. CLI Pipeline

```bash
# Full pipeline (metric scale + detection + stitching + overlay)
python run_pipeline.py --image-dir data/rescuenet_seq --ext .jpg

# All options
python run_pipeline.py \
  --image-dir   data/my_images \
  --output-dir  outputs/my_run \
  --yolo-weights detection/model/best.pt \
  --conf        0.25 \
  --sam-model   mobile_sam.pt \
  --neighbor-offsets 1,2,3 \
  --scale-bar   50.0 \
  --ext         .jpg
```

**Output files:**
```
outputs/pipeline/
  disaster_mosaic.jpg          Full-resolution mosaic with disaster overlay
  disaster_mosaic_preview.jpg  Downscaled preview
  pipeline_summary.json        Machine-readable result manifest
  metric_scale/gsd_results.csv Per-frame GSD table
  masks/mask_*.png             Per-image semantic masks
  seg_viz/viz_*.jpg            YOLO + SAM visualisation per image
  pair_graph/                  Accepted stitching pairs
  global_no_ba/
    global_poses.json          Per-image homographies to global canvas
    mosaic_no_ba.tif           Plain mosaic (no overlay)
```

---

## 8. Pipeline Stages in Detail

### Stage 1 — Metric Scale (GSD)

GSD (metres/pixel) = how much real-world ground each pixel covers.

```
GSD = altitude_m / focal_px
  altitude_m = AGL altitude from EXIF/XMP (DJI: drone-dji:RelativeAltitude)
  focal_px   = focal_mm / sensor_width_mm × image_width_px
```

**Estimation priority:**
| Method | When used |
|--------|-----------|
| `exif_full` | Altitude + focal length both in EXIF/XMP — most accurate |
| `exif_focal_depth_alt` | Focal in EXIF; altitude from Depth Anything V2 |
| `depth_full` | No EXIF; full depth estimation fallback |

### Stage 2 — Detection + Segmentation

**Two-stage pipeline per image:**
1. **YOLO tiled inference** (Raphael) — image sliced into 640×640 tiles with 64px overlap, each tile run through the model, tile-local boxes converted to full-image coords, cross-tile NMS applied
2. **SAM** (Kane) — each YOLO bounding box passed to MobileSAM as a prompt → pixel-precise mask returned

**Model details:** `rescuenet_v2` — YOLOv8 fine-tuned on RescueNet (post-Hurricane-Michael aerial imagery, 4,494 images at 3000×4000 px), two-phase training (backbone frozen → full fine-tune). 4 classes: water, building-damaged, road-blocked, vehicle.

### Stage 3 — Image Stitching (Zhaochen)

```
For each image pair (offsets ±1, ±2, ±3):
  1. Extract SIFT keypoints + descriptors (images resized to 1600px max)
  2. KNN match with Lowe ratio filter (0.75)
  3. Estimate homography via RANSAC
  4. Accept if: inliers ≥ 12, inlier ratio ≥ 0.10, good matches ≥ 30

Build connectivity graph → find main connected component
Spanning-tree global poses (anchor = most-central node)
Warp all images onto shared canvas → feather blend
```

### Stage 4 — Disaster Overlay

For each frame added to the mosaic:
1. Masks warped to mosaic canvas space using the same `H_to_anchor` homography
2. Per-class colour overlay blended at 45% opacity (`OVERLAY_ALPHA`)
3. Accumulated across all frames so earlier detections persist

**Mosaic GSD formula:**
```
mosaic_gsd_m_per_px = mean_gsd_original / render_scale
```

---

## 9. Configuration & Options

### `live_view.py` options

See [Section 5](#5-live-gui--full-guide) for full flag reference.

### `configs/stitching.yaml`

```yaml
preprocess:
  resize_max_dim: 1600     # increase for higher-res mosaic (slower matching)
matching:
  ratio_test: 0.75         # Lowe ratio — lower = stricter
geometry:
  min_inliers: 12          # minimum RANSAC inliers to accept a pair
  min_good_matches: 30
```

### Retraining the detection model

```bash
# 1. Prepare RescueNet dataset (requires Kaggle credentials)
python detection/prepare_rescuenet.py --tile --tile-size 640

# 2. Train (GPU recommended, ~30 min on RTX 3090 for 50 epochs)
python detection/train.py --epochs 50 --batch 32 --freeze-epochs 10

# 3. Evaluate on test split
python detection/evaluate.py --weights outputs/runs/rescuenet_v2/weights/best.pt
```

---

## 10. Project Structure

```
StitchWise/
├── live_view.py               ← Main entry point: live popup GUI
├── app.py                     Streamlit GUI (post-run browser)
├── run_pipeline.py            Batch pipeline orchestrator
├── requirements.txt
├── configs/
│   └── stitching.yaml
│
├── src/
│   ├── metric_scale.py
│   ├── exif_extractor.py
│   ├── depth_model.py
│   ├── visualize.py
│   ├── dataset.py
│   └── stitchwise/
│       ├── features.py
│       ├── matching.py
│       ├── geometry.py
│       ├── warping.py
│       ├── blending.py
│       ├── pipeline_pairwise.py
│       ├── config.py
│       ├── io_utils.py
│       └── debug_viz.py
│
├── detection/
│   ├── model/best.pt          Final YOLOv8 weights (rescuenet_v2, 49.6 MB)
│   ├── predict.py             Tiled orthomosaic inference
│   ├── train.py               Two-phase fine-tuning script
│   ├── evaluate.py            Test split evaluation
│   └── prepare_rescuenet.py   Dataset prep
│
├── segmentation/
│   └── segment_sam.py         YOLO + SAM two-stage segmentation
│
├── scripts/
│   ├── build_pair_graph.py
│   ├── solve_global_no_ba.py
│   ├── render_global_no_ba.py
│   └── validate_global_no_ba.py
│
└── data/
    ├── rescuenet_seq/         20-image consecutive RescueNet train sequence
    ├── rescuenet_test/        15 individual RescueNet test images
    ├── rescuenet_raw/         Full RescueNet zip (6.3 GB)
    └── aerial234/             Original Aerial234 dataset
```

---

## 11. Test Run Results

### Run 1 — Aerial234 dataset (non-disaster, stitching validation)

**15 images**, DJI UAV, Southeast University campus, ~90 m AGL.

| Metric | Value |
|--------|-------|
| Mean GSD | **4.833 cm/px** |
| Altitude range | 89.9 – 93.3 m AGL |
| GSD method | `exif_full` (all 15 frames) |
| Images placed | **12 / 15** |
| Accepted pairs | 28 / 39 (72%) |
| Canvas size | 1818 × 3965 px |
| Mosaic coverage | **89.5%** |
| Water detections | 12 / 15 frames |

---

### Run 2 — RescueNet train sequence (disaster imagery)

**15 images** (10870–10884), consecutive flight sequence from RescueNet train split.  
Post-Hurricane-Michael aerial survey, 4000×3000 px each.

**Environment:** Python 3.13, PyTorch CPU, YOLOv8 `rescuenet_v2` final weights (49.6 MB), MobileSAM.

#### Stage 1 — Metric Scale

RescueNet images lack EXIF altitude data — GSD estimated via depth fallback.

| Metric | Value |
|--------|-------|
| GSD method | `depth_full` (no EXIF altitude) |
| Mean GSD | ~5–8 cm/px (depth-estimated) |

#### Stage 2 — Detection + Segmentation (Raphael + Kane)

YOLO tiled inference (640px tiles, 64px overlap) + MobileSAM per frame.

| Class | Detected |
|-------|---------|
| Water / Flooding | Yes — flood water visible across multiple frames |
| Building Damaged | Yes — structural damage detected |
| Road Blocked | Yes |
| Vehicle | Yes |

Detections warped onto mosaic canvas and accumulated with class-specific colour overlay.

#### Stage 3 — Stitching (Zhaochen)

| Metric | Value |
|--------|-------|
| SIFT matches (frame 1→2) | **72 good matches** |
| Images placed | 15 / 15 (all connected) |
| Pair graph cached | `outputs/rescuenet_seq_live/` |
| Reuse with `--reuse` | Skips SIFT, renders immediately |

#### Stage 4 — Live GUI

- Map grows frame by frame with coloured disaster overlays
- Blue = water/flooding, Red = building damage, Orange = road blocked, Green = vehicle
- Pan, zoom, and distance measurement all functional
- `--reuse` flag enables instant startup on repeat runs

---

## 12. Troubleshooting

### `OSError: [Errno 2] … torch\include\ATen\…` on Windows
MAX_PATH limit. Fix:
```cmd
python -m venv C:\sw_env
C:\sw_env\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Window opens but map stays empty
The pair graph is being built (SIFT matching). Check the **status bar at the bottom** of the window for progress. On large images without GPU this takes 3–10 minutes. Once done, frames will start appearing.  
On repeat runs, always add `--reuse` to skip this step.

### Only 1–2 images appear in the mosaic
Images don't have enough visual overlap for SIFT to match them. This happens with non-sequential images (e.g. the RescueNet *test* split — individual curated samples, not a flight path). Use a consecutive flight sequence like `data/rescuenet_seq/`.

### `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

### `ImportError: DLL load failed while importing _C`
PyTorch DLL issue on Windows — use the `C:\sw_env` venv, not the system Python.

### SAM error falls back to bounding boxes
MobileSAM failed (usually a memory issue on large images). The pipeline automatically falls back to filled bounding boxes — detections are still shown, just less precise.

### Mosaic has gaps / disconnected images
- Increase `--offsets` to `1,2,3,4,5`
- Ensure images have ≥30% overlap
- Check images are sorted in flight order (filenames should be sequential)

### Distance measurement is inaccurate
GSD is estimated from EXIF or depth model. If your drone had significant altitude variation, the GSD will be approximate. Check the `GSD ≈ X cm/px` value shown at the bottom of the right panel.

---

## 13. Branch History

| Branch | Contributor | Contribution |
|--------|-------------|-------------|
| `main` | Ruey-Day | Metric scale (GSD) estimation, integration, GUI |
| `ruey-depth` | Ruey-Day | Depth Anything V2 fallback |
| `Raphael` | Raphael | YOLOv8 training, tiled inference (`predict.py`), final `rescuenet_v2` weights |
| `kane-segmentation-noML` | Kane | YOLO + SAM two-stage segmentation (`segment_sam.py`) |
| `zhaochen-image-stitching` | Zhaochen | SIFT/RANSAC global stitching pipeline |

All branches merged into `main`.
