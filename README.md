# StitchWise
**Metric-Accurate Disaster Region Mapping from Drone Imagery**

StitchWise takes a sequence of nadir (top-down) drone images, detects and segments disaster regions, stitches all images into a single georeferenced orthomosaic, and overlays the disaster regions with metric scale — so you can instantly measure exact distances on the result.

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Setup](#3-setup)
4. [Quick Start — GUI](#4-quick-start--gui)
5. [Quick Start — CLI](#5-quick-start--cli)
6. [Pipeline Stages in Detail](#6-pipeline-stages-in-detail)
7. [Configuration & Options](#7-configuration--options)
8. [Project Structure](#8-project-structure)
9. [Test Run Results](#9-test-run-results)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Overview

```
Drone images (JPG/PNG/TIFF)
        │
        ├─ Stage 1: Metric Scale ──────── GSD per image (m/px from EXIF + depth)
        │
        ├─ Stage 2: Disaster Detection ── YOLO v8 bounding boxes (5 classes)
        │         + Segmentation ──────── SAM pixel-perfect masks
        │
        ├─ Stage 3: Stitching ─────────── SIFT + RANSAC → global orthomosaic
        │
        └─ Stage 4: Overlay ───────────── Masks projected onto mosaic
                                          + scale bar + legend
                                          + click-to-measure distance tool
```

**Detected disaster classes:**
| Class | Colour |
|-------|--------|
| Water / Flooding | Blue |
| Building Major Damage | Orange |
| Building Total Destruction | Red |
| Road Blocked | Purple |
| Vehicle | Cyan |

---

## 2. Pipeline Architecture

```
run_pipeline.py          Unified orchestrator (all 4 stages)
app.py                   Streamlit GUI

src/
  metric_scale.py        GSD estimation (EXIF → depth fallback)
  exif_extractor.py      DJI XMP / standard EXIF parser
  depth_model.py         Depth Anything V2 wrapper
  visualize.py           Scale bar overlay

  stitchwise/            Image stitching library (from zhaochen branch)
    features.py          SIFT keypoint detection
    matching.py          KNN + Lowe ratio filter
    geometry.py          RANSAC homography
    warping.py           Perspective warp
    blending.py          Feather blend
    pipeline_pairwise.py End-to-end pairwise stitching

detection/
  train.py               YOLOv8 fine-tuning on RescueNet
  evaluate.py            Per-class metrics
  prepare_rescuenet.py   Dataset tiling + YOLO label conversion
  model/best.pt          Trained weights ← place here

segmentation/
  segment_sam.py         YOLO bbox → SAM pixel mask

scripts/                 Stitching pipeline scripts (from zhaochen branch)
  build_pair_graph.py    Evaluate pairwise SIFT matches
  solve_global_no_ba.py  Spanning-tree global pose solving
  render_global_no_ba.py Warp all images onto global canvas
  validate_global_no_ba.py Loop-closure quality check
```

---

## 3. Setup

### Prerequisites
- Python 3.10 – 3.13
- Windows / Linux / macOS
- GPU recommended for SAM inference (CPU works but is slower)

> **Windows users:** Python's Microsoft Store version has a MAX_PATH limitation that
> prevents PyTorch from installing normally. Create a venv at a short path:
> ```cmd
> python -m venv C:\sw_env
> C:\sw_env\Scripts\activate
> ```

### Install dependencies

```bash
# CPU-only (no CUDA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# With CUDA GPU (faster SAM / depth model inference)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Model weights

Place the trained YOLOv8 weights at:
```
detection/model/best.pt
```

The model was trained on [RescueNet](https://www.kaggle.com/datasets/aarafat27/rescuenet) aerial disaster imagery in two phases:
- Phase 1 (10 epochs): backbone frozen, detection head only
- Phase 2 (40 epochs): full fine-tune

Current weights: `v1/best.pt` (198 MB — full model, ~200 training epochs).

SAM weights (`sam_b.pt`, 357 MB) are auto-downloaded from Ultralytics on first run.

---

## 4. Quick Start — GUI

```bash
# Activate venv (Windows)
C:\sw_env\Scripts\activate

# Launch
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

### GUI walkthrough

**Sidebar:**
1. Set **Image directory** (local path) or upload files directly
2. Confirm **YOLO weights** path — green ✅ if found
3. Adjust confidence (default 0.25) and SAM model size
4. Set output directory (default `outputs/pipeline`)
5. Click **▶ Run Pipeline**

**Mosaic tab (main view):**
- Full interactive mosaic with disaster overlay (zoom/pan via Plotly)
- Disaster regions highlighted in class colours with legend
- **Distance tool** — click two points on the mosaic to get the real-world distance in metres
  - The GSD (m/px) is auto-computed from EXIF and stitching scale
  - You can manually override the GSD value if needed
- Download full-resolution disaster mosaic (JPEG)

**Per Image tab:**
- Browse each image: original + segmentation mask side-by-side
- Shows which disaster classes were detected in that frame

**GSD Stats tab:**
- GSD value per frame (cm/px)
- Timeline chart showing altitude/scale variation
- Download GSD CSV

### Distance measurement

1. Install the click plugin: `pip install streamlit-plotly-events`
2. Click point **A** on the mosaic, then point **B**
3. Distance is shown instantly: `Distance: X.XX m`

Without `streamlit-plotly-events`, use the manual coordinate entry box.

---

## 5. Quick Start — CLI

```bash
# Basic run (auto-detects YOLO weights at detection/model/best.pt)
python run_pipeline.py --image-dir data/aerial234 --ext .JPG

# Full options
python run_pipeline.py \
  --image-dir   data/my_images \
  --output-dir  outputs/my_run \
  --yolo-weights detection/model/best.pt \
  --conf        0.25 \
  --sam-model   sam_b.pt \
  --neighbor-offsets 1,2,3 \
  --scale-bar   50.0 \
  --ext         .JPG \
  --no-depth-fallback

# Metric scale only (no segmentation, no stitching)
python pipeline.py --data-dir data/aerial234 --max-images 20
```

**Output files:**
```
outputs/pipeline/
  disaster_mosaic.jpg          Full-resolution mosaic with disaster overlay
  disaster_mosaic_preview.jpg  Downscaled preview (≤2800px) for GUI
  pipeline_summary.json        Machine-readable result manifest

  metric_scale/
    gsd_results.csv            Per-frame GSD table

  masks/
    mask_001.png               Per-image semantic masks (0=bg, 1=water, …)

  seg_viz/
    viz_001.jpg                YOLO + SAM visualisation per image

  pair_graph/
    accepted_edges_*.json      Accepted stitching pairs with homographies

  global_no_ba/
    mosaic_no_ba.tif           Full-resolution plain mosaic
    global_poses.json          Per-image 3×3 homography to global canvas
    render_manifest.json       Canvas dimensions, render scale
```

---

## 6. Pipeline Stages in Detail

### Stage 1 — Metric Scale (GSD)

GSD (metres/pixel) tells you how much real-world ground each image pixel covers.

```
GSD = H / f_px
  H     = AGL altitude (metres)
  f_px  = focal_mm / sensor_width_mm × image_width_px
```

**Estimation priority:**
| Method | When used |
|--------|-----------|
| `exif_full` | Altitude + focal length both in EXIF/XMP ← most accurate |
| `exif_focal_depth_alt` | Focal in EXIF; altitude estimated via Depth Anything V2 |
| `exif_alt_depth_focal` | Altitude in EXIF; focal estimated via depth |
| `depth_full` | No EXIF at all; full depth estimation |

For DJI drones, `XMP drone-dji:RelativeAltitude` provides true AGL altitude.

### Stage 2 — Disaster Segmentation

**Two-stage pipeline:**
1. **YOLOv8** runs on the full image → bounding boxes around disaster regions
2. **SAM (Segment Anything Model)** refines each bounding box → pixel-perfect mask
3. Masks are saved as grayscale PNGs (pixel value = class ID + 1, 0 = background)

**Model:** Fine-tuned YOLOv8 on RescueNet (4,494 aerial images, 640×640 tiles).

To skip segmentation (e.g., no GPU, no weights): omit `--yolo-weights` or point to a non-existent path. The pipeline will produce the plain mosaic with scale bar only.

### Stage 3 — Image Stitching

**Algorithm:** SIFT feature matching + RANSAC homography + global spanning-tree pose solving (no bundle adjustment).

```
For each neighbouring image pair (offsets ±1, ±2, ±3):
  1. Extract SIFT keypoints + descriptors
  2. KNN match with Lowe ratio filter (0.75)
  3. Estimate homography via RANSAC
  4. Accept if: inliers ≥ 12, inlier ratio ≥ 0.10, good matches ≥ 30

Build connectivity graph → find main connected component
Solve global poses via spanning tree (anchor = most-central node)
Warp all images onto shared canvas using perspective transform
Blend with feather blending
```

**Tunable parameters** in `configs/stitching.yaml`:
```yaml
preprocess:
  resize_max_dim: 1600   # increase for higher-resolution mosaic (slower)
matching:
  ratio_test: 0.75       # lower = stricter matching
geometry:
  min_inliers: 20        # minimum inliers to accept a pair
```

### Stage 4 — Disaster Overlay

For each image in the mosaic:
1. Load its segmentation mask
2. Resize to the same dimensions used during stitching
3. Warp the mask using `H_to_anchor` (same homography as the image) → mask in global canvas space
4. Accumulate coloured overlay (class-specific RGB colours at 45% opacity)
5. Draw metric scale bar using the computed mosaic GSD
6. Draw class legend

**Distance measurement GSD** (for the preview image):
```
mosaic_gsd = mean_gsd_original / (image_resize_scale × render_scale × preview_scale)
```

---

## 7. Configuration & Options

### CLI options (`run_pipeline.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--image-dir` | *(required)* | Input image directory |
| `--output-dir` | `outputs/pipeline` | Where to save all results |
| `--yolo-weights` | `detection/model/best.pt` | YOLO model path |
| `--conf` | `0.25` | Detection confidence threshold |
| `--sam-model` | `sam_b.pt` | SAM variant: `mobile_sam.pt` (fast) / `sam_b.pt` / `sam_l.pt` (accurate) |
| `--neighbor-offsets` | `1,2,3` | Stitching: compare each frame to its ±N neighbours |
| `--scale-bar` | `10.0` | Scale bar length in metres |
| `--ext` | `.jpg` | Image file extension (case-sensitive on Linux) |
| `--no-depth-fallback` | off | Disable Depth Anything V2 (EXIF-only GSD) |

### Adding new images (incremental update)

Add new images to your image directory and re-run. The stitching step rebuilds the pair graph from scratch, incorporating the new frames. The disaster overlay is recomputed over the updated mosaic.

For the GUI: click **▶ Run Pipeline** again after adding new images.

### Retraining the detection model

```bash
# 1. Download and prepare RescueNet dataset
python detection/prepare_rescuenet.py --img-num 20 --tile --tile-size 640

# 2. Train (requires GPU, ~30 min for 50 epochs on RTX 3090)
python detection/train.py --epochs 50 --batch 32 --freeze-epochs 10

# 3. Evaluate
python detection/evaluate.py --weights outputs/runs/rescuenet_detect/weights/best.pt
```

---

## 8. Project Structure

```
StitchWise/
├── app.py                     Streamlit GUI
├── run_pipeline.py            Unified pipeline orchestrator
├── pipeline.py                Metric scale only (original)
├── requirements.txt
├── configs/
│   └── stitching.yaml         Stitching hyperparameters
│
├── src/                       Metric scale modules
│   ├── metric_scale.py
│   ├── exif_extractor.py
│   ├── depth_model.py
│   ├── visualize.py
│   ├── dataset.py
│   └── stitchwise/            Stitching library
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
│   ├── model/best.pt          ← trained YOLOv8 weights (place here)
│   ├── train.py
│   ├── evaluate.py
│   └── prepare_rescuenet.py
│
├── segmentation/
│   └── segment_sam.py
│
├── scripts/                   Stitching pipeline scripts
│   ├── build_pair_graph.py
│   ├── solve_global_no_ba.py
│   ├── render_global_no_ba.py
│   ├── validate_global_no_ba.py
│   └── run_global_no_ba.py
│
└── data/
    └── aerial234/             234-image Aerial234 dataset
```

---

## 9. Test Run Results

Tested on **15 images** from the Aerial234 dataset (DJI UAV, Southeast University campus area, ~90 m AGL).

### Environment
- Python 3.13, PyTorch 2.11 (CPU), Ultralytics 8.4.37
- YOLOv8 v1 weights (full training)
- SAM model: `sam_b.pt`

### Stage 1 — Metric Scale

All 15 images used `exif_full` method (DJI XMP altitude + EXIF focal length).

| Metric | Value |
|--------|-------|
| Mean GSD | **4.833 cm/px** |
| Min GSD | 4.795 cm/px (frames 5, 12–15) |
| Max GSD | 4.976 cm/px (frames 1, 3) |
| Altitude range | 89.9 – 93.3 m AGL |
| Method | `exif_full` (100% of frames) |

> At 4.8 cm/px, a 10-metre scale bar spans ≈208 pixels.
> Distance measurement precision: ≈5 cm per pixel.

### Stage 2 — Disaster Segmentation

Ran YOLO (conf=0.25) + SAM on all 15 images.

| Class | Images detected |
|-------|----------------|
| Water / Flooding | **12 / 15** |
| Building Major Damage | **2 / 15** |
| Vehicle | **1 / 15** |
| Road Blocked | 0 / 15 |
| Building Total Destruction | 0 / 15 |

Water/flooding was the dominant detected class, consistent with the dataset's
river/lake areas in the campus imagery.

### Stage 3 — Stitching

| Metric | Value |
|--------|-------|
| Candidate pairs evaluated | 39 (offsets 1, 2, 3) |
| Accepted pairs | 28 / 39 (72%) |
| Images placed in mosaic | **12 / 15** |
| Anchor image | `010.JPG` |
| Canvas size | 1818 × 3965 px |
| Render scale | 1.0 (no downscaling needed) |
| Preview size | 1284 × 2800 px |
| Mosaic coverage | **89.5%** |

Three images (001–003) did not connect to the main component — likely at the edge of the flight path where overlap with the main group was insufficient.

### Stage 4 — Disaster Mosaic

| Metric | Value |
|--------|-------|
| Full mosaic size | 1818 × 3965 px (≈2 MB JPEG) |
| Preview size | 1284 × 2800 px |
| Overlay opacity | 45% |
| Preview GSD | **6.84 cm/px** (accounts for render + preview downscale) |

At preview GSD of 6.84 cm/px, two clicks 100 pixels apart = **6.84 metres**.

### Output files produced

```
outputs/test_run/pipeline/
  disaster_mosaic.jpg          ✅ 1818×3965 px, 2.0 MB
  disaster_mosaic_preview.jpg  ✅ 1284×2800 px
  pipeline_summary.json        ✅
  metric_scale/gsd_results.csv ✅ 15 rows
  masks/mask_001.png … 015.png ✅ 15 masks
  seg_viz/viz_*.jpg            ✅ 15 YOLO+SAM visualisations
  pair_graph/                  ✅ 28 accepted pairs
  global_no_ba/
    mosaic_no_ba.tif           ✅
    global_poses.json          ✅ 12 nodes
    render_manifest.json       ✅
```

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'ultralytics'`
Install in the active Python environment:
```bash
pip install ultralytics
```
On Windows with the Store Python, create a venv at a short path first (see [Setup](#3-setup)).

### `FileNotFoundError: YOLO weights not found at detection/model/best.pt`
Place your `best.pt` at `detection/model/best.pt`. The pipeline will skip segmentation gracefully if the file is missing.

### `No images found in <dir>`
Check the `--ext` flag matches your actual file extension. Extensions are matched case-insensitively on Windows but case-sensitively on Linux.
```bash
# For .JPG files (uppercase):
python run_pipeline.py --image-dir data/aerial234 --ext .JPG
```

### `OSError: [Errno 2] … torch\include\ATen\…` on Windows
Windows MAX_PATH (260 chars) limit. Fix by creating a venv at `C:\sw_env`:
```cmd
python -m venv C:\sw_env
C:\sw_env\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Mosaic has gaps / disconnected images
Not enough feature overlap between consecutive frames. Try:
- Increasing `--neighbor-offsets` to `1,2,3,4,5`
- Lowering `--accept-min-inliers` in `build_pair_graph.py` (default 12)
- Ensuring images have ≥30% overlap

### Distance measurement is inaccurate
The auto-computed GSD assumes uniform altitude across all frames. If your drone varied altitude significantly, override the GSD manually in the GUI's **GSD (m/px)** field.

---

## Branch History

| Branch | Contribution |
|--------|-------------|
| `main` | Metric scale (GSD) estimation |
| `ruey-depth` | Depth Anything V2 integration |
| `Raphael` | YOLOv8 disaster detection training |
| `kane-segmentation-noML` | YOLO + SAM two-stage segmentation |
| `zhaochen-image-stitching` | SIFT/RANSAC global stitching pipeline |

All branches merged into `main` — the full integrated system lives here.

<!-- RESULTS:START -->

## Results — Test Run: 2026-04-13

### Summary Statistics

| Metric | Value |
|--------|-------|
| Frames processed | 15 |
| Images in mosaic | 12 |
| Mean GSD | 4.833 cm/px |
| Altitude range | 89.9 – 93.3 m AGL |
| Scale estimation method | `exif_full` (all frames) |
| Disaster detections | Water (12), Building damage (2), Vehicle (1) |
| Accepted stitching pairs | 28 / 39 |
| Mosaic coverage | 89.5% |
| Preview GSD (mosaic) | 6.84 cm/px |

<!-- RESULTS:END -->
