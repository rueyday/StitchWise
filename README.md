# RapidGeoStitch

Real-time disaster mapping from UAV imagery. RapidGeoStitch stitches consecutive drone frames into a georeferenced mosaic while segmenting disaster zones and overlaying them live as the mosaic builds.

![Alt Text](https://drive.google.com/uc?export=view&id=1_FRo-IeENbknfFpJ4ITxd_zMwTdFtmmc)

## Overview

RapidGeoStitch chains four modules into a single live GUI:

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Metric Scale | `src/metric_scale.py` | Estimates ground sampling distance (GSD, m/px) from EXIF data |
| 2. Detection | `detection/predict.py` | Tiled YOLOv8 inference on full-res images (640 px tiles, 64 px overlap) |
| 3. Segmentation | `segmentation/segment_cv.py` | Classical CV per-class masking inside YOLO boxes (no GPU required) |
| 4. Stitching | `src/stitchwise/` | SIFT feature matching → RANSAC homographies → global pose solve → warp mosaic |

The result is a pannable mosaic with color-coded disaster overlays and two interactive tools: distance measurement and A\* shortest-path routing that avoids all hazard zones.

## Project Structure

```
RapidGeoStitch/
├── live_view.py                 # Main entry point — GUI + pipeline orchestrator
├── detection/
│   ├── model/best.pt            # Final YOLOv8 weights (rescuenet_v2, 4 classes)
│   ├── predict.py               # Tiled inference + cross-tile NMS
│   ├── train.py                 # Training script
│   ├── prepare_rescuenet.py     # Dataset preparation for RescueNet
│   └── evaluate.py              # Evaluation script
│
├── segmentation/
│   └── segment_cv.py            # Classical CV segmenter
│
├── src/
│   ├── metric_scale.py          # GSD estimation from EXIF
│   ├── exif_extractor.py        # EXIF parser
│   ├── depth_model.py           # Depth Anything V2 fallback
│   └── stitchwise/              # Stitching library
│       ├── features.py          # SIFT extraction
│       ├── matching.py          # Feature matching
│       ├── geometry.py          # Homography estimation
│       ├── pipeline_pairwise.py # Pairwise registration
│       ├── blending.py          # Multi-band blending
│       ├── warping.py           # Perspective warp utilities
│       ├── io_utils.py          # Image loading / resize
│       └── config.py            # Config dataclass
│
├── scripts/
│   ├── build_pair_graph.py      # Build SIFT match graph from image directory
│   ├── solve_global_no_ba.py    # Solve global poses
│   ├── render_global_no_ba.py   # Render final mosaic from cached poses
│   ├── validate_global_no_ba.py # Validate reprojection errors
│   └── ...
│
├── configs/
│   └── stitching.yaml           # Stitching hyper-parameters
└── requirements.txt
```

## Disaster Classes

The model detects 4 classes matching the RescueNet dataset:

| ID | Class | Overlay Colour |
|----|-------|---------------|
| 0 | Water / Flooding | Blue |
| 1 | Building Damaged | Red |
| 2 | Road Blocked | Orange |
| 3 | Vehicle | Green |

## Setup

### Prerequisites

- Python 3.10+
- Windows / macOS / Linux
- GPU optional (CUDA or MPS auto-detected; CPU works fine)

### Install

```bash
python -m venv C:/sw_env          # Windows
python -m venv venv               # macOS/Linux

C:/sw_env/Scripts/activate        # Windows
source venv/bin/activate          # macOS/Linux

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

## Running the GUI

```bash
python live_view.py --image-dir data/rescuenet_big --ext .jpg
```

### All options

```
--image-dir   PATH    Directory of input images (required)
--ext         STR     File extension, e.g. .jpg or .JPG  [default: .jpg]
--n-frames    INT     Process only the first N frames
--output-dir  PATH    Cache and output directory  [default: outputs/live_view]
--yolo-weights PATH   Path to YOLOv8 weights  [default: detection/model/best.pt]
--conf        FLOAT   YOLO confidence threshold  [default: 0.25]
--offsets     STR     Stitching neighbour offsets  [default: 1,2,3]
--no-seg              Skip detection and segmentation (plain stitching)
--fresh               Delete cached stitching results and rebuild from scratch
```

## GUI Controls

| Action | Control |
|--------|---------|
| Pan | Click + drag |
| Zoom | Scroll wheel (cursor-anchored) |
| Fit to window | Double-click or **Fit** button |
| Switch tool | **Measure** / **Path** buttons |
| Measure distance | two points distance shown in meters |
| Find shortest path | two points shortest path in orange 
| Clear points | **Clear** button |

### Path Tool

The Path tool finds the shortest safe route between two clicked points using A\* search. If the destination is unreachable, the path terminates at the nearest accessible cell. Path distance is reported in meters using the estimated mosaic GSD.

## Output

Each run saves:

```
outputs/<run-name>/
├── pair_graph/              SIFT match graph
├── global_no_ba/
│   └── global_poses.json    Homographies for all frames
└── final_mosaic.jpg         Full disaster overlay mosaic (JPEG 95%)
```

## Training the Detection Model

The detection model was fine-tuned on [RescueNet](https://github.com/BinaLab/RescueNet-Challenge) (UAV post-disaster imagery, 4-class subset).

```bash
# 1. Prepare dataset
python detection/prepare_rescuenet.py --data-dir data/rescuenet_raw

# 2. Train
python detection/train.py --data configs/rescuenet.yaml --epochs 100

# 3. Evaluate
python detection/evaluate.py --weights detection/model/best.pt
```

Model Details
- Architecture: YOLOv8n (nano)
- Inference: tiled 640 px crops with 64 px overlap; cross-tile NMS via `torchvision.ops.batched_nms`

## Segmentation

Classical CV segmentation runs inside every YOLO bounding box

| Class | Method |
|-------|--------|
| Water | Otsu threshold on LAB L-channel + HSV hue mask (blue-green) |
| Building Damaged | GrabCut + ellipse morph-close (kernel 7) |
| Road Blocked | GrabCut + elongated kernel aligned to road axis |
| Vehicle | GrabCut + morph-open (noise removal) |

Large crops are downscaled to ≤150 px before GrabCut and upscaled back.

## Dataset

Tested on [RescueNet](https://github.com/BinaLab/RescueNet-Challenge) consecutive UAV sequences extracted from the training split (images 10850–10941). The test split is non-consecutive and unsuitable for stitching.
