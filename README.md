# StitchWise
A Metric-Accurate Orthomosaic Pipeline for Real-Time Disaster Response

## Overview

StitchWise processes drone nadir (top-down) image sequences from the [Aerial234 dataset](https://huggingface.co/datasets/RussRobin/Aerial234) and assigns a **Ground Sampling Distance (GSD)** — meters per pixel — to each frame. This provides the metric scale needed to build accurate orthomosaics.

## Dataset

**RussRobin/Aerial234** — 234 aerial images from a continuous UAV scan of Southeast University campus. Used in:
- *"Object-level Geometric Structure Preserving for Natural Image Stitching"* (AAAI 2025)
- UAV image stitching with orthograph estimation (JVCI 2023)

## Metric Scale Strategy

### Primary: EXIF/XMP Extrinsics

For a nadir-pointing camera, GSD (meters/pixel) is derived analytically:

```
GSD = H / f_px

where:
  H     = AGL altitude in meters
  f_px  = focal length in pixels = (focal_mm / sensor_width_mm) × image_width_px
```

Metadata sources, in priority order:

| Priority | Source | Field | Notes |
|----------|--------|-------|-------|
| 1 | XMP | `drone-dji:RelativeAltitude` | DJI drones; true AGL |
| 2 | XMP | `Camera:AboveGroundAltitude` | Generic UAV XMP |
| 3 | EXIF | `GPS GPSAltitude` | Sea-level altitude; less accurate |
| 4 | EXIF | `FocalLength` | mm; combined with sensor DB |
| 5 | EXIF | `FocalLengthIn35mmFilm` | 35mm equivalent → inferred sensor |

### Fallback: Depth Anything V2 (Metric Outdoor)

When EXIF is incomplete, [Depth Anything V2 Metric Outdoor Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf) provides **absolute depth in meters**. For a nadir view over flat terrain:

```
altitude_est = median(depth_map[center 50%])  ≈ camera AGL
```

This gives metric scale without any EXIF data.

### Fallback Decision Tree

```
EXIF altitude + focal length?  →  exif_full       (most accurate)
Focal length only?             →  exif_focal_depth_alt  (depth for altitude)
Altitude only?                 →  depth_full      (depth for scene width)
Neither?                       →  depth_full      (full depth estimation)
```

## Project Structure

```
StitchWise/
├── pipeline.py          Main processing pipeline
├── update_readme.py     Append run results to this README
├── src/
│   ├── dataset.py       HuggingFace dataset loader (Aerial234)
│   ├── exif_extractor.py  EXIF/XMP parsing → CameraParams
│   ├── metric_scale.py  GSD estimation (extrinsic + depth)
│   ├── depth_model.py   Depth Anything V2 wrapper
│   └── visualize.py     Scale bar overlay, depth colormap, GSD plot
├── data/aerial234/      Downloaded dataset (auto-created)
└── results/             Pipeline outputs
    ├── *_metric.jpg     Annotated frames with scale bar
    ├── *_depth.jpg      Depth map heatmap (when depth used)
    ├── results.csv      Per-frame GSD table
    ├── gsd_timeline.png GSD vs. frame index plot
    └── summary.txt      Human-readable run summary
```

## Quick Start

```bash
# Install dependencies
pip install datasets huggingface_hub exifread opencv-python torch torchvision transformers tqdm matplotlib

# Run the full pipeline (downloads dataset automatically)
python pipeline.py

# Process with forced altitude (if EXIF is missing)
python pipeline.py --altitude 80.0

# Disable depth fallback (EXIF-only mode)
python pipeline.py --no-depth-fallback

# Process a subset for testing
python pipeline.py --max-images 10 --scale-bar 5.0

# Update this README with the latest results
python update_readme.py
```

## Output Interpretation

| Column | Description |
|--------|-------------|
| `gsd_cm_per_px` | Centimeters per pixel at ground level |
| `altitude_m` | Estimated AGL altitude |
| `method` | Scale estimation method used |
| `source_altitude` | Which metadata field provided altitude |

**Typical GSD for UAV mapping:**
- 30–100 m altitude → 0.8–3.0 cm/px
- Sub-cm GSD requires very low altitude or large sensor

<!-- RESULTS:START -->

## Results — Last Run: 2026-04-08 01:48

### Summary Statistics

| Metric | Value |
|--------|-------|
| Frames processed | 235 |
| Failed frames | 0 |
| Mean GSD | 4.802 cm/px |
| Min GSD | 4.779 cm/px |
| Max GSD | 4.976 cm/px |
| Processing time | N/A s |
| Scale methods used | `exif_full` (235 frames) |

### Sample Frame Results (first 5)

| Frame | GSD (cm/px) | Altitude (m) | Method |
|-------|-------------|--------------|--------|
| `001.JPG` | 4.9760 | 93.30 | `exif_full` |
| `002.JPG` | 4.9653 | 93.10 | `exif_full` |
| `003.JPG` | 4.9760 | 93.30 | `exif_full` |
| `004.JPG` | 4.8000 | 90.00 | `exif_full` |
| `005.JPG` | 4.7947 | 89.90 | `exif_full` |

### GSD Timeline

![GSD Timeline](results/gsd_timeline.png)

<!-- RESULTS:END -->
