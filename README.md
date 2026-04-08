# StitchWise: Disaster Analysis (Segmentation)

## Project Overview
This component of the StitchWise system is responsible for localized damage assessment and scene analysis. Using low-altitude drone imagery, this pipeline identifies and delineates disaster-affected regions to provide actionable intelligence for emergency responders.

### Primary Classes
* **Water:** Flood zones and standing water.
* **Buildings:** Major damage and total destruction categories.
* **Roads:** Blocked or compromised transportation routes.
* **Vehicles:** Localization for potential search and rescue targets.

---

## Setup & Installation

### 1. Environment Configuration
Create a Conda environment and install the required dependencies:

```bash
conda create -n stitchwise python=3.10
conda activate stitchwise
pip install -r requirements.txt
```

### 2. Kaggle API Credentials
To download the dataset automatically, ensure your `kaggle.json` file is located at `~/.kaggle/kaggle.json`.

---

## Data Pipeline
We utilize the RescueNet dataset, featuring high-resolution aerial imagery. Because the raw images are too large for standard GPU memory, we implement a metric-consistent tiling strategy.

### 1. Dataset Download
Use the Kaggle CLI to pull and unzip the raw imagery:

```bash
kaggle datasets download yaroslavchyrko/rescuenet -p data/raw/rescuenet --unzip
```

### 2. Pre-processing & Tiling
The preparation script converts grayscale semantic masks into YOLO-compatible polygons and slices the large images into 640x640 patches.

```bash
python scripts/prepare_rescuenet.py --mode segment --img-num 500 --dataset-path data/raw/rescuenet/RescueNet
```

---

## Model Training
We utilize the YOLOv8-seg architecture, optimized for local GPU constraints (NVIDIA RTX A500 4GB).

### Training Execution
Run the following command to begin the fine-tuning process:

```bash
python scripts/train.py
```

### Training Specifications
* **Base Model:** YOLOv8n-seg (Nano)
* **Input Resolution:** 640x640
* **Epochs:** 100
* **Batch Size:** 16

---

## Directory Structure
* `data/raw/`: Original RescueNet dataset files.
* `data/processed/`: Tiled images and polygon labels ready for YOLO.
* `models/checkpoints/`: Saved weights (`.pt` files) and training metrics.
* `scripts/`: Utility scripts for data preparation and training execution.
* `src/`: Source code for model architecture and inference integration.

---

## Current Status
* **Data Pipeline:** Verified and robust against RescueNet directory variations.
* **Baseline:** Successfully completed a 20-image smoke test with healthy loss convergence.
* **Production:** Scaling to 500-image training to improve rare-class detection (e.g., Building Major Damage).