"""
StitchWise — Two-Stage Segmentation (YOLOv8 + SAM)
===================================================
This script implements Path 2 of the pipeline:
  1. Runs the fine-tuned YOLOv8 model to detect disaster classes (bounding boxes).
  2. Passes those bounding boxes to the Segment Anything Model (SAM).
  3. SAM generates pixel-perfect masks for the orthomosaic pipeline.
  4. Saves visualizations and raw binary masks for downstream processing.

Usage:
    python segment_sam.py --source data/rescuenet_yolo/test/images/sample.jpg
    python segment_sam.py --source dir_of_images/ --conf 0.4

Requirements:
    pip install ultralytics opencv-python-headless numpy
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO, SAM

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "detection" / "model" / "best.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "runs" / "sam_segmentation"

def parse_args():
    parser = argparse.ArgumentParser(description="Two-Stage YOLO+SAM Segmentation")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image or directory of images to segment")
    parser.add_argument("--yolo-weights", type=str, default=str(DEFAULT_YOLO_WEIGHTS),
                        help="Path to your fine-tuned YOLOv8 detection weights")
    parser.add_argument("--sam-model", type=str, default="sam_b.pt",
                        help="SAM model size to use (mobile_sam.pt, sam_b.pt, sam_l.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO detection confidence threshold")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR),
                        help="Where to save masks and visualizations")
    return parser.parse_args()

def process_image(img_path: Path, yolo_model, sam_model, out_dir: Path, conf: float):
    print(f"Processing: {img_path.name}")
    
    # Stage 1: YOLO Detection
    yolo_results = yolo_model.predict(source=str(img_path), conf=conf, verbose=False)[0]
    
    # Check if anything was detected
    if len(yolo_results.boxes) == 0:
        print(f"  └─ No objects detected. Skipping SAM.")
        return

    # Extract boxes (xyxy format required by SAM) and classes
    boxes = yolo_results.boxes.xyxy.tolist()
    class_ids = yolo_results.boxes.cls.tolist()
    names = yolo_results.names

    print(f"  └─ Detected {len(boxes)} objects. Running SAM...")

    # Stage 2: SAM Segmentation prompted by YOLO boxes
    sam_results = sam_model.predict(source=str(img_path), bboxes=boxes, verbose=False)[0]

    # Stage 3: Export Data
    # 3a. Save visual composite (overlaying masks on the image)
    composite = sam_results.plot() 
    cv2.imwrite(str(out_dir / "visualizations" / f"viz_{img_path.name}"), composite)

    # 3b. Save raw masks for the orthomosaic pipeline
    if sam_results.masks is not None:
        mask_data = sam_results.masks.data.cpu().numpy()
        
        # Combine masks into a single 2D semantic map (or save individually)
        h, w = mask_data.shape[1:]
        semantic_map = np.zeros((h, w), dtype=np.uint8)

        for i, mask in enumerate(mask_data):
            cls_id = int(class_ids[i])
            # Assign the YOLO class ID to the mask pixels
            semantic_map[mask > 0.5] = cls_id 
        
        # Save as a grayscale PNG where pixel values correspond to class IDs
        cv2.imwrite(str(out_dir / "masks" / f"mask_{img_path.stem}.png"), semantic_map)
        print(f"  └─ Saved masks and visualization to {out_dir}")

def main():
    args = parse_args()
    source_path = Path(args.source)
    out_dir = Path(args.out_dir)
    
    # Create output directories
    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  StitchWise — Two-Stage Segmentation (YOLO + SAM)        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # Load Models
    print("[1/2] Loading Models...")
    if not Path(args.yolo_weights).exists():
        raise FileNotFoundError(f"YOLO weights not found at {args.yolo_weights}")
    
    yolo_model = YOLO(args.yolo_weights)
    print("  └─ YOLOv8 (Detection) loaded ✓")
    
    sam_model = SAM(args.sam_model)
    print(f"  └─ {args.sam_model} loaded ✓\n")

    # Gather images
    if source_path.is_file():
        images = [source_path]
    elif source_path.is_dir():
        images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    else:
        raise ValueError("Source must be a valid file or directory.")

    print(f"[2/2] Segmenting {len(images)} images...")
    for img in images:
        process_image(img, yolo_model, sam_model, out_dir, args.conf)
        
    print(f"\nDone. All outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()