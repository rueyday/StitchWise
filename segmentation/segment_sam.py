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
import time
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

# ---------------------------------------------------------------------------
# CLASS METADATA  (kept in sync with segment_cv.py)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "water",
    1: "building-major-damage",
    2: "building-total-destruction",
    3: "road-blocked",
    4: "vehicle",
}

# BGR palette — same as segment_cv.py
CLASS_COLORS = {
    0: (180, 80,   0),   # water                  → deep blue
    1: (0,   200, 255),  # building-major-damage   → yellow
    2: (0,   0,   220),  # building-total-destr.   → red
    3: (0,   140, 255),  # road-blocked            → orange
    4: (0,   210,   0),  # vehicle                 → green
}


def visualize(image: np.ndarray, sam_results, boxes: list,
              class_ids: list, confs: list) -> np.ndarray:
    """
    Build a visualization consistent with segment_cv.py:
      - translucent colour mask overlay per detected class
      - bounding box rectangles
      - labels formatted as  "class_name  conf"
    """
    vis = image.copy()
    overlay = image.copy()

    if sam_results.masks is not None:
        mask_data = sam_results.masks.data.cpu().numpy()
        for i, raw_mask in enumerate(mask_data):
            cls_id = int(class_ids[i])
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            # raw_mask may be smaller than the image; resize if needed
            mh, mw = raw_mask.shape
            ih, iw = image.shape[:2]
            if (mh, mw) != (ih, iw):
                raw_mask = cv2.resize(raw_mask, (iw, ih),
                                      interpolation=cv2.INTER_NEAREST)
            region = raw_mask > 0.5
            overlay[region] = color

    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

    for box, cls_id, conf in zip(boxes, class_ids, confs):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = (int(v) for v in box)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


# ===========================================================================
# BENCHMARK
# ===========================================================================

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(intersection) / float(union) if union > 0 else float("nan")


def _sam_map_from_result(sam_res, class_ids: list, ref_shape: tuple) -> np.ndarray:
    """Build a (H, W) semantic map from a SAM result, resizing masks if needed."""
    H, W = ref_shape
    sem = np.zeros((H, W), dtype=np.uint8)
    if sam_res.masks is None:
        return sem
    for i, m in enumerate(sam_res.masks.data.cpu().numpy()):
        mh, mw = m.shape
        if (mh, mw) != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        sem[m > 0.5] = int(class_ids[i])
    return sem


def run_benchmark(args):
    """
    Measure SAM segmentation speed and optionally compare IoU against CV masks.

    Reference mask source priority
    --------------------------------
    1. --cv-masks dir   (pre-computed CV mask PNGs — fastest, no extra inference)
    2. --compare-cv     (run CV segmentation live — needs no pre-comp, but slower)
    3. Neither          (speed-only benchmark)
    """
    source = Path(args.source)
    images = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))
    images = images[: args.max_images]

    if not images:
        print("No images found for benchmark.")
        return

    print(f"\nBenchmark: {len(images)} images")
    yolo_model = YOLO(args.yolo_weights)
    sam_model  = SAM(args.sam_model)
    print(f"  └─ Models loaded: YOLO + {args.sam_model}")

    if args.compare_cv:
        from segment_cv import segment as cv_segment
        print("  └─ CV segmenter loaded for live comparison")

    cv_masks_dir = Path(args.cv_masks) if args.cv_masks else None

    sam_times, cv_times = [], []
    per_class_iou = {c: [] for c in range(5)}   # SAM vs CV, per class

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        yolo_results = yolo_model.predict(source=str(img_path),
                                          conf=args.conf, verbose=False)[0]
        if len(yolo_results.boxes) == 0:
            continue

        boxes     = yolo_results.boxes.xyxy.tolist()
        class_ids = yolo_results.boxes.cls.tolist()
        H, W      = image.shape[:2]

        # --- SAM timing ---
        t0 = time.perf_counter()
        sam_res = sam_model.predict(source=str(img_path),
                                    bboxes=boxes, verbose=False)[0]
        sam_times.append(time.perf_counter() - t0)
        sam_map = _sam_map_from_result(sam_res, class_ids, (H, W))

        # --- CV reference: live ---
        cv_map = None
        if args.compare_cv:
            t0 = time.perf_counter()
            cv_map, _ = cv_segment(image, boxes, class_ids)
            cv_times.append(time.perf_counter() - t0)

        # --- CV reference: pre-computed masks ---
        if cv_masks_dir is not None and cv_map is None:
            candidate = cv_masks_dir / f"mask_{img_path.stem}.png"
            if candidate.exists():
                loaded = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if loaded is not None and loaded.shape == sam_map.shape:
                    cv_map = loaded

        # --- Per-class IoU (SAM vs CV) ---
        if cv_map is not None:
            for cls_id in {int(c) for c in class_ids}:
                iou = _iou(sam_map == cls_id, cv_map == cls_id)
                if not np.isnan(iou):
                    per_class_iou[cls_id].append(iou)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    if sam_times:
        print(f"\nSegmentation speed  (excl. YOLO inference):")
        print(f"  SAM  avg {np.mean(sam_times)*1000:6.1f} ms  "
              f"[{np.min(sam_times)*1000:.1f} – {np.max(sam_times)*1000:.1f} ms]")
        if cv_times:
            speedup = np.mean(sam_times) / np.mean(cv_times)
            print(f"  CV   avg {np.mean(cv_times)*1000:6.1f} ms  "
                  f"[{np.min(cv_times)*1000:.1f} – {np.max(cv_times)*1000:.1f} ms]")
            print(f"  → SAM is {speedup:.1f}× slower than CV")

    if any(per_class_iou[c] for c in range(5)):
        ref_label = "CV (live)" if args.compare_cv else "CV masks"
        print(f"\nIoU vs {ref_label} (per class):")
        for cls_id in range(5):
            ious = per_class_iou[cls_id]
            if ious:
                print(f"  {CLASS_NAMES[cls_id]:30s}  "
                      f"{np.mean(ious):.3f}  (n={len(ious)})")
        all_ious = [v for lst in per_class_iou.values() for v in lst]
        if all_ious:
            print(f"  {'mean IoU':30s}  {np.mean(all_ious):.3f}")

    print("=" * 60)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Two-Stage YOLO+SAM Segmentation")

    # Shared
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

    # Benchmark-only
    parser.add_argument("--benchmark", action="store_true",
                        help="Run in benchmark mode (speed + IoU evaluation)")
    parser.add_argument("--compare-cv", action="store_true",
                        help="[benchmark] Also run live CV segmentation and compare speed/IoU")
    parser.add_argument("--cv-masks", type=str, default=None,
                        help="[benchmark] Dir of pre-computed CV mask PNGs to compare IoU against")
    parser.add_argument("--max-images", type=int, default=50,
                        help="[benchmark] Cap number of images evaluated")

    return parser.parse_args()


def process_image(img_path: Path, yolo_model, sam_model, out_dir: Path, conf: float):
    print(f"Processing: {img_path.name}")

    # Stage 1: YOLO Detection
    yolo_results = yolo_model.predict(source=str(img_path), conf=conf, verbose=False)[0]

    if len(yolo_results.boxes) == 0:
        print(f"  └─ No objects detected. Skipping SAM.")
        return

    # Extract boxes (xyxy format required by SAM), classes, and confidences
    boxes     = yolo_results.boxes.xyxy.tolist()
    class_ids = yolo_results.boxes.cls.tolist()
    confs     = yolo_results.boxes.conf.tolist()

    print(f"  └─ Detected {len(boxes)} objects. Running SAM...")

    # Stage 2: SAM Segmentation prompted by YOLO boxes
    image = cv2.imread(str(img_path))
    sam_results = sam_model.predict(source=str(img_path), bboxes=boxes, verbose=False)[0]

    # Stage 3: Export Data
    # 3a. Save visual composite with class names + confidence labels
    composite = visualize(image, sam_results, boxes, class_ids, confs)
    cv2.imwrite(str(out_dir / "visualizations" / f"viz_{img_path.name}"), composite)

    # 3b. Save raw masks for the orthomosaic pipeline
    if sam_results.masks is not None:
        H, W = image.shape[:2]
        semantic_map = _sam_map_from_result(sam_results, class_ids, (H, W))
        cv2.imwrite(str(out_dir / "masks" / f"mask_{img_path.stem}.png"), semantic_map)
        print(f"  └─ Saved masks and visualization to {out_dir}")


def main():
    args = parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  StitchWise — Two-Stage Segmentation (YOLO + SAM)        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.benchmark:
        run_benchmark(args)
        return

    # ------------------------------------------------------------------
    # Normal inference mode
    # ------------------------------------------------------------------
    source_path = Path(args.source)
    out_dir = Path(args.out_dir)

    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    print("[1/2] Loading Models...")
    if not Path(args.yolo_weights).exists():
        raise FileNotFoundError(f"YOLO weights not found at {args.yolo_weights}")

    yolo_model = YOLO(args.yolo_weights)
    print("  └─ YOLOv8 (Detection) loaded ✓")

    sam_model = SAM(args.sam_model)
    print(f"  └─ {args.sam_model} loaded ✓\n")

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