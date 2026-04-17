"""
RapidGeoStitch — Classical CV segmentation (no extra model weights).

  Water        : Otsu on LAB L-channel + HSV hue mask
  Building     : GrabCut + ellipse morph-close
  Road blocked : GrabCut + elongated directional kernel
  Vehicle      : GrabCut + morph-open

Usage:
    python segment_cv.py --source path/to/image.jpg
    python segment_cv.py --source dir_of_images/ --conf 0.4
    python segment_cv.py --benchmark --source dir/ [--compare-sam] [--max-images 50]
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "detection" / "model" / "best.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "runs" / "cv_segmentation"

# ---------------------------------------------------------------------------
# CLASS METADATA
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "water",
    1: "building-major-damage",
    2: "building-total-destruction",
    3: "road-blocked",
    4: "vehicle",
}

CLASS_COLORS = {
    0: (180,  80,   0),
    1: (  0, 200, 255),
    2: (  0,   0, 220),
    3: (  0, 140, 255),
    4: (  0, 210,   0),
}

# ---------------------------------------------------------------------------
# HYPER-PARAMETERS
# ---------------------------------------------------------------------------
BOX_PAD_FRAC = 0.15
GRABCUT_ITER = 3
MIN_BOX_PX   = 12
MAX_GC_DIM   = 150


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _padded_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 pad_frac: float = BOX_PAD_FRAC):
    H, W = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px = max(int(bw * pad_frac), 4)
    py = max(int(bh * pad_frac), 4)
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(W, x2 + px)
    cy2 = min(H, y2 + py)
    crop = image[cy1:cy2, cx1:cx2]
    rect = (x1 - cx1, y1 - cy1, bw, bh)   # (x, y, w, h) — GrabCut format
    return crop, (cx1, cy1, cx2, cy2), rect


def _grabcut(crop: np.ndarray, rect: tuple, n_iter: int = GRABCUT_ITER) -> np.ndarray:
    rx, ry, rw, rh = rect
    ch, cw = crop.shape[:2]

    if rw < 3 or rh < 3 or rx < 0 or ry < 0 or rx + rw > cw or ry + rh > ch:
        fallback = np.zeros((ch, cw), np.uint8)
        fallback[ry:ry+rh, rx:rx+rw] = 1
        return fallback

    scale = 1.0
    max_dim = max(cw, ch)
    if max_dim > MAX_GC_DIM:
        scale = MAX_GC_DIM / max_dim

    if scale < 1.0:
        new_w, new_h = max(5, int(cw * scale)), max(5, int(ch * scale))
        crop_gc = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Scale the rect and clamp strictly to new bounds
        rx_s, ry_s = int(rx * scale), int(ry * scale)
        rw_s, rh_s = int(rw * scale), int(rh * scale)
        
        rx_s = max(0, min(rx_s, new_w - 3))
        ry_s = max(0, min(ry_s, new_h - 3))
        rw_s = max(3, min(rw_s, new_w - rx_s))
        rh_s = max(3, min(rh_s, new_h - ry_s))
        
        rect_gc = (rx_s, ry_s, rw_s, rh_s)
    else:
        crop_gc = crop
        rect_gc = rect

    gc_h, gc_w = crop_gc.shape[:2]

    try:
        gc_mask = np.zeros((gc_h, gc_w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(crop_gc, gc_mask, rect_gc, bgd, fgd, n_iter, cv2.GC_INIT_WITH_RECT)
        binary_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)
        if scale < 1.0:
            binary_mask = cv2.resize(binary_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
        return binary_mask

    except cv2.error:
        fallback = np.zeros((ch, cw), np.uint8)
        fallback[ry:ry+rh, rx:rx+rw] = 1
        return fallback


def _paste_mask(full_h: int, full_w: int,
                crop_mask: np.ndarray, crop_coords: tuple) -> np.ndarray:
    """Place a crop-sized binary mask back onto a full-image canvas."""
    cx1, cy1, cx2, cy2 = crop_coords
    out = np.zeros((full_h, full_w), np.uint8)
    out[cy1:cy2, cx1:cx2] = crop_mask
    return out


# ===========================================================================
# PER-CLASS SEGMENTERS
# ===========================================================================

def segment_water(image: np.ndarray, box: tuple) -> np.ndarray:
    x1, y1, x2, y2 = box
    H, W = image.shape[:2]
    crop, crop_coords, rect = _padded_crop(image, x1, y1, x2, y2)
    rx, ry, rw, rh = rect

    lab    = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    _, otsu = cv2.threshold(lab[:, :, 0], 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hsv      = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue_mask = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] > 25)).astype(np.uint8)

    combined = np.zeros(otsu.shape, np.uint8)
    combined[ry:ry+rh, rx:rx+rw] = otsu[ry:ry+rh, rx:rx+rw] | hue_mask[ry:ry+rh, rx:rx+rw]

    coverage = combined.sum() / max(rw * rh, 1)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    if 0.05 < coverage < 0.90:
        return _paste_mask(H, W, cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel), crop_coords)

    gc_mask = _grabcut(crop, rect)
    return _paste_mask(H, W, cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel), crop_coords)


def segment_building(image: np.ndarray, box: tuple, morph_size: int = 7) -> np.ndarray:
    x1, y1, x2, y2 = box
    H, W = image.shape[:2]

    if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
        mask = np.zeros((H, W), np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    crop, crop_coords, rect = _padded_crop(image, x1, y1, x2, y2)
    gc_mask = _grabcut(crop, rect)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
    return _paste_mask(H, W, gc_mask, crop_coords)


def segment_road(image: np.ndarray, box: tuple) -> np.ndarray:
    x1, y1, x2, y2 = box
    H, W = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1

    if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
        mask = np.zeros((H, W), np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    crop, crop_coords, rect = _padded_crop(image, x1, y1, x2, y2)
    gc_mask = _grabcut(crop, rect)

    if bw >= bh:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

    gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
    return _paste_mask(H, W, gc_mask, crop_coords)


def segment_vehicle(image: np.ndarray, box: tuple) -> np.ndarray:
    x1, y1, x2, y2 = box
    H, W = image.shape[:2]

    if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
        mask = np.zeros((H, W), np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    crop, crop_coords, rect = _padded_crop(image, x1, y1, x2, y2)
    gc_mask = _grabcut(crop, rect)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_OPEN, kernel)
    return _paste_mask(H, W, gc_mask, crop_coords)


# ===========================================================================
# PUBLIC API
# ===========================================================================

def segment(image: np.ndarray, boxes: list, class_ids: list) -> tuple:
    H, W = image.shape[:2]
    semantic_map = np.zeros((H, W), dtype=np.uint8)
    detected = np.zeros((H, W), dtype=bool)

    for box, cls_id in zip(boxes, class_ids):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = (int(v) for v in box)

        if x2 <= x1 or y2 <= y1:
            continue

        if cls_id == 0:
            mask = segment_water(image, (x1, y1, x2, y2))
        elif cls_id == 1:
            mask = segment_building(image, (x1, y1, x2, y2), morph_size=7)
        elif cls_id == 2:
            mask = segment_building(image, (x1, y1, x2, y2), morph_size=11)
        elif cls_id == 3:
            mask = segment_road(image, (x1, y1, x2, y2))
        elif cls_id == 4:
            mask = segment_vehicle(image, (x1, y1, x2, y2))
        else:
            continue

        semantic_map[mask > 0] = cls_id
        detected[mask > 0] = True

    return semantic_map, detected


def visualize(image: np.ndarray, semantic_map: np.ndarray, detected: np.ndarray,
              boxes: list, class_ids: list, confs: list) -> np.ndarray:
    vis = image.copy()
    overlay = image.copy()

    for cls_id, color in CLASS_COLORS.items():
        if cls_id == 0:
            region = (semantic_map == 0) & detected
        else:
            region = semantic_map == cls_id
        if not region.any():
            continue
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
# PIPELINE
# ===========================================================================

def process_image(img_path: Path, yolo_model, out_dir: Path,
                  conf: float) -> tuple:
    """
    Run the full two-stage pipeline on one image.

    Returns (semantic_map, elapsed_seconds) or (None, 0.0) if skipped.
    """
    print(f"Processing: {img_path.name}")

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  └─ Could not read image. Skipping.")
        return None, 0.0

    yolo_results = yolo_model.predict(source=str(img_path), conf=conf, verbose=False)[0]

    if len(yolo_results.boxes) == 0:
        print(f"  └─ No objects detected. Skipping.")
        return None, 0.0

    boxes = yolo_results.boxes.xyxy.tolist()
    class_ids = yolo_results.boxes.cls.tolist()
    confs = yolo_results.boxes.conf.tolist()
    print(f"  └─ {len(boxes)} detections — running CV segmentation...")

    t0 = time.perf_counter()
    semantic_map, detected = segment(image, boxes, class_ids)
    elapsed = time.perf_counter() - t0

    vis = visualize(image, semantic_map, detected, boxes, class_ids, confs)
    cv2.imwrite(str(out_dir / "visualizations" / f"viz_{img_path.name}"), vis)
    cv2.imwrite(str(out_dir / "masks" / f"mask_{img_path.stem}.png"), semantic_map)

    print(f"  └─ CV seg done in {elapsed * 1000:.1f} ms  →  {out_dir}")
    return semantic_map, elapsed


# ===========================================================================
# BENCHMARK
# ===========================================================================

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(intersection) / float(union) if union > 0 else float("nan")


def run_benchmark(args):
    """
    Measure CV segmentation speed and IoU against SAM or ground-truth masks.

    IoU source priority
    -------------------
    1. --sam-masks dir  (pre-computed SAM mask PNGs, fastest to compare)
    2. --compare-sam    (run SAM live during benchmark — slow but no pre-comp needed)
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

    sam_model = None
    if args.compare_sam:
        from ultralytics import SAM
        sam_model = SAM(args.sam_model)
        print(f"SAM model loaded: {args.sam_model}")

    sam_masks_dir = Path(args.sam_masks) if args.sam_masks else None

    cv_times, sam_times = [], []
    # per-class IoU lists; filled only when a reference mask is available
    per_class_iou_cv  = {c: [] for c in range(5)}   # CV vs reference
    per_class_iou_sam = {c: [] for c in range(5)}   # SAM vs reference (if live SAM)

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

        # --- CV timing ---
        t0 = time.perf_counter()
        cv_map, _ = segment(image, boxes, class_ids)
        cv_times.append(time.perf_counter() - t0)

        # --- SAM (live) timing + map ---
        sam_map = None
        if sam_model is not None:
            t0 = time.perf_counter()
            sam_res = sam_model.predict(source=str(img_path),
                                        bboxes=boxes, verbose=False)[0]
            sam_times.append(time.perf_counter() - t0)
            sam_map = np.zeros_like(cv_map)
            if sam_res.masks is not None:
                for i, m in enumerate(sam_res.masks.data.cpu().numpy()):
                    sam_map[m > 0.5] = int(class_ids[i])

        # --- Load pre-computed SAM masks if dir provided ---
        if sam_masks_dir is not None and sam_map is None:
            candidate = sam_masks_dir / f"mask_{img_path.stem}.png"
            if candidate.exists():
                loaded = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if loaded is not None and loaded.shape == cv_map.shape:
                    sam_map = loaded

        # --- Compute IoU per detected class ---
        if sam_map is not None:
            for cls_id in {int(c) for c in class_ids}:
                iou_cv = _iou(cv_map == cls_id, sam_map == cls_id)
                if not np.isnan(iou_cv):
                    per_class_iou_cv[cls_id].append(iou_cv)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    if cv_times:
        print(f"\nSegmentation speed  (excl. YOLO inference):")
        print(f"  CV   avg {np.mean(cv_times)*1000:6.1f} ms  "
              f"[{np.min(cv_times)*1000:.1f} – {np.max(cv_times)*1000:.1f} ms]")
        if sam_times:
            speedup = np.mean(sam_times) / np.mean(cv_times)
            print(f"  SAM  avg {np.mean(sam_times)*1000:6.1f} ms  "
                  f"[{np.min(sam_times)*1000:.1f} – {np.max(sam_times)*1000:.1f} ms]")
            print(f"  → CV is {speedup:.1f}× faster than SAM")

    ref_label = "SAM" if (sam_model or sam_masks_dir) else None
    if ref_label and any(per_class_iou_cv[c] for c in range(5)):
        print(f"\nIoU vs {ref_label} masks (per class):")
        for cls_id in range(5):
            ious = per_class_iou_cv[cls_id]
            if ious:
                print(f"  {CLASS_NAMES[cls_id]:30s}  "
                      f"{np.mean(ious):.3f}  (n={len(ious)})")
        all_ious = [v for lst in per_class_iou_cv.values() for v in lst]
        if all_ious:
            print(f"  {'mean IoU':30s}  {np.mean(all_ious):.3f}")

    print("=" * 60)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="StitchWise — Two-Stage Segmentation (YOLO + Traditional CV)"
    )

    # Shared
    p.add_argument("--source", type=str, required=True,
                   help="Image file or directory of images to segment")
    p.add_argument("--yolo-weights", type=str, default=str(DEFAULT_YOLO_WEIGHTS),
                   help="Path to fine-tuned YOLOv8 detection weights")
    p.add_argument("--conf", type=float, default=0.25,
                   help="YOLO detection confidence threshold")
    p.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR),
                   help="Output directory for masks and visualisations")

    # Benchmark-only
    p.add_argument("--benchmark", action="store_true",
                   help="Run in benchmark mode (speed + IoU evaluation)")
    p.add_argument("--compare-sam", action="store_true",
                   help="[benchmark] Also run live SAM inference and compare")
    p.add_argument("--sam-model", type=str, default="sam_b.pt",
                   help="[benchmark] SAM model to use if --compare-sam")
    p.add_argument("--sam-masks", type=str, default=None,
                   help="[benchmark] Dir of pre-computed SAM mask PNGs to compare against")
    p.add_argument("--max-images", type=int, default=50,
                   help="[benchmark] Cap number of images evaluated")

    return p.parse_args()


def main():
    args = parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  StitchWise — Two-Stage Segmentation (YOLO + CV)         ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    if args.benchmark:
        run_benchmark(args)
        return

    # ------------------------------------------------------------------
    # Normal inference mode
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    if not Path(args.yolo_weights).exists():
        raise FileNotFoundError(f"YOLO weights not found at {args.yolo_weights}")

    print("[1/2] Loading YOLO model...")
    yolo_model = YOLO(args.yolo_weights)
    print("  └─ YOLOv8 (Detection) loaded ✓\n")

    source_path = Path(args.source)
    if source_path.is_file():
        images = [source_path]
    elif source_path.is_dir():
        images = sorted(source_path.glob("*.jpg")) + sorted(source_path.glob("*.png"))
    else:
        raise ValueError("--source must be a valid image file or directory.")

    print(f"[2/2] Segmenting {len(images)} image(s)...")
    total_times = []
    for img in images:
        _, elapsed = process_image(img, yolo_model, out_dir, args.conf)
        if elapsed > 0:
            total_times.append(elapsed)

    if total_times:
        print(f"\nAvg CV segmentation time: {np.mean(total_times)*1000:.1f} ms/image")
    print(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()