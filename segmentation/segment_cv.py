"""
StitchWise — Two-Stage Segmentation (YOLOv8 + Traditional CV)
=============================================================
Drop-in replacement for segment_sam.py using classical computer vision instead
of SAM.  No extra model weights, no GPU required for segmentation.

Algorithm per class
-------------------
  water                    →  Otsu on LAB L-channel (dark flood water) + GrabCut fallback
  building-major-damage    →  GrabCut (rect init, 5 iter) + ellipse morph-close (k=7)
  building-total-destr.    →  GrabCut + aggressive ellipse morph-close (k=11)
  road-blocked             →  GrabCut + elongated rect kernel (road linearity)
  vehicle                  →  GrabCut + small morph-open (noise removal for tiny objects)

Output format (identical to segment_sam.py)
-------------------------------------------
  masks/mask_<stem>.png     —  grayscale PNG, pixel value = YOLO class ID (0-4)
  visualizations/viz_<name> —  colour-overlay + bounding-box annotation

Usage (identical CLI to segment_sam.py)
---------------------------------------
    python segment_cv.py --source data/rescuenet_yolo/test/images/sample.jpg
    python segment_cv.py --source dir_of_images/ --conf 0.4

Benchmark mode (speed + IoU vs SAM or GT masks)
------------------------------------------------
    python segment_cv.py --benchmark \\
        --source data/rescuenet_yolo/test/images/ \\
        [--compare-sam] [--sam-model sam_b.pt] \\
        [--sam-masks outputs/runs/sam_segmentation/masks/] \\
        [--max-images 50]
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

# BGR palette — matches common disaster-map conventions
CLASS_COLORS = {
    0: (180, 80,   0),   # water                  → deep blue
    1: (0,   200, 255),  # building-major-damage   → yellow
    2: (0,   0,   220),  # building-total-destr.   → red
    3: (0,   140, 255),  # road-blocked            → orange
    4: (0,   210,   0),  # vehicle                 → green
}

# ---------------------------------------------------------------------------
# HYPER-PARAMETERS
# ---------------------------------------------------------------------------
BOX_PAD_FRAC  = 0.15   # padding added around YOLO box before GrabCut (% of box side)
GRABCUT_ITER  = 5      # GrabCut EM iterations (5 is standard; more = slower, marginal gain)
MIN_BOX_PX    = 12     # boxes smaller than this skip GrabCut and use filled rect fallback


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _padded_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 pad_frac: float = BOX_PAD_FRAC):
    """
    Extract a padded crop around the YOLO bounding box.

    Returns
    -------
    crop        : np.ndarray  — the cropped sub-image
    crop_coords : (cx1,cy1,cx2,cy2) in full-image space
    rect        : (rx, ry, rw, rh)  — the original box in crop-local coords
                  used directly as the GrabCut rect
    """
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
    """
    Run GrabCut on *crop* initialised with *rect* = (x, y, w, h) in crop coords.

    Returns a binary mask (uint8, same H×W as crop): 1 = foreground, 0 = background.
    Falls back to a filled-rect mask if GrabCut raises (too-small image, degenerate box).
    """
    rx, ry, rw, rh = rect
    ch, cw = crop.shape[:2]

    # Guard: GrabCut needs the rect fully inside the crop and min 3×3
    if rw < 3 or rh < 3 or rx < 0 or ry < 0 or rx + rw > cw or ry + rh > ch:
        fallback = np.zeros((ch, cw), np.uint8)
        fallback[ry:ry+rh, rx:rx+rw] = 1
        return fallback

    try:
        gc_mask = np.zeros((ch, cw), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, gc_mask, rect, bgd, fgd, n_iter, cv2.GC_INIT_WITH_RECT)
        return np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)
    except cv2.error:
        # Degenerate case — return filled bounding rect
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
    """
    Otsu threshold on the LAB L-channel (flood water is dark) inside the box,
    with a GrabCut fallback when Otsu produces a degenerate result.

    Why LAB?  L (lightness) is perceptually uniform and decouples brightness
    from colour, so dark water stands apart from bright land regardless of RGB
    hue shifts caused by atmospheric haze or sensor response.
    """
    x1, y1, x2, y2 = box
    H, W = image.shape[:2]
    crop, crop_coords, rect = _padded_crop(image, x1, y1, x2, y2)

    # --- Otsu on L channel (inverted: water is dark → bright in INV mask) ---
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    _, otsu = cv2.threshold(l_chan, 0, 1,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Restrict to the actual box region within the padded crop
    rx, ry, rw, rh = rect
    box_mask = np.zeros_like(otsu)
    box_mask[ry:ry+rh, rx:rx+rw] = otsu[ry:ry+rh, rx:rx+rw]

    coverage = box_mask.sum() / max(rw * rh, 1)

    if 0.05 < coverage < 0.90:
        # Plausible Otsu result — close small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_CLOSE, kernel)
        return _paste_mask(H, W, box_mask, crop_coords)

    # --- Fallback: GrabCut ---
    gc_mask = _grabcut(crop, rect)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
    return _paste_mask(H, W, gc_mask, crop_coords)


def segment_building(image: np.ndarray, box: tuple,
                     morph_size: int = 7) -> np.ndarray:
    """
    GrabCut for building damage classes.

    morph_size=7  for major-damage (mostly intact roof structure)
    morph_size=11 for total-destruction (fragmented rubble needs larger close)
    """
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
    """
    GrabCut + elongated morphological kernel aligned with the dominant box axis.

    Roads appear as thin linear features in nadir imagery; an isotropic kernel
    would break the linearity of the mask.  We choose orientation based on
    whether the box is landscape (horizontal road) or portrait (vertical road).
    """
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
    """
    GrabCut for small, compact vehicle objects.

    Uses morph-open (not close) to remove salt-and-pepper noise without
    dilating tiny vehicle blobs outward.
    """
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

def segment(image: np.ndarray, boxes: list, class_ids: list) -> np.ndarray:
    """
    Segment all detected objects and return a semantic map.

    Parameters
    ----------
    image     : H×W×3 BGR image (as returned by cv2.imread)
    boxes     : list of [x1,y1,x2,y2] pixel coords (YOLO xyxy format)
    class_ids : list of int class IDs (0-4)

    Returns
    -------
    semantic_map : np.ndarray uint8, shape (H, W)
        Pixel value = YOLO class ID.
        NOTE: class 0 (water) and unmasked background are both value 0 — same
        convention as segment_sam.py; downstream code should intersect with
        YOLO boxes to distinguish them if needed.
    """
    H, W = image.shape[:2]
    semantic_map = np.zeros((H, W), dtype=np.uint8)

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

    return semantic_map


def visualize(image: np.ndarray, semantic_map: np.ndarray,
              boxes: list, class_ids: list) -> np.ndarray:
    """Return a BGR image with translucent mask overlays and box annotations."""
    vis = image.copy()
    overlay = image.copy()

    for cls_id, color in CLASS_COLORS.items():
        region = semantic_map == cls_id
        # class 0 mask regions are inside a detected box — don't paint entire background
        if cls_id == 0:
            box_canvas = np.zeros(semantic_map.shape, np.uint8)
            for box, cid in zip(boxes, class_ids):
                if int(cid) == 0:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    box_canvas[y1:y2, x1:x2] = 1
            region = region & (box_canvas > 0)
        if not region.any():
            continue
        overlay[region] = color

    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

    for box, cls_id in zip(boxes, class_ids):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = (int(v) for v in box)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cls_id, str(cls_id))
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
    print(f"  └─ {len(boxes)} detections — running CV segmentation...")

    t0 = time.perf_counter()
    semantic_map = segment(image, boxes, class_ids)
    elapsed = time.perf_counter() - t0

    vis = visualize(image, semantic_map, boxes, class_ids)
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
        cv_map = segment(image, boxes, class_ids)
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
