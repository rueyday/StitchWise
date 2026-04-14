"""
StitchWise — Tiled Inference on Orthomosaics
=============================================
Runs a fine-tuned YOLOv8 model over a large orthomosaic image using a tiled
inference strategy:

  1. Slice the orthomosaic into overlapping 640 × 640 px tiles
     (identical boundary logic to prepare_rescuenet.py — edge tiles are always
     full-size, shifted back rather than padded or truncated).
  2. Run model.predict() on each tile; collect all detections.
  3. Convert tile-local box coordinates to full-image coordinates.
  4. Apply per-class NMS across the whole image to remove duplicate detections
     that arise where adjacent tiles overlap.
  5. Draw semi-transparent filled boxes, label chips, and a class legend onto
     the full orthomosaic and write it to outputs/detections/.

This script is the final integration point between the stitching pipeline
(outputs an orthomosaic) and the detection component.

Usage:
    python predict.py --input outputs/orthomosaics/scene_01.jpg
    python predict.py --input outputs/orthomosaics/scene_01.jpg --conf 0.3 --save-tiles
    python predict.py --input outputs/orthomosaics/scene_01.jpg --no-merge

Requirements:
    pip install ultralytics opencv-python-headless torch torchvision tqdm
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PATHS  (resolved relative to this script, so they work from any cwd)
# ---------------------------------------------------------------------------

SCRIPT_DIR     = Path(__file__).parent.resolve()
REPO_ROOT      = SCRIPT_DIR.parent
RUNS_DIR       = REPO_ROOT / "outputs" / "runs"
DETECTIONS_DIR = REPO_ROOT / "outputs" / "detections"

# Active class names — must match the order produced by prepare_rescuenet.py
YOLO_CLASS_NAMES = ["water", "building-damaged", "road-blocked", "vehicle"]

# Per-class bounding box colors in BGR (OpenCV convention)
#   water          → blue
#   building-damaged → red
#   road-blocked   → orange
#   vehicle        → green
CLASS_COLORS = {
    0: (210,  90,  20),
    1: ( 20,  20, 210),
    2: (  0, 140, 255),
    3: ( 20, 185,  20),
}

# Total number of pipeline steps shown in [N/STEPS] indicators
N_STEPS = 5


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _default_weights = str(RUNS_DIR / "rescuenet_v2" / "weights" / "best.pt")
    parser = argparse.ArgumentParser(
        description="Run tiled YOLOv8 inference on a large orthomosaic image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input orthomosaic (or any aerial image)")
    parser.add_argument("--weights", type=str, default=_default_weights,
                        help="Path to the .pt checkpoint to use for inference")
    parser.add_argument("--output-dir", type=str, default=str(DETECTIONS_DIR),
                        help="Directory to save the annotated output image")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: GPU index (e.g. '0') or 'cpu'")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Inference tile size in pixels")
    parser.add_argument("--overlap", type=int, default=64,
                        help="Tile overlap in pixels — prevents missing objects "
                             "at tile borders")
    parser.add_argument("--save-tiles", action="store_true",
                        help="Save individually annotated tiles to "
                             "{output-dir}/tiles/ for debugging")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip cross-tile NMS (keeps all raw tile detections; "
                             "useful for debugging tile coverage)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# STEP 1: VALIDATE + LOAD
# ---------------------------------------------------------------------------

def validate_and_load(args: argparse.Namespace):
    """
    Verify that the input image and weights file exist, load the YOLO model,
    and read the image from disk.

    Returns (model, img_bgr, input_path).
    """
    print(f"\n[1/{N_STEPS}] Validating setup...")

    # --- input image ---
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {input_path}\n"
            "Make sure the stitching pipeline has produced the orthomosaic first."
        )
    print(f"  input      : {input_path}  ✓")

    # --- weights ---
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            "Run train.py first, or pass --weights with the correct path."
        )
    print(f"  weights    : {weights_path}  ✓")

    # --- read image ---
    img = cv2.imread(str(input_path))
    if img is None:
        raise RuntimeError(
            f"cv2.imread failed to open: {input_path}\n"
            "File may be corrupt or in an unsupported format."
        )
    h, w = img.shape[:2]
    print(f"  image size : {w} × {h} px  ({w * h / 1e6:.1f} MP)")

    # --- load model ---
    from ultralytics import YOLO
    print(f"\n  Loading model: {weights_path.name} ...")
    model = YOLO(str(weights_path))
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Model loaded  ✓  ({n_params:,} parameters)")

    return model, img, input_path


# ---------------------------------------------------------------------------
# TILING HELPER
# ---------------------------------------------------------------------------

def generate_tiles(img: np.ndarray,
                   tile_size: int, overlap: int) -> list[tuple]:
    """
    Slice the image into overlapping square tiles.

    Boundary strategy mirrors prepare_rescuenet.py: when the remaining strip
    at the right/bottom edge is narrower than tile_size, the final tile is
    shifted back so it is always exactly tile_size × tile_size. This avoids
    small edge-case tiles with different resolutions, at the cost of slightly
    extra overlap on the last row/column.

    Returns a list of (tile_img, x_offset, y_offset), where x_offset and
    y_offset are the top-left coordinates of the tile in the full image.
    """
    if overlap >= tile_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than tile_size ({tile_size})"
        )
    h, w   = img.shape[:2]
    stride = tile_size - overlap
    tiles  = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)   # shift back for right/bottom edges
            y1 = max(0, y2 - tile_size)
            tiles.append((img[y1:y2, x1:x2].copy(), x1, y1))

    return tiles


# ---------------------------------------------------------------------------
# STEP 3: TILED INFERENCE
# ---------------------------------------------------------------------------

def run_tiled_inference(args: argparse.Namespace, model,
                        tiles: list[tuple],
                        out_dir: Path) -> tuple[list, list, list]:
    """
    Run model.predict() on every tile and collect detections in full-image
    pixel coordinates.

    For each detection, the tile-local xyxy box is shifted by the tile's
    (x_offset, y_offset) to get coordinates relative to the full orthomosaic.

    If --save-tiles is set, every tile that contains at least one detection is
    saved as an annotated JPEG under {out_dir}/tiles/.

    Returns:
        all_boxes   : [[x1, y1, x2, y2], ...] in full-image pixels
        all_scores  : [float, ...]
        all_classes : [int, ...]
    """
    print(f"\n[3/{N_STEPS}] Running inference on {len(tiles)} tile(s)...")

    tiles_dir = out_dir / "tiles"
    if args.save_tiles:
        tiles_dir.mkdir(parents=True, exist_ok=True)
        print(f"  --save-tiles: annotated tiles → {tiles_dir}")

    all_boxes:   list[list[float]] = []
    all_scores:  list[float]       = []
    all_classes: list[int]         = []

    for idx, (tile_img, x_off, y_off) in enumerate(
            tqdm(tiles, desc="  Tiles", unit="tile", ncols=72)):

        results = model.predict(
            source=tile_img,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.tile_size,
            verbose=False,
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        # Convert tile-local xyxy coordinates to full-image coordinates
        for box in result.boxes:
            tx1, ty1, tx2, ty2 = box.xyxy[0].tolist()
            all_boxes.append([tx1 + x_off, ty1 + y_off,
                               tx2 + x_off, ty2 + y_off])
            all_scores.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

        # Save annotated tile for debugging if requested
        if args.save_tiles:
            annotated_tile = result.plot()
            tile_name = f"tile_{idx:04d}_x{x_off:05d}_y{y_off:05d}.jpg"
            cv2.imwrite(str(tiles_dir / tile_name), annotated_tile)

    print(f"  Raw detections before cross-tile NMS: {len(all_boxes)}")
    return all_boxes, all_scores, all_classes


# ---------------------------------------------------------------------------
# STEP 4: CROSS-TILE NMS
# ---------------------------------------------------------------------------

def merge_detections(boxes: list, scores: list, classes: list,
                     iou_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply per-class NMS across detections from all tiles to remove duplicate
    boxes that arise in overlapping tile regions.

    Uses torchvision.ops.batched_nms, which only suppresses within the same
    class — matching the semantics Ultralytics uses per-tile internally.

    Returns (boxes, scores, classes) as numpy arrays, or empty arrays if
    there were no input detections.
    """
    import torch
    from torchvision.ops import batched_nms

    if not boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0,      dtype=np.float32),
            np.zeros(0,      dtype=np.int32),
        )

    boxes_t   = torch.tensor(boxes,   dtype=torch.float32)
    scores_t  = torch.tensor(scores,  dtype=torch.float32)
    classes_t = torch.tensor(classes, dtype=torch.int64)

    keep = batched_nms(boxes_t, scores_t, classes_t, iou_thresh)

    return (
        boxes_t[keep].numpy(),
        scores_t[keep].numpy(),
        classes_t[keep].numpy().astype(np.int32),
    )


# ---------------------------------------------------------------------------
# STEP 5 HELPERS: ANNOTATION
# ---------------------------------------------------------------------------

def _draw_legend(img: np.ndarray) -> None:
    """
    Draw a class-color legend in the top-left corner of img, in-place.
    The legend background is a semi-transparent dark panel.
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    swatch_sz  = 16
    row_h      = swatch_sz + 6
    pad        = 10

    # Measure the widest label so we can size the background panel
    max_tw = max(
        cv2.getTextSize(name, font, font_scale, thickness)[0][0]
        for name in YOLO_CLASS_NAMES
    )
    panel_w = pad + swatch_sz + 6 + max_tw + pad
    panel_h = pad + len(YOLO_CLASS_NAMES) * row_h + pad
    x0, y0  = 12, 12

    # Semi-transparent dark background panel
    panel = img.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    img[:] = cv2.addWeighted(panel, 0.68, img, 0.32, 0)

    # Color swatches + text labels
    for cls_id, name in enumerate(YOLO_CLASS_NAMES):
        color = CLASS_COLORS[cls_id]
        sy = y0 + pad + cls_id * row_h

        # Color swatch with thin border
        cv2.rectangle(img, (x0 + pad, sy),
                      (x0 + pad + swatch_sz, sy + swatch_sz), color, -1)
        cv2.rectangle(img, (x0 + pad, sy),
                      (x0 + pad + swatch_sz, sy + swatch_sz), (200, 200, 200), 1)

        # Class name
        cv2.putText(img, name,
                    (x0 + pad + swatch_sz + 6, sy + swatch_sz - 2),
                    font, font_scale, (230, 230, 230), thickness, cv2.LINE_AA)


def annotate_image(img: np.ndarray,
                   boxes: np.ndarray,
                   scores: np.ndarray,
                   classes: np.ndarray) -> np.ndarray:
    """
    Draw semi-transparent filled boxes, opaque borders, and label chips
    onto a copy of img. Adds a class-color legend in the top-left corner.

    Two-pass approach:
      Pass 1 — draw all filled rectangles on an overlay, then blend at 25%
               opacity so underlying image structure remains visible.
      Pass 2 — draw opaque 2 px borders and label chips on top.

    Returns the annotated image (input is not modified).
    """
    annotated = img.copy()

    if len(boxes) == 0:
        _draw_legend(annotated)
        return annotated

    # --- Pass 1: semi-transparent filled boxes ---
    overlay = annotated.copy()
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), CLASS_COLORS[int(cls)], -1)
    # Blend: 25% fill color, 75% original image
    annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

    # --- Pass 2: opaque borders + label chips ---
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cls_i  = int(cls)
        color  = CLASS_COLORS[cls_i]

        # Box border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label chip: solid colored background + white text
        label         = f"{YOLO_CLASS_NAMES[cls_i]} {score:.2f}"
        (tw, th), _   = cv2.getTextSize(label, font, font_scale, thickness)
        chip_y1       = max(y1 - th - 6, 0)
        chip_y2       = chip_y1 + th + 6
        cv2.rectangle(annotated, (x1, chip_y1), (x1 + tw + 6, chip_y2), color, -1)
        cv2.putText(annotated, label, (x1 + 3, chip_y2 - 3),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # --- Legend ---
    _draw_legend(annotated)

    return annotated


# ---------------------------------------------------------------------------
# STEP 5 HELPERS: SUMMARY + SAVE
# ---------------------------------------------------------------------------

def print_summary(boxes: np.ndarray, scores: np.ndarray,
                  classes: np.ndarray, out_path: Path) -> None:
    """Print per-class detection counts and confidence statistics."""
    total = len(boxes)
    col   = max(len(n) for n in YOLO_CLASS_NAMES)

    print(f"\n  Detection Summary")
    print(f"  {'─' * 50}")

    if total == 0:
        print("  No detections above the confidence threshold.")
        print(f"\n  Output : {out_path}")
        return

    print(f"  Total detections : {total}")
    print()

    for cls_id, name in enumerate(YOLO_CLASS_NAMES):
        mask = classes == cls_id
        n    = int(mask.sum())
        if n == 0:
            print(f"  {name:<{col}s} :   0")
            continue
        c = scores[mask]
        print(f"  {name:<{col}s} :  {n:4d}  "
              f"(conf  min={c.min():.2f}  max={c.max():.2f}  mean={c.mean():.2f})")

    print(f"\n  Output : {out_path}")


def save_output(annotated: np.ndarray, input_path: Path,
                out_dir: Path) -> Path:
    """Write the annotated image to {out_dir}/{stem}_detections.jpg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_detections.jpg"
    ok = cv2.imwrite(str(out_path), annotated,
                     [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(
            f"cv2.imwrite failed — check that {out_dir} is writable."
        )
    return out_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  StitchWise — Tiled Orthomosaic Inference               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  input      : {args.input}")
    print(f"  weights    : {args.weights}")
    print(f"  output dir : {out_dir}")
    print(f"  conf / iou : {args.conf} / {args.iou}")
    print(f"  tile size  : {args.tile_size}px  overlap: {args.overlap}px")
    print(f"  device     : {args.device}")
    print(f"  save-tiles : {'yes' if args.save_tiles else 'no'}")
    print(f"  no-merge   : {'yes — NMS skipped' if args.no_merge else 'no'}")

    # ------------------------------------------------------------------
    # Step 1: Validate inputs, load model, read image
    # ------------------------------------------------------------------
    model, img, input_path = validate_and_load(args)
    img_h, img_w = img.shape[:2]

    # ------------------------------------------------------------------
    # Step 2: Tile the image (or skip for images smaller than one tile)
    # ------------------------------------------------------------------
    print(f"\n[2/{N_STEPS}] Preparing tiles...")

    is_small = img_h <= args.tile_size and img_w <= args.tile_size
    if is_small:
        print(f"  Image ({img_w}×{img_h}) fits within one tile — skipping tiling.")
        tiles = [(img, 0, 0)]
    else:
        tiles  = generate_tiles(img, args.tile_size, args.overlap)
        stride = args.tile_size - args.overlap
        n_cols = len(range(0, img_w, stride))
        n_rows = len(range(0, img_h, stride))
        print(f"  {len(tiles)} tiles  "
              f"({n_cols} cols × {n_rows} rows, "
              f"stride={stride}px, overlap={args.overlap}px)")

    # ------------------------------------------------------------------
    # Step 3: Run inference tile-by-tile; collect full-image coordinates
    # ------------------------------------------------------------------
    all_boxes, all_scores, all_classes = run_tiled_inference(
        args, model, tiles, out_dir
    )

    # ------------------------------------------------------------------
    # Step 4: Cross-tile NMS — remove duplicates at tile borders
    # ------------------------------------------------------------------
    print(f"\n[4/{N_STEPS}] Merging detections (cross-tile NMS)...")

    if args.no_merge:
        print("  --no-merge: skipping NMS, keeping all raw tile detections.")
        boxes   = np.array(all_boxes,   dtype=np.float32) if all_boxes   else np.zeros((0, 4), dtype=np.float32)
        scores  = np.array(all_scores,  dtype=np.float32) if all_scores  else np.zeros(0,      dtype=np.float32)
        classes = np.array(all_classes, dtype=np.int32)   if all_classes else np.zeros(0,      dtype=np.int32)
    else:
        boxes, scores, classes = merge_detections(
            all_boxes, all_scores, all_classes, args.iou
        )
        n_removed = len(all_boxes) - len(boxes)
        print(f"  {len(all_boxes)} raw → {len(boxes)} after NMS  "
              f"({n_removed} duplicate(s) removed)")

    if len(boxes) == 0:
        print(f"\n  WARNING: No detections found above conf={args.conf}.")
        print(f"  The unannotated image will be saved with the legend only.")
        print(f"  Consider lowering --conf or checking --weights.")

    # ------------------------------------------------------------------
    # Step 5: Annotate full image, save, and print detection summary
    # ------------------------------------------------------------------
    print(f"\n[5/{N_STEPS}] Annotating and saving output...")

    annotated = annotate_image(img, boxes, scores, classes)
    out_path  = save_output(annotated, input_path, out_dir)

    print_summary(boxes, scores, classes, out_path)

    print(f"\n{'='*62}")
    print("  Inference complete.")
    print(f"  Annotated image : {out_path}")
    if args.save_tiles and len(all_boxes) > 0:
        print(f"  Tile images     : {out_dir / 'tiles'}/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
