"""
RapidGeoStitch — Tiled YOLOv8 inference on orthomosaics.

Usage:
    python predict.py --input outputs/detections/scene_01.jpg
    python predict.py --input scene.jpg --conf 0.3 --save-tiles
    python predict.py --input scene.jpg --no-merge
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR     = Path(__file__).parent.resolve()
REPO_ROOT      = SCRIPT_DIR.parent
DETECTIONS_DIR = REPO_ROOT / "outputs" / "detections"

YOLO_CLASS_NAMES = ["water", "building-damaged", "road-blocked", "vehicle"]

CLASS_COLORS = {
    0: (210,  90,  20),
    1: ( 20,  20, 210),
    2: (  0, 140, 255),
    3: ( 20, 185,  20),
}


def parse_args() -> argparse.Namespace:
    _default_weights = str(REPO_ROOT / "detection" / "model" / "best.pt")
    parser = argparse.ArgumentParser(
        description="Run tiled YOLOv8 inference on an aerial image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--weights",    type=str, default=_default_weights)
    parser.add_argument("--output-dir", type=str, default=str(DETECTIONS_DIR))
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--iou",        type=float, default=0.6)
    parser.add_argument("--device",     type=str,   default="0")
    parser.add_argument("--tile-size",  type=int,   default=640)
    parser.add_argument("--overlap",    type=int,   default=64)
    parser.add_argument("--save-tiles", action="store_true")
    parser.add_argument("--no-merge",   action="store_true")
    return parser.parse_args()


def validate_and_load(args: argparse.Namespace):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    img = cv2.imread(str(input_path))
    if img is None:
        raise RuntimeError(f"Could not read: {input_path}")

    h, w = img.shape[:2]
    print(f"  image: {w}×{h} px  |  weights: {weights_path.name}")

    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    return model, img, input_path


def generate_tiles(img: np.ndarray, tile_size: int, overlap: int) -> list[tuple]:
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be < tile_size ({tile_size})")
    h, w   = img.shape[:2]
    stride = tile_size - overlap
    tiles  = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((img[y1:y2, x1:x2].copy(), x1, y1))
    return tiles


def run_tiled_inference(args, model, tiles: list[tuple], out_dir: Path):
    tiles_dir = out_dir / "tiles"
    if args.save_tiles:
        tiles_dir.mkdir(parents=True, exist_ok=True)

    all_boxes, all_scores, all_classes = [], [], []

    for idx, (tile_img, x_off, y_off) in enumerate(tqdm(tiles, desc="  Tiles", ncols=72)):
        results = model.predict(
            source=tile_img, conf=args.conf, iou=args.iou,
            device=args.device, imgsz=args.tile_size, verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            tx1, ty1, tx2, ty2 = box.xyxy[0].tolist()
            all_boxes.append([tx1 + x_off, ty1 + y_off, tx2 + x_off, ty2 + y_off])
            all_scores.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

        if args.save_tiles:
            cv2.imwrite(
                str(tiles_dir / f"tile_{idx:04d}_x{x_off:05d}_y{y_off:05d}.jpg"),
                result.plot(),
            )

    return all_boxes, all_scores, all_classes


def merge_detections(boxes, scores, classes, iou_thresh):
    import torch
    from torchvision.ops import batched_nms

    if not boxes:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))

    b = torch.tensor(boxes,   dtype=torch.float32)
    s = torch.tensor(scores,  dtype=torch.float32)
    c = torch.tensor(classes, dtype=torch.int64)
    keep = batched_nms(b, s, c, iou_thresh)
    return b[keep].numpy(), s[keep].numpy(), c[keep].numpy().astype(np.int32)


def _draw_legend(img: np.ndarray) -> None:
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    sw, rh, pad  = 16, 22, 10
    max_tw = max(cv2.getTextSize(n, font, fs, th)[0][0] for n in YOLO_CLASS_NAMES)
    pw = pad + sw + 6 + max_tw + pad
    ph = pad + len(YOLO_CLASS_NAMES) * rh + pad
    x0, y0 = 12, 12
    panel = img.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
    img[:] = cv2.addWeighted(panel, 0.68, img, 0.32, 0)
    for cls_id, name in enumerate(YOLO_CLASS_NAMES):
        color = CLASS_COLORS[cls_id]
        sy = y0 + pad + cls_id * rh
        cv2.rectangle(img, (x0 + pad, sy), (x0 + pad + sw, sy + sw), color, -1)
        cv2.rectangle(img, (x0 + pad, sy), (x0 + pad + sw, sy + sw), (200, 200, 200), 1)
        cv2.putText(img, name, (x0 + pad + sw + 6, sy + sw - 2), font, fs, (230, 230, 230), th, cv2.LINE_AA)


def annotate_image(img, boxes, scores, classes) -> np.ndarray:
    annotated = img.copy()
    if len(boxes) == 0:
        _draw_legend(annotated)
        return annotated

    overlay = annotated.copy()
    for box, cls in zip(boxes, classes):
        cv2.rectangle(overlay, tuple(map(int, box[:2])), tuple(map(int, box[2:])), CLASS_COLORS[int(cls)], -1)
    annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cls_i, color = int(cls), CLASS_COLORS[int(cls)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{YOLO_CLASS_NAMES[cls_i]} {score:.2f}"
        (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
        cy1 = max(y1 - tht - 6, 0)
        cv2.rectangle(annotated, (x1, cy1), (x1 + tw + 6, cy1 + tht + 6), color, -1)
        cv2.putText(annotated, label, (x1 + 3, cy1 + tht + 3), font, fs, (255, 255, 255), th, cv2.LINE_AA)

    _draw_legend(annotated)
    return annotated


def print_summary(boxes, scores, classes, out_path) -> None:
    total = len(boxes)
    col   = max(len(n) for n in YOLO_CLASS_NAMES)
    print(f"\n  Total detections: {total}")
    for cls_id, name in enumerate(YOLO_CLASS_NAMES):
        mask = classes == cls_id
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:<{col}s}: 0")
        else:
            c = scores[mask]
            print(f"  {name:<{col}s}: {n:4d}  (conf min={c.min():.2f} max={c.max():.2f} mean={c.mean():.2f})")
    print(f"  Output: {out_path}")


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)

    model, img, input_path = validate_and_load(args)
    img_h, img_w = img.shape[:2]

    if img_h <= args.tile_size and img_w <= args.tile_size:
        tiles = [(img, 0, 0)]
    else:
        tiles = generate_tiles(img, args.tile_size, args.overlap)
    print(f"  {len(tiles)} tile(s)")

    all_boxes, all_scores, all_classes = run_tiled_inference(args, model, tiles, out_dir)

    if args.no_merge:
        boxes   = np.array(all_boxes,   dtype=np.float32) if all_boxes   else np.zeros((0, 4), np.float32)
        scores  = np.array(all_scores,  dtype=np.float32) if all_scores  else np.zeros(0,      np.float32)
        classes = np.array(all_classes, dtype=np.int32)   if all_classes else np.zeros(0,      np.int32)
    else:
        boxes, scores, classes = merge_detections(all_boxes, all_scores, all_classes, args.iou)

    annotated = annotate_image(img, boxes, scores, classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_detections.jpg"
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print_summary(boxes, scores, classes, out_path)


if __name__ == "__main__":
    main()
