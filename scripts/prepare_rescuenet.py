"""
RescueNet Dataset Preparation for YOLOv8
=========================================
This script:
  1. Downloads the RescueNet dataset from Kaggle via kagglehub
  2. Inspects the dataset structure and class distribution
  3. Converts grayscale semantic segmentation masks -> YOLO format labels
       - Detection mode:  one .txt per image with bounding boxes
       - Segmentation mode: one .txt per image with polygon contours  (stretch goal)
  4. Tiles large 3000x4000 images into smaller patches (recommended for YOLO)
  5. Generates data.yaml for Ultralytics YOLOv8

Usage:
    python prepare_rescuenet.py [--mode detect|segment] [--tile] [--tile-size 640]

Requirements:
    pip install kagglehub ultralytics opencv-python-headless numpy pyyaml tqdm
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
import random

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# RescueNet class definitions (pixel value -> class name)
# Index 0 = background (ignored for YOLO labels)
CLASSES = {
    0:  "background",          # ignored
    1:  "water",
    2:  "building-no-damage",
    3:  "building-minor-damage",
    4:  "building-major-damage",
    5:  "building-total-destruction",
    6:  "road-clear",
    7:  "road-blocked",
    8:  "vehicle",
    9:  "tree",
    10: "pool",
}

# Classes to INCLUDE in YOLO labels (exclude background and undamaged classes
# if you only care about disaster-relevant detections).
# Edit this list to focus on specific classes.
ACTIVE_CLASSES = [1, 4, 5, 7, 8]  # water, major-damage, total-destruction, road-blocked, vehicle
# For all classes: ACTIVE_CLASSES = list(range(1, 11))

# Mapping from original class index -> YOLO class index (0-based)
YOLO_CLASS_MAP = {orig: yolo for yolo, orig in enumerate(ACTIVE_CLASSES)}
YOLO_CLASS_NAMES = [CLASSES[i] for i in ACTIVE_CLASSES]

# Minimum contour area in pixels to filter noise
MIN_CONTOUR_AREA = 500

# ---------------------------------------------------------------------------
# STEP 1: DOWNLOAD
# ---------------------------------------------------------------------------

def download_dataset(output_dir: Path) -> Path:
    """Download RescueNet from Kaggle using kagglehub."""
    print("\n[1/5] Downloading RescueNet from Kaggle...")
    print("      Make sure your Kaggle API credentials are set up:")
    print("      https://www.kaggle.com/docs/api#authentication\n")

    try:
        import kagglehub
        path = kagglehub.dataset_download("yaroslavchyrko/rescuenet")
        path = Path(path)
        print(f"      Downloaded to: {path}")
        return path
    except Exception as e:
        # Fallback: assume already downloaded
        fallback = Path("/kaggle/input/rescuenet")
        if fallback.exists():
            print(f"      Using fallback path: {fallback}")
            return fallback
        raise RuntimeError(
            f"Download failed: {e}\n"
            "Make sure kagglehub is installed and Kaggle credentials are configured.\n"
            "Alternatively, download manually and set --dataset-path."
        )


# ---------------------------------------------------------------------------
# STEP 2: INSPECT
# ---------------------------------------------------------------------------

def inspect_dataset(dataset_path: Path):
    """Print dataset structure and class distribution from a sample of masks."""
    print("\n[2/5] Inspecting dataset structure...")

    # We know the structure now: train/train-org-img, etc.
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
            
        # Search for the actual RescueNet image/label directories
        img_dir = next(split_dir.glob("*-org-img"), None)
        lbl_dir = next(split_dir.glob("*-label-img"), None)

        imgs = list(img_dir.glob("*.jpg")) if img_dir else []
        lbls = list(lbl_dir.glob("*.png")) if lbl_dir else []

        print(f"      [{split}] folder found! images: {len(imgs)}, masks: {len(lbls)}")

    # Sample mask for class distribution
    all_masks = list(dataset_path.glob("**/*-label-img/*.png"))
    if all_masks:
        print(f"\n      Sampling {min(50, len(all_masks))} masks for class distribution...")
        class_counts = np.zeros(11, dtype=np.int64)
        for mask_path in all_masks[:50]:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                for cls_id in range(11):
                    class_counts[cls_id] += np.sum(mask == cls_id)

        print("\n      Class pixel distribution (sampled):")
        total = class_counts.sum()
        for cls_id, name in CLASSES.items():
            pct = 100 * class_counts[cls_id] / total if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"        {cls_id:2d}  {name:<30s}  {pct:5.1f}%  {bar}")
    else:
        print("      WARNING: No mask files found during inspection.")


# ---------------------------------------------------------------------------
# STEP 3: FIND IMAGE-MASK PAIRS
# ---------------------------------------------------------------------------

def find_pairs(dataset_path: Path):
    """
    Return a dict matching RescueNet folders by sorted alignment 
    (most robust for RescueNet's naming variations).
    """
    pairs = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue

        img_dir = next(split_dir.glob("*-org-img"), None)
        lbl_dir = next(split_dir.glob("*-label-img"), None)

        if not img_dir or not lbl_dir:
            continue

        # Get sorted lists of all images and all masks
        img_list = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG")))
        lbl_list = sorted(list(lbl_dir.glob("*.png")) + list(lbl_dir.glob("*.PNG")))

        if len(img_list) != len(lbl_list):
            print(f"      WARNING: [{split}] Mismatch in counts! Images: {len(img_list)}, Masks: {len(lbl_list)}")
            # If counts differ, we fall back to stem-matching to be safe
            split_pairs = []
            lbl_stems = {f.stem for f in lbl_list}
            for img_path in img_list:
                if img_path.stem in lbl_stems:
                    mask_path = lbl_dir / (img_path.stem + ".png")
                    split_pairs.append((img_path, mask_path))
        else:
            # If counts match, direct zip is the most reliable for RescueNet
            split_pairs = list(zip(img_list, lbl_list))

        if split_pairs:
            pairs[split] = split_pairs
            print(f"      [{split}] Successfully matched {len(split_pairs)} pairs.")
        else:
            print(f"      ERROR: Could not match any pairs in {split} using stem or count methods.")
            # Let's see the first few names to debug
            if img_list: print(f"      Sample Image: {img_list[0].name}")
            if lbl_list: print(f"      Sample Mask: {lbl_list[0].name}")

    return pairs


# ---------------------------------------------------------------------------
# STEP 4A: MASK -> YOLO DETECTION LABELS (bounding boxes)
# ---------------------------------------------------------------------------

def mask_to_yolo_detect(mask: np.ndarray, img_h: int, img_w: int) -> list[str]:
    """
    Convert a grayscale segmentation mask to YOLO detection format lines.
    Returns list of strings: "class_id cx cy w h" (normalized 0-1)
    """
    lines = []
    for orig_cls, yolo_cls in YOLO_CLASS_MAP.items():
        binary = (mask == orig_cls).astype(np.uint8)
        if binary.sum() == 0:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Normalize
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            # Clamp
            cx, cy, nw, nh = [min(max(v, 0.0), 1.0) for v in [cx, cy, nw, nh]]
            if nw > 0 and nh > 0:
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


# ---------------------------------------------------------------------------
# STEP 4B: MASK -> YOLO SEGMENTATION LABELS (polygons) - stretch goal
# ---------------------------------------------------------------------------

def mask_to_yolo_segment(mask: np.ndarray, img_h: int, img_w: int) -> list[str]:
    """
    Convert a grayscale segmentation mask to YOLO segmentation format lines.
    Returns list of strings: "class_id x1 y1 x2 y2 ..." (normalized, polygon points)
    """
    lines = []
    for orig_cls, yolo_cls in YOLO_CLASS_MAP.items():
        binary = (mask == orig_cls).astype(np.uint8)
        if binary.sum() == 0:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            # Simplify polygon to reduce point count
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2)
            coords = []
            for px, py in pts:
                coords.append(f"{px / img_w:.6f} {py / img_h:.6f}")
            lines.append(f"{yolo_cls} " + " ".join(coords))
    return lines


# ---------------------------------------------------------------------------
# STEP 4C: OPTIONAL TILING (recommended for 3000x4000 images)
# ---------------------------------------------------------------------------

def tile_image_and_mask(img: np.ndarray, mask: np.ndarray,
                         tile_size: int, overlap: int = 64):
    """
    Yield (tile_img, tile_mask, x_offset, y_offset) patches.
    overlap: pixel overlap between adjacent tiles to avoid border artifacts.
    """
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")
    h, w = img.shape[:2]
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)
            yield img[y1:y2, x1:x2], mask[y1:y2, x1:x2], x1, y1


# ---------------------------------------------------------------------------
# STEP 4: MAIN CONVERSION
# ---------------------------------------------------------------------------

def sample_pairs(pairs: dict, img_num: int | None) -> dict:
    """
    Randomly subsample pairs per split while preserving the original
    80/10/10 train/val/test ratio.

    If img_num is None, returns pairs unchanged (full dataset).
    img_num refers to the number of SOURCE images (before tiling).
    """
    if img_num is None:
        return pairs

    import random

    # Compute split sizes proportional to the original distribution
    total_orig = sum(len(v) for v in pairs.values())
    sampled = {}
    remaining = img_num

    split_order = ["train", "val", "test"]
    for i, split in enumerate(split_order):
        if split not in pairs:
            continue
        orig_count = len(pairs[split])
        if i < len(split_order) - 1:
            # Proportional allocation
            proportion = orig_count / total_orig
            n = min(max(1, round(img_num * proportion)), orig_count)
        else:
            # Give remainder to last split to avoid rounding loss
            n = min(remaining, orig_count)
        sampled[split] = random.sample(pairs[split], n)
        remaining -= n
        print(f"      [{split}] sampling {n} / {orig_count} source images")

    total_sampled = sum(len(v) for v in sampled.values())
    print(f"      Total sampled: {total_sampled} source images "
          f"(requested: {img_num})")
    return sampled


def convert_dataset(pairs: dict, output_dir: Path, mode: str = "detect",
                     tile: bool = True, tile_size: int = 640,
                     img_num: int | None = None):
    """Convert all image/mask pairs to YOLO format."""

    # Apply sampling BEFORE conversion so we only process what we need
    if img_num is not None:
        print(f"\n[3/5] Converting masks to YOLO {mode} format "
              f"(limited to {img_num} source images)...")
        pairs = sample_pairs(pairs, img_num)
    else:
        print(f"\n[3/5] Converting masks to YOLO {mode} format (full dataset)...")

    if tile:
        print(f"      Tiling enabled: {tile_size}x{tile_size}px patches")
        approx_tiles = sum(len(v) for v in pairs.values()) * 20
        print(f"      Estimated tiles to write: ~{approx_tiles:,} "
              f"(only tiles with labels are saved)")

    convert_fn = mask_to_yolo_detect if mode == "detect" else mask_to_yolo_segment

    stats = {"total_images": 0, "total_labels": 0, "skipped": 0}

    for split, split_pairs in pairs.items():
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in tqdm(split_pairs, desc=f"  {split}"):
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                stats["skipped"] += 1
                continue

            stem = img_path.stem

            if tile:
                for tile_idx, (t_img, t_mask, _, _) in enumerate(
                        tile_image_and_mask(img, mask, tile_size)):
                    tile_name = f"{stem}_t{tile_idx:04d}"
                    lines = convert_fn(t_mask, t_img.shape[0], t_img.shape[1])
                    # Only save tiles that have at least one label
                    if lines:
                        # Save image and labels for tiles WITH objects
                        cv2.imwrite(str(img_out / f"{tile_name}.jpg"), t_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        with open(lbl_out / f"{tile_name}.txt", "w") as f:
                            f.write("\n".join(lines))
                        stats["total_images"] += 1
                        stats["total_labels"] += len(lines)
                    elif random.random() < 0.10:  # Keep ~10% of empty background tiles
                        # Save image and an EMPTY label file for background tiles
                        cv2.imwrite(str(img_out / f"{tile_name}.jpg"), t_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        with open(lbl_out / f"{tile_name}.txt", "w") as f:
                            f.write("") # Empty file
                        stats["total_images"] += 1
            else:
                h, w = img.shape[:2]
                lines = convert_fn(mask, h, w)
                # Only save images that have at least one label
                if lines:
                    cv2.imwrite(str(img_out / f"{stem}.jpg"), img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(lbl_out / f"{stem}.txt", "w") as f:
                        f.write("\n".join(lines))
                    stats["total_images"] += 1
                    stats["total_labels"] += len(lines)

    print(f"\n      Conversion complete:")
    print(f"        Images written : {stats['total_images']}")
    print(f"        Label entries  : {stats['total_labels']}")
    print(f"        Skipped (bad)  : {stats['skipped']}")


# ---------------------------------------------------------------------------
# STEP 5: GENERATE data.yaml
# ---------------------------------------------------------------------------

def generate_data_yaml(output_dir: Path, mode: str):
    """Write the data.yaml file that YOLOv8 expects."""
    print("\n[4/5] Generating data.yaml...")

    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(YOLO_CLASS_NAMES),
        "names": YOLO_CLASS_NAMES,
        # metadata
        "_mode": mode,
        "_source": "RescueNet (yaroslavchyrko/rescuenet on Kaggle)",
        "_active_classes": ACTIVE_CLASSES,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"      Saved: {yaml_path}")
    print(f"      Classes ({len(YOLO_CLASS_NAMES)}): {YOLO_CLASS_NAMES}")
    return yaml_path


# ---------------------------------------------------------------------------
# STEP 6: SANITY CHECK
# ---------------------------------------------------------------------------

def sanity_check(output_dir: Path):
    """Quick check: verify a few label files look reasonable."""
    print("\n[5/5] Running sanity check...")
    label_files = list(output_dir.glob("**/*.txt"))

    if not label_files:
        print("      WARNING: No label files found!")
        return

    sample = label_files[:5]
    print(f"      Checking {len(sample)} sample label files:")
    for lf in sample:
        lines = lf.read_text().strip().splitlines()
        print(f"        {lf.name}: {len(lines)} objects")
        if lines:
            print(f"          sample: {lines[0]}")

    # Check for empty label files
    empty = sum(1 for lf in label_files if lf.stat().st_size == 0)
    print(f"\n      Total label files : {len(label_files)}")
    print(f"      Empty label files : {empty}  (tiles with no active classes — expected)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare RescueNet for YOLOv8")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to already-downloaded RescueNet dataset. "
                             "If not set, will download via kagglehub.")
    parser.add_argument("--output-dir", type=str, default="../data/processed",
                        help="Where to write the converted YOLO dataset (default: ../data/processed)")
    parser.add_argument("--mode", choices=["detect", "segment"], default="detect",
                        help="detect = bounding boxes, segment = polygons (default: detect)")
    parser.add_argument("--tile", action="store_true", default=True,
                        help="Tile large images into patches (recommended, default: True)")
    parser.add_argument("--no-tile", dest="tile", action="store_false",
                        help="Disable tiling (keeps full 3000x4000 images)")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Tile size in pixels (default: 640)")
    parser.add_argument("--img-num", type=int, default=20,
                        help="Max SOURCE images to sample across all splits, "
                             "preserving the 80/10/10 train/val/test ratio. "
                             "Default: 20 images. "
                             "Use 0 for the full dataset. "
                             "Suggested values: "
                             "quick smoke test=20, dev run=200, full=0")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only inspect dataset structure, do not convert")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert 0 -> None (sentinel for "full dataset")
    img_num = args.img_num if args.img_num > 0 else None

    # Step 1: Download or use provided path
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        print(f"[1/5] Using provided dataset path: {dataset_path}")
    else:
        dataset_path = download_dataset(output_dir)

    # Step 2: Inspect
    inspect_dataset(dataset_path)

    if args.inspect_only:
        print("\nInspect-only mode. Exiting.")
        return

    # Step 3: Find pairs
    print("\n[2.5/5] Finding image-mask pairs...")
    pairs = find_pairs(dataset_path)
    if not pairs:
        raise RuntimeError("No image-mask pairs found. Check dataset structure.")

    # Step 4: Convert
    if img_num is not None:
        print(f"\n      --img-num={img_num}: sampling {img_num} source images "
              f"(~{img_num * 20:,} tiles estimated after tiling)")
    convert_dataset(pairs, output_dir, mode=args.mode,
                    tile=args.tile, tile_size=args.tile_size,
                    img_num=img_num)

    # Step 5: data.yaml
    yaml_path = generate_data_yaml(output_dir, mode=args.mode)

    # Step 6: Sanity check
    sanity_check(output_dir)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Dataset preparation complete!                           ║
║                                                          ║
║  Output directory : {str(output_dir):<37s}║
║  data.yaml        : {str(yaml_path):<37s}║
║                                                          ║
║  Next step — train YOLOv8:                               ║
║                                                          ║
║  from ultralytics import YOLO                            ║
║  model = YOLO('yolov8m.pt')                              ║
║  model.train(data='{yaml_path}',   ║
║              epochs=50, imgsz=640, batch=16)             ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()