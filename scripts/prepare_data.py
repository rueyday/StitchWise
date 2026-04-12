"""
RescueNet Dataset Preparation for YOLOv8
=========================================
Converts the local RescueNet semantic segmentation dataset into YOLO format and
tiles the large aerial images into model-ready patches.

Pipeline:
  1. Verify local dataset path
  2. Inspect structure and class distribution
  3. Find image-mask pairs across train/val/test splits
  4. Convert grayscale masks to YOLO labels (detect or segment mode)
  5. Tile 3000x4000 images into 640x640 patches with configurable overlap
  6. Write data.yaml
  7. Sanity-check label files

All parameters (active classes, tile size, sampling limits, etc.) are read
from config.yaml.  Override the config path with --config.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --mode segment --config config.yaml
    python scripts/prepare_data.py --inspect-only
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.data.converter import (
    build_yolo_class_map,
    mask_to_yolo_detect,
    mask_to_yolo_segment,
    tile_image_and_mask,
)
from src.utils.classes import RESCUENET_CLASSES
from src.utils.logger import setup_logger, get_logger


# ---------------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# STEP 1: INSPECT
# ---------------------------------------------------------------------------

def inspect_dataset(dataset_path: Path, cfg: dict):
    """Print dataset structure and class distribution from a sample of masks."""
    log = get_logger("prepare_data")
    log.info("Inspecting dataset structure...")

    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        img_dir = next(split_dir.glob("*-org-img"), None)
        lbl_dir = next(split_dir.glob("*-label-img"), None)
        imgs = list(img_dir.glob("*.jpg")) if img_dir else []
        lbls = list(lbl_dir.glob("*.png")) if lbl_dir else []
        log.info(f"[{split}] images: {len(imgs)}, masks: {len(lbls)}")

    all_masks = list(dataset_path.glob("**/*-label-img/*.png"))
    if all_masks:
        sample_n = min(50, len(all_masks))
        log.info(f"Sampling {sample_n} masks for class distribution...")
        class_counts = np.zeros(11, dtype=np.int64)
        for mask_path in all_masks[:sample_n]:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                for cls_id in range(11):
                    class_counts[cls_id] += np.sum(mask == cls_id)

        total = class_counts.sum()
        print("\n  Class pixel distribution (sampled):")
        for cls_id, name in RESCUENET_CLASSES.items():
            pct = 100 * class_counts[cls_id] / total if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {cls_id:2d}  {name:<30s}  {pct:5.1f}%  {bar}")
    else:
        log.warning("No mask files found during inspection.")


# ---------------------------------------------------------------------------
# STEP 2: FIND IMAGE-MASK PAIRS
# ---------------------------------------------------------------------------

def find_pairs(dataset_path: Path) -> dict:
    """
    Match image-mask pairs per split by sorted alignment (most robust for
    RescueNet's naming variations).  Falls back to stem-matching if counts differ.
    """
    log = get_logger("prepare_data")
    pairs = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue

        img_dir = next(split_dir.glob("*-org-img"), None)
        lbl_dir = next(split_dir.glob("*-label-img"), None)
        if not img_dir or not lbl_dir:
            continue

        img_list = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG")))
        lbl_list = sorted(list(lbl_dir.glob("*.png")) + list(lbl_dir.glob("*.PNG")))

        if len(img_list) != len(lbl_list):
            log.warning(f"[{split}] Count mismatch — images: {len(img_list)}, masks: {len(lbl_list)}. Falling back to stem matching.")
            lbl_stems = {f.stem for f in lbl_list}
            split_pairs = [
                (img, lbl_dir / (img.stem + ".png"))
                for img in img_list if img.stem in lbl_stems
            ]
        else:
            split_pairs = list(zip(img_list, lbl_list))

        if split_pairs:
            pairs[split] = split_pairs
            log.info(f"[{split}] Matched {len(split_pairs)} pairs.")
        else:
            log.error(f"Could not match any pairs in [{split}].")
            if img_list:
                log.error(f"  Sample image: {img_list[0].name}")
            if lbl_list:
                log.error(f"  Sample mask:  {lbl_list[0].name}")

    return pairs


# ---------------------------------------------------------------------------
# STEP 3: SAMPLE + CONVERT
# ---------------------------------------------------------------------------

def sample_pairs(pairs: dict, img_num: int, cfg: dict) -> dict:
    """
    Subsample pairs per split while preserving the original train/val/test ratio.
    img_num refers to the number of source images (before tiling).
    """
    log = get_logger("prepare_data")
    total_orig = sum(len(v) for v in pairs.values())
    sampled = {}
    remaining = img_num
    split_order = ["train", "val", "test"]

    for i, split in enumerate(split_order):
        if split not in pairs:
            continue
        orig_count = len(pairs[split])
        if i < len(split_order) - 1:
            proportion = orig_count / total_orig
            n = min(max(1, round(img_num * proportion)), orig_count)
        else:
            n = min(remaining, orig_count)
        sampled[split] = random.sample(pairs[split], n)
        remaining -= n
        log.info(f"[{split}] sampling {n} / {orig_count} source images")

    log.info(f"Total sampled: {sum(len(v) for v in sampled.values())} source images (requested: {img_num})")
    return sampled


def convert_dataset(
    pairs: dict,
    output_dir: Path,
    mode: str,
    cfg: dict,
    img_num: int | None,
):
    """Convert all image-mask pairs to YOLO format, with optional tiling."""
    log = get_logger("prepare_data")
    prep = cfg["data_preparation"]

    active_classes = prep["active_classes"]
    yolo_class_map = build_yolo_class_map(active_classes)
    min_contour_area = prep["min_contour_area"]
    tile = True  # tiling is always on; use --no-tile flag to disable
    tile_size = prep["tile_size"]
    overlap = prep["tile_overlap"]
    water_dropout = prep["water_dropout_ratio"]
    bg_keep = prep["background_keep_ratio"]

    if img_num is not None:
        log.info(f"Converting to YOLO [{mode}] format (sampled: {img_num} source images)")
        pairs = sample_pairs(pairs, img_num, cfg)
    else:
        log.info(f"Converting to YOLO [{mode}] format (full dataset)")

    convert_fn = mask_to_yolo_detect if mode == "detect" else mask_to_yolo_segment

    stats = {"total_images": 0, "total_labels": 0, "skipped": 0}
    class_counts = {yolo_cls: 0 for yolo_cls in range(len(active_classes))}

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

            for tile_idx, (t_img, t_mask, _, _) in enumerate(
                tile_image_and_mask(img, mask, tile_size, overlap)
            ):
                tile_name = f"{stem}_t{tile_idx:04d}"

                # Smart tile filtering: drop most redundant water-dominant tiles
                water_ratio = np.sum(t_mask == 1) / t_mask.size
                has_rare_class = np.any(np.isin(t_mask, [4, 5, 7]))
                if water_ratio > 0.80 and not has_rare_class:
                    if random.random() > (1.0 - water_dropout):
                        continue

                lines = convert_fn(t_mask, t_img.shape[0], t_img.shape[1],
                                   yolo_class_map, min_contour_area)

                if lines:
                    cv2.imwrite(str(img_out / f"{tile_name}.jpg"), t_img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (lbl_out / f"{tile_name}.txt").write_text("\n".join(lines))
                    stats["total_images"] += 1
                    stats["total_labels"] += len(lines)
                    for line in lines:
                        cls_id = int(line.split()[0])
                        class_counts[cls_id] += 1
                elif random.random() < bg_keep:
                    # Keep a small fraction of empty tiles as negative examples
                    cv2.imwrite(str(img_out / f"{tile_name}.jpg"), t_img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (lbl_out / f"{tile_name}.txt").write_text("")
                    stats["total_images"] += 1

    log.info(f"Conversion complete — images: {stats['total_images']}, labels: {stats['total_labels']}, skipped: {stats['skipped']}")

    total_objects = sum(class_counts.values())
    if total_objects > 0:
        print("\n  Final YOLO class distribution (object counts):")
        yolo_names = [RESCUENET_CLASSES[c] for c in active_classes]
        for yolo_cls, count in class_counts.items():
            pct = (count / total_objects) * 100
            print(f"    {yolo_cls} ({yolo_names[yolo_cls]:<26}): {count:<6} ({pct:.1f}%)")
    else:
        print("  No objects found in the sampled dataset.")


# ---------------------------------------------------------------------------
# STEP 4: GENERATE data.yaml
# ---------------------------------------------------------------------------

def generate_data_yaml(output_dir: Path, mode: str, cfg: dict) -> Path:
    """Write the data.yaml file that YOLOv8 expects."""
    log = get_logger("prepare_data")
    prep = cfg["data_preparation"]
    active_classes = prep["active_classes"]
    class_names = [RESCUENET_CLASSES[i] for i in active_classes]

    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
        "_mode": mode,
        "_source": "Local RescueNet",
        "_active_classes": active_classes,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    log.info(f"data.yaml written to {yaml_path}")
    log.info(f"Classes ({len(class_names)}): {class_names}")
    return yaml_path


# ---------------------------------------------------------------------------
# STEP 5: SANITY CHECK
# ---------------------------------------------------------------------------

def sanity_check(output_dir: Path):
    log = get_logger("prepare_data")
    label_files = list(output_dir.glob("**/*.txt"))
    if not label_files:
        log.warning("No label files found!")
        return

    sample = label_files[:5]
    log.info(f"Checking {len(sample)} sample label files:")
    for lf in sample:
        lines = lf.read_text().strip().splitlines()
        log.info(f"  {lf.name}: {len(lines)} objects" + (f" — sample: {lines[0]}" if lines else ""))

    empty = sum(1 for lf in label_files if lf.stat().st_size == 0)
    log.info(f"Total label files: {len(label_files)} | Empty (background tiles): {empty}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare RescueNet for YOLOv8")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to project config.yaml (default: config.yaml)")
    parser.add_argument("--dataset-path", default="data/raw/rescuenet/RescueNet",
                        help="Path to local RescueNet dataset root.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for YOLO dataset. Overrides config dataset.data_root.")
    parser.add_argument("--mode", choices=["detect", "segment"], default="segment",
                        help="detect = bounding boxes, segment = polygons (default: segment)")
    parser.add_argument("--img-num", type=int, default=None,
                        help="Max source images to sample (overrides config data_preparation.img_num). "
                             "Use 0 for full dataset.")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only inspect dataset structure; skip conversion")
    args = parser.parse_args()

    cfg = load_config(args.config)

    log_cfg = cfg["logging"]
    setup_logger(log_level=log_cfg["log_level"])
    log = get_logger("prepare_data")

    output_dir = Path(args.output_dir or cfg["dataset"]["data_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve img_num: CLI > config > None (full dataset)
    if args.img_num is not None:
        img_num = args.img_num if args.img_num > 0 else None
    else:
        cfg_num = cfg["data_preparation"].get("img_num", 20)
        img_num = cfg_num if cfg_num > 0 else None

    # Step 1: Verify Local Data
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        log.error(f"Dataset not found at: {dataset_path.resolve()}")
        raise FileNotFoundError(f"Cannot find local dataset. Ensure your data is extracted to {args.dataset_path}")
    log.info(f"Using local dataset path: {dataset_path}")

    # Step 2: Inspect
    inspect_dataset(dataset_path, cfg)
    if args.inspect_only:
        log.info("Inspect-only mode. Exiting.")
        return

    # Step 3: Find image-mask pairs
    log.info("Finding image-mask pairs...")
    pairs = find_pairs(dataset_path)
    if not pairs:
        raise RuntimeError("No image-mask pairs found. Check dataset structure.")

    # Step 4: Convert
    convert_dataset(pairs, output_dir, args.mode, cfg, img_num)

    # Step 5: YAML
    yaml_path = generate_data_yaml(output_dir, args.mode, cfg)

    # Step 6: Verify
    sanity_check(output_dir)

    log.info(f"Dataset preparation complete. Output: {output_dir}")
    log.info(f"Next step: python scripts/train.py --config {args.config}")


if __name__ == "__main__":
    main()