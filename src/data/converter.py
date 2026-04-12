"""Mask-to-YOLO conversion and image tiling for RescueNet data preparation.

These are the core data processing primitives called by scripts/prepare_data.py.
All class-mapping logic is driven by the caller (via yolo_class_map), keeping
these functions independent of any specific active_classes configuration.
"""

from typing import Generator

import cv2
import numpy as np

from src.utils.classes import RESCUENET_CLASSES


def build_yolo_class_map(active_classes: list) -> dict:
    """Return a RescueNet pixel-value → YOLO-index mapping for conversion functions.

    Args:
        active_classes: Ordered list of RescueNet pixel values, e.g. [1, 4, 5, 7, 8].
                        Read from config.yaml data_preparation.active_classes.

    Returns:
        Dict mapping original pixel value to YOLO 0-based index.
    """
    return {orig: yolo for yolo, orig in enumerate(active_classes)}


def mask_to_yolo_detect(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    yolo_class_map: dict,
    min_contour_area: int = 500,
) -> list:
    """Convert a grayscale segmentation mask to YOLO detection format lines.

    Args:
        mask: Grayscale mask array with RescueNet pixel values.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        yolo_class_map: Dict mapping RescueNet pixel value → YOLO class index.
                        Build with build_yolo_class_map(active_classes).
        min_contour_area: Minimum contour area (px²) to filter noise.

    Returns:
        List of strings in "class_id cx cy w h" format (values normalized 0-1).
    """
    lines = []
    for orig_cls, yolo_cls in yolo_class_map.items():
        binary = (mask == orig_cls).astype(np.uint8)
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = min(max((x + w / 2) / img_w, 0.0), 1.0)
            cy = min(max((y + h / 2) / img_h, 0.0), 1.0)
            nw = min(max(w / img_w, 0.0), 1.0)
            nh = min(max(h / img_h, 0.0), 1.0)
            if nw > 0 and nh > 0:
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def mask_to_yolo_segment(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    yolo_class_map: dict,
    min_contour_area: int = 500,
) -> list:
    """Convert a grayscale segmentation mask to YOLO segmentation format lines.

    Args:
        mask: Grayscale mask array with RescueNet pixel values.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        yolo_class_map: Dict mapping RescueNet pixel value → YOLO class index.
        min_contour_area: Minimum contour area (px²) to filter noise.

    Returns:
        List of strings in "class_id x1 y1 x2 y2 ..." format (normalized polygon).
    """
    lines = []
    for orig_cls, yolo_cls in yolo_class_map.items():
        binary = (mask == orig_cls).astype(np.uint8)
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_contour_area:
                continue
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2)
            coords = [f"{px / img_w:.6f} {py / img_h:.6f}" for px, py in pts]
            lines.append(f"{yolo_cls} " + " ".join(coords))
    return lines


def tile_image_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    overlap: int = 64,
) -> Generator:
    """Yield (tile_img, tile_mask, x_offset, y_offset) patches from a large image.

    Args:
        img: Full-resolution image array (H x W x C).
        mask: Corresponding grayscale mask array (H x W).
        tile_size: Side length of each square tile in pixels.
        overlap: Pixel overlap between adjacent tiles to avoid border artifacts.

    Yields:
        Tuples of (tile_img, tile_mask, x1, y1).
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
