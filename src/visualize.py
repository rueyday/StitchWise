"""
Visualization utilities for metric scale results.

Outputs:
  - Scale bar overlaid on image (saved to results/)
  - Depth map heatmap (when depth model was used)
  - Per-frame GSD timeline plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def draw_scale_bar(
    image: np.ndarray,
    gsd_m_per_px: float,
    target_length_m: float = 10.0,
    position: str = "bottom-left",
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 3,
) -> np.ndarray:
    """
    Overlay a metric scale bar on the image.

    Args:
        image: BGR image (H×W×3)
        gsd_m_per_px: Ground sampling distance in meters/pixel
        target_length_m: Desired scale bar length in meters
        position: "bottom-left" | "bottom-right" | "top-left" | "top-right"
        color: BGR color tuple
        thickness: Line thickness in pixels

    Returns:
        Image with scale bar drawn (copy, original unchanged)
    """
    img = image.copy()
    h, w = img.shape[:2]

    bar_px = int(round(target_length_m / gsd_m_per_px))
    bar_px = max(bar_px, 10)  # always visible

    margin = int(min(h, w) * 0.04)
    bar_h = max(thickness * 4, 12)

    if position == "bottom-left":
        x0, y0 = margin, h - margin - bar_h
    elif position == "bottom-right":
        x0, y0 = w - margin - bar_px, h - margin - bar_h
    elif position == "top-left":
        x0, y0 = margin, margin
    else:
        x0, y0 = w - margin - bar_px, margin

    x1, y1 = x0 + bar_px, y0 + bar_h

    # Draw filled rectangle as background
    cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

    # Tick marks at ends
    tick_h = bar_h + 6
    cv2.line(img, (x0, y0 - 3), (x0, y0 + tick_h), color, thickness)
    cv2.line(img, (x1, y0 - 3), (x1, y0 + tick_h), color, thickness)

    # Label
    label = f"{target_length_m:.0f} m" if target_length_m >= 1 else f"{target_length_m*100:.0f} cm"
    font_scale = max(0.5, min(1.2, bar_px / 150))
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    tx = x0 + (bar_px - text_size[0]) // 2
    ty = y0 - 8
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return img


def depth_to_colormap(depth_map: np.ndarray) -> np.ndarray:
    """Convert float32 depth map to BGR PLASMA colormap image."""
    d_norm = depth_map.copy()
    d_min, d_max = d_norm.min(), d_norm.max()
    if d_max > d_min:
        d_norm = (d_norm - d_min) / (d_max - d_min)
    d_uint8 = (d_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(d_uint8, cv2.COLORMAP_PLASMA)


def save_result(
    image_path: Path,
    gsd_m_per_px: float,
    method: str,
    output_dir: Path,
    depth_map: Optional[np.ndarray] = None,
    scale_bar_m: float = 10.0,
) -> dict[str, Path]:
    """
    Save annotated image (and optional depth map) to output_dir.

    Returns dict of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[visualize] Warning: could not read {image_path}")
        return saved

    # Annotate with scale bar
    annotated = draw_scale_bar(img, gsd_m_per_px, target_length_m=scale_bar_m)

    # GSD info text
    info = f"GSD: {gsd_m_per_px*100:.2f} cm/px | method: {method}"
    cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_img = output_dir / f"{image_path.stem}_metric.jpg"
    cv2.imwrite(str(out_img), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    saved["annotated"] = out_img

    # Depth map
    if depth_map is not None:
        depth_vis = depth_to_colormap(depth_map)
        out_depth = output_dir / f"{image_path.stem}_depth.jpg"
        cv2.imwrite(str(out_depth), depth_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["depth"] = out_depth

    return saved


def plot_gsd_timeline(
    frame_names: list[str],
    gsd_values: list[float],
    output_path: Path,
) -> None:
    """Save a GSD vs. frame index plot using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(gsd_values)), [g * 100 for g in gsd_values], "b-o", markersize=3)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("GSD (cm/px)")
        ax.set_title("Ground Sampling Distance across image sequence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=120)
        plt.close()
        print(f"[visualize] GSD timeline saved -> {output_path}")
    except ImportError:
        print("[visualize] matplotlib not available; skipping timeline plot.")
