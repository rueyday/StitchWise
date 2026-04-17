"""
Depth Anything V2 (Metric Outdoor) wrapper for metric scale estimation.

Model: depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
  - Returns depth in meters
  - Trained on outdoor scenes including aerial-adjacent views
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

_pipeline = None
_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            import sys
            sys.path.insert(0, "C:/torch_pkg")
            from transformers import pipeline as hf_pipeline
            print(f"[depth_model] Loading {_MODEL_ID} ...")
            _pipeline = hf_pipeline(
                "depth-estimation",
                model=_MODEL_ID,
            )
            print("[depth_model] Model loaded.")
        except ImportError as e:
            raise RuntimeError(
                "PyTorch + transformers are required for depth estimation. "
                f"Install them and retry. Original error: {e}"
            )
    return _pipeline


def estimate_depth(image_path: str | Path) -> np.ndarray:
    """
    Run Depth Anything V2 Metric Outdoor on an image.

    Returns a float32 depth map (H×W) in meters.
    """
    from PIL import Image as PILImage
    pipe = _load_pipeline()
    img = PILImage.open(image_path).convert("RGB")
    result = pipe(img)
    depth = np.array(result["depth"], dtype=np.float32)
    return depth


def estimate_altitude_from_depth(depth_map: np.ndarray, center_fraction: float = 0.5) -> float:
    h, w = depth_map.shape
    pad_h = int(h * (1 - center_fraction) / 2)
    pad_w = int(w * (1 - center_fraction) / 2)
    center = depth_map[pad_h: h - pad_h, pad_w: w - pad_w]
    return float(np.median(center))


def estimate_gsd(
    image_path: str | Path,
    focal_length_mm: Optional[float] = None,
    sensor_width_mm: Optional[float] = None,
    image_width_px: Optional[int] = None,
) -> tuple[float, float, np.ndarray]:
    """
    Estimate GSD using Depth Anything V2 Metric.

    Returns:
        (gsd_m_per_px, altitude_m, depth_map)
    """
    depth_map = estimate_depth(image_path)
    altitude_m = estimate_altitude_from_depth(depth_map)

    if focal_length_mm and sensor_width_mm and image_width_px:
        focal_px = focal_length_mm / sensor_width_mm * image_width_px
        gsd = altitude_m / focal_px
    else:
        import math
        assumed_hfov_rad = math.radians(90)
        scene_width_m = 2 * altitude_m * math.tan(assumed_hfov_rad / 2)
        if image_width_px:
            gsd = scene_width_m / image_width_px
        else:
            raise ValueError(
                "Cannot estimate GSD: missing focal length and image width."
            )

    return gsd, altitude_m, depth_map
