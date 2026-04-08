"""
Metric scale estimation for drone nadir imagery.

Strategy (in priority order):
  1. EXIF/XMP extrinsics -> GSD directly
  2. EXIF focal length only (no altitude) -> depth model for altitude
  3. EXIF altitude only (no focal length) -> depth model for focal length confirmation
  4. No extrinsics -> full depth model estimation

GSD (Ground Sampling Distance) in meters/pixel is the key output.
For a nadir view:  GSD = H / f_px  where H = AGL altitude, f_px = focal length in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from src.exif_extractor import CameraParams, extract as extract_exif


class ScaleMethod(Enum):
    EXIF_FULL = "exif_full"           # altitude + focal from EXIF
    EXIF_FOCAL_DEPTH_ALT = "exif_focal_depth_alt"   # focal from EXIF, alt from depth
    EXIF_ALT_DEPTH_FOCAL = "exif_alt_depth_focal"   # alt from EXIF, focal from depth (not used)
    DEPTH_FULL = "depth_full"         # everything from depth model
    UNKNOWN = "unknown"


@dataclass
class ScaleResult:
    gsd_m_per_px: float               # meters per pixel (the key metric)
    altitude_m: float                 # estimated AGL altitude in meters
    focal_length_px: Optional[float]  # focal length in pixels (if known)
    method: ScaleMethod
    depth_map: Optional[np.ndarray]   # depth map if depth model was used
    camera_params: CameraParams       # raw EXIF params

    @property
    def cm_per_px(self) -> float:
        return self.gsd_m_per_px * 100

    def summary(self) -> str:
        lines = [
            f"  Method        : {self.method.value}",
            f"  GSD           : {self.cm_per_px:.3f} cm/px  ({self.gsd_m_per_px:.5f} m/px)",
            f"  Altitude AGL  : {self.altitude_m:.2f} m",
        ]
        if self.focal_length_px:
            lines.append(f"  Focal (px)    : {self.focal_length_px:.1f} px")
        lines.append("  EXIF details  :")
        lines.append(self.camera_params.summary())
        return "\n".join(lines)


def estimate(
    image_path: str | Path,
    force_altitude_m: Optional[float] = None,
    use_depth_fallback: bool = True,
) -> ScaleResult:
    """
    Estimate metric scale (GSD) for a single drone nadir image.

    Args:
        image_path: Path to the image.
        force_altitude_m: Override altitude (m AGL) if EXIF is missing/wrong.
        use_depth_fallback: Whether to invoke Depth Anything V2 when EXIF is incomplete.

    Returns:
        ScaleResult with GSD and supporting metadata.
    """
    path = Path(image_path)
    params = extract_exif(path)

    altitude = force_altitude_m or params.altitude_m
    focal_px = params.focal_length_px

    depth_map: Optional[np.ndarray] = None

    # --- Case 1: full extrinsics available ---
    if altitude and focal_px:
        gsd = altitude / focal_px
        return ScaleResult(
            gsd_m_per_px=gsd,
            altitude_m=altitude,
            focal_length_px=focal_px,
            method=ScaleMethod.EXIF_FULL,
            depth_map=None,
            camera_params=params,
        )

    # --- Case 2: focal length known, altitude missing -> depth for altitude ---
    if focal_px and not altitude and use_depth_fallback:
        from src.depth_model import estimate_depth, estimate_altitude_from_depth
        print(f"[metric_scale] No altitude in EXIF; using depth model for altitude.")
        depth_map = estimate_depth(path)
        altitude = estimate_altitude_from_depth(depth_map)
        gsd = altitude / focal_px
        return ScaleResult(
            gsd_m_per_px=gsd,
            altitude_m=altitude,
            focal_length_px=focal_px,
            method=ScaleMethod.EXIF_FOCAL_DEPTH_ALT,
            depth_map=depth_map,
            camera_params=params,
        )

    # --- Case 3: altitude known, no focal length -> use depth GSD estimation ---
    if altitude and not focal_px and use_depth_fallback:
        from src.depth_model import estimate_gsd
        print(f"[metric_scale] No focal length; using depth model with known altitude {altitude:.1f}m.")
        gsd, depth_alt, depth_map = estimate_gsd(
            path,
            focal_length_mm=params.focal_length_mm,
            sensor_width_mm=params.sensor_width_mm,
            image_width_px=params.image_width_px,
        )
        # Trust the EXIF altitude over depth-derived altitude when both are available
        gsd_corrected = altitude / depth_alt * gsd if depth_alt > 0 else gsd
        return ScaleResult(
            gsd_m_per_px=gsd_corrected,
            altitude_m=altitude,
            focal_length_px=None,
            method=ScaleMethod.DEPTH_FULL,
            depth_map=depth_map,
            camera_params=params,
        )

    # --- Case 4: no extrinsics at all -> full depth estimation ---
    if use_depth_fallback:
        from src.depth_model import estimate_gsd
        print("[metric_scale] No EXIF extrinsics; falling back to full depth model estimation.")
        gsd, depth_alt, depth_map = estimate_gsd(
            path,
            focal_length_mm=params.focal_length_mm,
            sensor_width_mm=params.sensor_width_mm,
            image_width_px=params.image_width_px,
        )
        return ScaleResult(
            gsd_m_per_px=gsd,
            altitude_m=depth_alt,
            focal_length_px=None,
            method=ScaleMethod.DEPTH_FULL,
            depth_map=depth_map,
            camera_params=params,
        )

    raise ValueError(
        f"Cannot estimate metric scale for {path.name}: EXIF extrinsics are incomplete "
        "and depth fallback is disabled. Provide --altitude or enable depth fallback."
    )
