from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from src.exif_extractor import CameraParams, extract as extract_exif


class ScaleMethod(Enum):
    EXIF_FULL           = "exif_full"
    EXIF_FOCAL_DEPTH_ALT = "exif_focal_depth_alt"
    DEPTH_FULL          = "depth_full"
    UNKNOWN             = "unknown"


@dataclass
class ScaleResult:
    gsd_m_per_px:    float
    altitude_m:      float
    focal_length_px: Optional[float]
    method:          ScaleMethod
    depth_map:       Optional[np.ndarray]
    camera_params:   CameraParams

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
    path = Path(image_path)
    params = extract_exif(path)

    altitude = force_altitude_m or params.altitude_m
    focal_px = params.focal_length_px

    if altitude and focal_px:
        return ScaleResult(
            gsd_m_per_px=altitude / focal_px,
            altitude_m=altitude,
            focal_length_px=focal_px,
            method=ScaleMethod.EXIF_FULL,
            depth_map=None,
            camera_params=params,
        )

    if focal_px and not altitude and use_depth_fallback:
        from src.depth_model import estimate_depth, estimate_altitude_from_depth
        depth_map = estimate_depth(path)
        altitude  = estimate_altitude_from_depth(depth_map)
        return ScaleResult(
            gsd_m_per_px=altitude / focal_px,
            altitude_m=altitude,
            focal_length_px=focal_px,
            method=ScaleMethod.EXIF_FOCAL_DEPTH_ALT,
            depth_map=depth_map,
            camera_params=params,
        )

    if use_depth_fallback:
        from src.depth_model import estimate_gsd
        gsd, depth_alt, depth_map = estimate_gsd(
            path,
            focal_length_mm=params.focal_length_mm,
            sensor_width_mm=params.sensor_width_mm,
            image_width_px=params.image_width_px,
        )
        if altitude and depth_alt > 0:
            gsd = altitude / depth_alt * gsd
        return ScaleResult(
            gsd_m_per_px=gsd,
            altitude_m=altitude or depth_alt,
            focal_length_px=None,
            method=ScaleMethod.DEPTH_FULL,
            depth_map=depth_map,
            camera_params=params,
        )

    raise ValueError(
        f"Cannot estimate metric scale for {path.name}: EXIF extrinsics incomplete "
        "and depth fallback disabled."
    )
