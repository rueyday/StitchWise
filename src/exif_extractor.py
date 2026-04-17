import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import exifread

SENSOR_DB: dict[str, float] = {
    "FC220":   6.17,   # Mavic Pro
    "FC330":   6.17,   # Phantom 4
    "FC350":  13.2,    # Inspire 1
    "FC6310": 13.2,    # Phantom 4 Pro
    "FC6520": 17.3,    # Zenmuse X5S
    "FC6310S":13.2,
    "FC7203":  6.3,    # Mini 2
    "FC3170":  9.6,    # Air 2S
    "FC3411":  9.6,    # Mavic 3
    "L1D-20c":17.3,    # Mavic 2 Pro
    "_default":6.17,
}


@dataclass
class CameraParams:
    altitude_m: Optional[float]
    focal_length_mm: Optional[float]
    sensor_width_mm: Optional[float]
    image_width_px: Optional[int]
    image_height_px: Optional[int]
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    source_altitude: str = "unknown"
    source_focal: str = "unknown"

    @property
    def focal_length_px(self) -> Optional[float]:
        if self.focal_length_mm and self.sensor_width_mm and self.image_width_px:
            return self.focal_length_mm / self.sensor_width_mm * self.image_width_px
        return None

    @property
    def gsd_m_per_px(self) -> Optional[float]:
        f_px = self.focal_length_px
        if self.altitude_m and f_px:
            return self.altitude_m / f_px
        return None

    def summary(self) -> str:
        lines = [
            f"  Altitude AGL  : {self.altitude_m:.2f} m  [{self.source_altitude}]" if self.altitude_m else "  Altitude AGL  : N/A",
            (f"  Focal length  : {self.focal_length_mm:.2f} mm -> {self.focal_length_px:.1f} px  [{self.source_focal}]"
             if self.focal_length_mm else "  Focal length  : N/A"),
            f"  Sensor width  : {self.sensor_width_mm:.2f} mm" if self.sensor_width_mm else "  Sensor width  : N/A",
            f"  Image size    : {self.image_width_px}×{self.image_height_px} px" if self.image_width_px else "  Image size    : N/A",
            (f"  GSD           : {self.gsd_m_per_px*100:.2f} cm/px ({self.gsd_m_per_px:.4f} m/px)"
             if self.gsd_m_per_px else "  GSD           : N/A"),
        ]
        return "\n".join(lines)


def _rational_to_float(rational) -> float:
    try:
        return float(rational)
    except Exception:
        try:
            return rational.num / rational.den
        except Exception:
            return float(str(rational).split("/")[0])


def _gps_to_decimal(values, ref: str) -> float:
    d = _rational_to_float(values[0])
    m = _rational_to_float(values[1])
    s = _rational_to_float(values[2])
    dec = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        dec = -dec
    return dec


def _extract_xmp_value(xmp_bytes: bytes, tag: str) -> Optional[float]:
    try:
        xmp_str = xmp_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(rf'{re.escape(tag)}[=:>"\s]+([+-]?\d+\.?\d*)', xmp_str)
    return float(m.group(1)) if m else None


def _read_xmp_from_file(path: Path) -> bytes:
    try:
        data = path.read_bytes()
        start = data.find(b"<x:xmpmeta")
        end   = data.find(b"</x:xmpmeta>")
        if start != -1 and end != -1:
            return data[start:end + 12]
    except Exception:
        pass
    return b""


def extract(image_path: str | Path) -> CameraParams:
    path = Path(image_path)

    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    iw = int(str(tags["EXIF ExifImageWidth"])) if "EXIF ExifImageWidth" in tags else None
    ih = int(str(tags["EXIF ExifImageLength"])) if "EXIF ExifImageLength" in tags else None
    if iw is None and "Image ImageWidth"  in tags: iw = int(str(tags["Image ImageWidth"]))
    if ih is None and "Image ImageLength" in tags: ih = int(str(tags["Image ImageLength"]))

    model = str(tags.get("Image Model", "")).strip().upper()
    make  = str(tags.get("Image Make",  "")).strip().upper()
    sensor_w: Optional[float] = None
    for key, val in SENSOR_DB.items():
        if key in model or key in make:
            sensor_w = val
            break

    focal_mm: Optional[float] = None
    source_focal = "none"
    if "EXIF FocalLength" in tags:
        focal_mm     = _rational_to_float(tags["EXIF FocalLength"].values[0])
        source_focal = "EXIF FocalLength"
    if focal_mm and sensor_w is None and "EXIF FocalLengthIn35mmFilm" in tags:
        fl35 = float(str(tags["EXIF FocalLengthIn35mmFilm"]))
        if fl35 > 0:
            sensor_w      = focal_mm / fl35 * 36.0
            source_focal += " + sensor inferred from 35mm equiv"
    if focal_mm is None and "EXIF FocalLengthIn35mmFilm" in tags:
        fl35         = float(str(tags["EXIF FocalLengthIn35mmFilm"]))
        focal_mm     = fl35
        sensor_w     = sensor_w or 36.0
        source_focal = "35mm equiv (full-frame assumed)"
    if sensor_w is None:
        sensor_w      = SENSOR_DB["_default"]
        source_focal += " + default sensor (1/2.3\")"

    altitude_m: Optional[float] = None
    source_alt = "none"
    xmp = _read_xmp_from_file(path)
    for xmp_tag in ["drone-dji:RelativeAltitude", "Camera:AboveGroundAltitude", "drone:AboveGroundAltitude"]:
        val = _extract_xmp_value(xmp, xmp_tag)
        if val is not None:
            altitude_m = abs(val)
            source_alt = f"XMP {xmp_tag}"
            break
    if altitude_m is None and "GPS GPSAltitude" in tags:
        gps_alt    = _rational_to_float(tags["GPS GPSAltitude"].values[0])
        ref        = str(tags.get("GPS GPSAltitudeRef", "0"))
        altitude_m = gps_alt if ref == "0" else -gps_alt
        source_alt = "GPS GPSAltitude (sea-level)"

    gps_lat = gps_lon = None
    if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
        gps_lat = _gps_to_decimal(tags["GPS GPSLatitude"].values,  str(tags.get("GPS GPSLatitudeRef",  "N")))
        gps_lon = _gps_to_decimal(tags["GPS GPSLongitude"].values, str(tags.get("GPS GPSLongitudeRef", "E")))

    return CameraParams(
        altitude_m=altitude_m,
        focal_length_mm=focal_mm,
        sensor_width_mm=sensor_w,
        image_width_px=iw,
        image_height_px=ih,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        source_altitude=source_alt,
        source_focal=source_focal,
    )
