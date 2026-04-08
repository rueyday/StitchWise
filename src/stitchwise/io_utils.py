from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_image_path(image_ref: str | Path, data_dir: str | Path) -> Path:
    candidate = Path(image_ref)
    if candidate.exists():
        return candidate
    candidate = Path(data_dir) / str(image_ref)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Image not found: {image_ref}")


def load_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def save_image(image_path: str | Path, image: np.ndarray) -> None:
    image_path = Path(image_path)
    ensure_dir(image_path.parent)
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise ValueError(f"Failed to save image: {image_path}")


def save_json(json_path: str | Path, data: dict) -> None:
    json_path = Path(json_path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resize_by_max_dim(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    if max_dim <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    cur_max = max(h, w)
    if cur_max <= max_dim:
        return image, 1.0
    scale = max_dim / float(cur_max)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), path.name
    return 10**9, path.name


def list_images(data_dir: str | Path, extensions: tuple[str, ...]) -> list[Path]:
    data_dir = Path(data_dir)
    ext_set = {e.lower() for e in extensions}
    files = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in ext_set]
    files.sort(key=_sort_key)
    return files
