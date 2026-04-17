from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StitchingConfig:
    data_dir: str
    output_dir: str
    image1: str
    image2: str
    resize_max_dim: int
    feature_type: str
    sift_nfeatures: int
    knn_k: int
    ratio_test: float
    ransac_reproj_threshold: float
    ransac_confidence: float
    ransac_max_iters: int
    min_inliers: int
    max_draw_matches: int


def _nested_get(data: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_config(config_path: str | Path) -> StitchingConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return StitchingConfig(
        data_dir=str(_nested_get(raw, ("paths", "data_dir"), "data")),
        output_dir=str(_nested_get(raw, ("paths", "output_dir"), "outputs/pairwise")),
        image1=str(_nested_get(raw, ("pair", "image1"), "001.JPG")),
        image2=str(_nested_get(raw, ("pair", "image2"), "002.JPG")),
        resize_max_dim=int(_nested_get(raw, ("preprocess", "resize_max_dim"), 1600)),
        feature_type=str(_nested_get(raw, ("features", "type"), "sift")).lower(),
        sift_nfeatures=int(_nested_get(raw, ("features", "sift_nfeatures"), 4000)),
        knn_k=int(_nested_get(raw, ("matching", "knn_k"), 2)),
        ratio_test=float(_nested_get(raw, ("matching", "ratio_test"), 0.75)),
        ransac_reproj_threshold=float(_nested_get(raw, ("geometry", "ransac_reproj_threshold"), 4.0)),
        ransac_confidence=float(_nested_get(raw, ("geometry", "ransac_confidence"), 0.995)),
        ransac_max_iters=int(_nested_get(raw, ("geometry", "ransac_max_iters"), 5000)),
        min_inliers=int(_nested_get(raw, ("geometry", "min_inliers"), 20)),
        max_draw_matches=int(_nested_get(raw, ("debug", "max_draw_matches"), 200)),
    )
