from __future__ import annotations

import cv2
import numpy as np

from .config import StitchingConfig


def create_detector(cfg: StitchingConfig):
    if cfg.feature_type != "sift":
        raise ValueError(f"Unsupported feature type: {cfg.feature_type}. Only 'sift' is supported in MVP.")
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV build does not include SIFT. Please install a compatible opencv-python version.")
    return cv2.SIFT_create(nfeatures=cfg.sift_nfeatures)


def detect_and_describe(image: np.ndarray, cfg: StitchingConfig):
    detector = create_detector(cfg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors
