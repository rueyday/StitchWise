from __future__ import annotations

import cv2
import numpy as np

from .config import StitchingConfig


def estimate_homography(kp1, kp2, matches, cfg: StitchingConfig):
    if len(matches) < 4:
        return None, np.array([], dtype=bool)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # H maps image2 -> image1 so image2 can be warped into image1's coordinate frame.
    H, mask = cv2.findHomography(
        pts2,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_reproj_threshold,
        maxIters=cfg.ransac_max_iters,
        confidence=cfg.ransac_confidence,
    )
    if H is None or mask is None:
        return None, np.array([], dtype=bool)
    return H, mask.ravel().astype(bool)
