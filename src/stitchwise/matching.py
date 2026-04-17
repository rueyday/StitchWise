from __future__ import annotations

import cv2

from .config import StitchingConfig


def match_descriptors(desc1, desc2, cfg: StitchingConfig):
    if desc1 is None or desc2 is None:
        return [], [], []

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=cfg.knn_k)

    raw_primary = []
    good_matches = []
    for pair in knn_matches:
        if not pair:
            continue
        raw_primary.append(pair[0])
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < cfg.ratio_test * n.distance:
            good_matches.append(m)

    return knn_matches, raw_primary, good_matches
