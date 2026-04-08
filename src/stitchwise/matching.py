from __future__ import annotations

import cv2

from .config import StitchingConfig


def match_descriptors(desc1, desc2, cfg: StitchingConfig):
    if desc1 is None or desc2 is None:
        return [], [], []

    # For SIFT descriptors we use L2 distance with KNN search.
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=cfg.knn_k)

    raw_primary = []
    good_matches = []
    for pair in knn_matches:
        if not pair:
            continue
        # Keep the first neighbor for "raw match" visualization.
        raw_primary.append(pair[0])
        if len(pair) < 2:
            continue
        m, n = pair
        # Lowe ratio test removes many ambiguous correspondences.
        if m.distance < cfg.ratio_test * n.distance:
            good_matches.append(m)

    return knn_matches, raw_primary, good_matches
