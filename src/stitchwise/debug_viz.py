from __future__ import annotations

import cv2
import numpy as np


def draw_matches(image1, kp1, image2, kp2, matches, max_draw: int = 200):
    if not matches:
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        h = max(h1, h2)
        canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = image1
        canvas[:h2, w1 : w1 + w2] = image2
        return canvas

    matches_to_draw = matches[: max_draw if max_draw > 0 else len(matches)]
    vis = cv2.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis
