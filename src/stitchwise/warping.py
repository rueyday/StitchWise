from __future__ import annotations

import cv2
import numpy as np


def warp_two_images(base_image: np.ndarray, image_to_warp: np.ndarray, H_2_to_1: np.ndarray):
    h1, w1 = base_image.shape[:2]
    h2, w2 = image_to_warp.shape[:2]

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H_2_to_1)

    # Compute output canvas bounds by combining base corners and transformed corners.
    all_corners = np.vstack((corners1, warped_corners2)).reshape(-1, 2)
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

    # Translate coordinates when transformed corners are negative.
    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0
    translation = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)

    out_w = int(x_max - x_min)
    out_h = int(y_max - y_min)
    out_size = (out_w, out_h)

    base_canvas = np.zeros((out_h, out_w, 3), dtype=base_image.dtype)
    base_canvas[ty : ty + h1, tx : tx + w1] = base_image

    mask_base = np.zeros((out_h, out_w), dtype=np.uint8)
    mask_base[ty : ty + h1, tx : tx + w1] = 255

    # Warp the second image and its binary mask into the shared canvas.
    warp_matrix = translation @ H_2_to_1
    warped_image = cv2.warpPerspective(image_to_warp, warp_matrix, out_size, flags=cv2.INTER_LINEAR)

    src_mask = np.ones((h2, w2), dtype=np.uint8) * 255
    mask_warped = cv2.warpPerspective(src_mask, warp_matrix, out_size, flags=cv2.INTER_NEAREST)

    return base_canvas, warped_image, mask_base, mask_warped
