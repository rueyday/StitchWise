from __future__ import annotations

import numpy as np


def blend_images(base_canvas, warped_image, mask_base, mask_warped, method: str = "feather"):
    if method != "feather":
        raise ValueError(f"Unsupported blend method: {method}. Only 'feather' is supported in MVP.")

    w1 = (mask_base.astype(np.float32) / 255.0)[..., None]
    w2 = (mask_warped.astype(np.float32) / 255.0)[..., None]
    denom = w1 + w2
    denom_safe = np.where(denom == 0.0, 1.0, denom)

    blended = (base_canvas.astype(np.float32) * w1 + warped_image.astype(np.float32) * w2) / denom_safe
    blended[denom.squeeze(-1) == 0.0] = 0.0
    return np.clip(blended, 0, 255).astype(np.uint8)
