from __future__ import annotations

from pathlib import Path

from .blending import blend_images
from .config import StitchingConfig
from .debug_viz import draw_matches
from .features import detect_and_describe
from .geometry import estimate_homography
from .io_utils import (
    ensure_dir,
    load_image,
    resolve_image_path,
    resize_by_max_dim,
    save_image,
    save_json,
)
from .matching import match_descriptors
from .warping import warp_two_images


def _should_compute_warp_preview(cfg: StitchingConfig) -> bool:
    # Optional runtime switch used by graph-building/evaluation scripts.
    return bool(getattr(cfg, "compute_warp_preview", True))


def _run_pairwise_debug(cfg: StitchingConfig) -> dict:
    # Resolve input image paths (supports absolute path or filename under data_dir).
    image1_path = resolve_image_path(cfg.image1, cfg.data_dir)
    image2_path = resolve_image_path(cfg.image2, cfg.data_dir)

    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Resize for faster debugging/iteration while preserving aspect ratio.
    img1_proc, scale1 = resize_by_max_dim(img1, cfg.resize_max_dim)
    img2_proc, scale2 = resize_by_max_dim(img2, cfg.resize_max_dim)

    # Feature extraction and descriptor computation.
    kp1, desc1 = detect_and_describe(img1_proc, cfg)
    kp2, desc2 = detect_and_describe(img2_proc, cfg)

    # KNN matching + Lowe ratio filtering.
    knn_matches, raw_primary, good_matches = match_descriptors(desc1, desc2, cfg)

    H_2_to_1 = None
    inlier_mask = []
    inlier_matches = []
    failure_reason = ""

    if len(good_matches) < 4:
        failure_reason = f"Not enough good matches for homography: {len(good_matches)}"
    else:
        # Robust homography estimation with RANSAC.
        H_2_to_1, inlier_mask = estimate_homography(kp1, kp2, good_matches, cfg)
        if H_2_to_1 is None:
            failure_reason = "Homography estimation failed."
        else:
            inlier_matches = [m for m, keep in zip(good_matches, inlier_mask) if keep]
            inlier_count = int(sum(inlier_mask))
            if inlier_count < cfg.min_inliers:
                failure_reason = f"Inlier count too low: {inlier_count} < min_inliers ({cfg.min_inliers})"

    if inlier_matches:
        inlier_count = len(inlier_matches)
    else:
        inlier_count = int(sum(inlier_mask)) if len(inlier_mask) else 0
    inlier_ratio = float(inlier_count / max(len(good_matches), 1))

    compute_warp_preview = _should_compute_warp_preview(cfg)
    warping_preview = None
    if H_2_to_1 is not None and compute_warp_preview:
        # Warp image2 into image1 coordinates, then blend.
        try:
            base_canvas, warped_image, mask_base, mask_warped = warp_two_images(img1_proc, img2_proc, H_2_to_1)
            warping_preview = blend_images(base_canvas, warped_image, mask_base, mask_warped, method="feather")
        except Exception as exc:
            if not failure_reason:
                failure_reason = f"Warping failed: {exc}"

    # Save debug visualizations for fast failure analysis.
    raw_vis = draw_matches(img1_proc, kp1, img2_proc, kp2, raw_primary, max_draw=cfg.max_draw_matches)
    good_vis = draw_matches(img1_proc, kp1, img2_proc, kp2, good_matches, max_draw=cfg.max_draw_matches)
    inlier_vis = draw_matches(img1_proc, kp1, img2_proc, kp2, inlier_matches, max_draw=cfg.max_draw_matches)

    if compute_warp_preview:
        success = (failure_reason == "") and (warping_preview is not None)
    else:
        success = (failure_reason == "") and (H_2_to_1 is not None)
    stats_core = {
        "image1": str(image1_path),
        "image2": str(image2_path),
        "image1_processed_shape": list(img1_proc.shape),
        "image2_processed_shape": list(img2_proc.shape),
        "resize_scale_image1": scale1,
        "resize_scale_image2": scale2,
        "keypoints_image1": len(kp1),
        "keypoints_image2": len(kp2),
        "raw_match_count": len(knn_matches),
        "good_match_count": len(good_matches),
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "homography_2_to_1": H_2_to_1.tolist() if H_2_to_1 is not None else None,
        "warp_preview_computed": compute_warp_preview,
        "success_or_failure": "success" if success else "failure",
        "failure_reason": "" if success else failure_reason,
    }
    return {
        "success": success,
        "failure_reason": failure_reason,
        "stats_core": stats_core,
        "raw_vis": raw_vis,
        "good_vis": good_vis,
        "inlier_vis": inlier_vis,
        "warping_preview": warping_preview,
    }


def _save_pair_outputs(cfg: StitchingConfig, debug: dict) -> dict:
    image1_name = Path(cfg.image1).stem
    image2_name = Path(cfg.image2).stem
    pair_name = f"{image1_name}_{image2_name}"
    out_dir = ensure_dir(Path(cfg.output_dir) / pair_name)

    raw_path = out_dir / "raw_matches.jpg"
    good_path = out_dir / "good_matches.jpg"
    inlier_path = out_dir / "inlier_matches.jpg"
    preview_path = out_dir / "warping_preview.jpg"
    stitched_path = out_dir / "stitched.jpg"
    stats_path = out_dir / "stats.json"

    save_image(raw_path, debug["raw_vis"])
    save_image(good_path, debug["good_vis"])
    save_image(inlier_path, debug["inlier_vis"])

    preview = debug["warping_preview"] if debug["warping_preview"] is not None else debug["good_vis"]
    save_image(preview_path, preview)
    save_image(stitched_path, preview)

    stats = dict(debug["stats_core"])
    stats.update(
        {
            "output_dir": str(out_dir),
            "raw_matches_viz": str(raw_path),
            "good_matches_viz": str(good_path),
            "inlier_matches_viz": str(inlier_path),
            "warping_preview": str(preview_path),
            "stitched_result": str(stitched_path),
        }
    )
    save_json(stats_path, stats)
    return stats


def stitch_pair(cfg: StitchingConfig) -> dict:
    debug = _run_pairwise_debug(cfg)
    stats = _save_pair_outputs(cfg, debug)
    if not debug["success"]:
        raise RuntimeError(debug["failure_reason"])
    return stats
