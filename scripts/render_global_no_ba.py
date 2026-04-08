from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stitchwise.config import load_config
from stitchwise.io_utils import ensure_dir, load_image, resolve_image_path, resize_by_max_dim, save_image


def parse_index(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    if stem.isdigit():
        return int(stem), name
    return 10**9, name


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_global_bbox(nodes: list[dict]) -> tuple[float, float, float, float]:
    all_pts = []
    for n in nodes:
        shape = n.get("image_processed_shape")
        h_to_anchor = n.get("H_to_anchor")
        if shape is None or h_to_anchor is None:
            continue
        h, w = int(shape[0]), int(shape[1])
        H = np.array(h_to_anchor, dtype=np.float64)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        all_pts.append(warped)

    if not all_pts:
        raise RuntimeError("No valid node corners for global bbox.")

    pts = np.vstack(all_pts)
    x_min = float(np.floor(pts[:, 0].min()))
    y_min = float(np.floor(pts[:, 1].min()))
    x_max = float(np.ceil(pts[:, 0].max()))
    y_max = float(np.ceil(pts[:, 1].max()))
    return x_min, y_min, x_max, y_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a global mosaic from global poses (no BA).")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stitching.yaml"))
    parser.add_argument(
        "--poses-json",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "global_no_ba" / "global_poses.json"),
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "global_no_ba"),
    )
    parser.add_argument("--max-canvas-side", type=int, default=8000, help="Auto-downscale if canvas side exceeds this.")
    parser.add_argument("--max-canvas-area", type=float, default=40_000_000, help="Auto-downscale if canvas area exceeds this.")
    parser.add_argument("--preview-max-side", type=int, default=2800)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    poses_path = Path(args.poses_json)
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses file: {poses_path}")
    poses_payload = load_json(poses_path)
    nodes = poses_payload.get("nodes", [])
    if not nodes:
        raise RuntimeError("No nodes found in global_poses.json")

    output_dir = ensure_dir(Path(args.output_dir))

    x_min, y_min, x_max, y_max = compute_global_bbox(nodes)
    base_w = int(max(1.0, x_max - x_min))
    base_h = int(max(1.0, y_max - y_min))
    base_area = float(base_w * base_h)

    scale_side = min(1.0, float(args.max_canvas_side) / float(max(base_w, base_h)))
    scale_area = min(1.0, float(np.sqrt(float(args.max_canvas_area) / max(base_area, 1.0))))
    render_scale = min(scale_side, scale_area)
    final_w = int(max(1, round(base_w * render_scale)))
    final_h = int(max(1, round(base_h * render_scale)))

    T_shift_scale = np.array(
        [
            [render_scale, 0.0, -x_min * render_scale],
            [0.0, render_scale, -y_min * render_scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    try:
        accum = np.zeros((final_h, final_w, 3), dtype=np.float32)
        weights = np.zeros((final_h, final_w), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to allocate mosaic buffers for {final_w}x{final_h}: {exc}") from exc

    placed = 0
    skipped: list[dict] = []
    nodes_sorted = sorted(nodes, key=lambda n: parse_index(str(n.get("image", ""))))
    for i, n in enumerate(nodes_sorted, start=1):
        image_name = str(n.get("image"))
        shape = n.get("image_processed_shape")
        h_to_anchor = n.get("H_to_anchor")
        if not image_name or shape is None or h_to_anchor is None:
            skipped.append({"image": image_name, "reason": "missing_pose_or_shape"})
            continue

        print(f"[{i}/{len(nodes_sorted)}] warp {image_name}")
        try:
            image_path = resolve_image_path(image_name, cfg.data_dir)
            img = load_image(image_path)
            img_proc, _ = resize_by_max_dim(img, cfg.resize_max_dim)
            target_h, target_w = int(shape[0]), int(shape[1])
            if img_proc.shape[0] != target_h or img_proc.shape[1] != target_w:
                img_proc = cv2.resize(img_proc, (target_w, target_h), interpolation=cv2.INTER_AREA)

            H = np.array(h_to_anchor, dtype=np.float64)
            warp_mat = T_shift_scale @ H
            warped = cv2.warpPerspective(img_proc, warp_mat, (final_w, final_h), flags=cv2.INTER_LINEAR)

            src_mask = np.ones((img_proc.shape[0], img_proc.shape[1]), dtype=np.uint8) * 255
            mask = cv2.warpPerspective(src_mask, warp_mat, (final_w, final_h), flags=cv2.INTER_NEAREST)
            w = mask.astype(np.float32) / 255.0

            accum += warped.astype(np.float32) * w[..., None]
            weights += w
            placed += 1
        except Exception as exc:
            skipped.append({"image": image_name, "reason": str(exc)})

    denom = np.maximum(weights, 1e-6)
    mosaic = (accum / denom[..., None]).astype(np.uint8)
    alpha = np.where(weights > 0.0, 255, 0).astype(np.uint8)
    mosaic[alpha == 0] = 0

    tif_path = output_dir / "mosaic_no_ba.tif"
    preview_path = output_dir / "mosaic_no_ba_preview.jpg"
    alpha_path = output_dir / "mosaic_alpha.png"
    manifest_path = output_dir / "render_manifest.json"

    save_image(tif_path, mosaic)
    save_image(alpha_path, alpha)

    ph, pw = mosaic.shape[:2]
    pscale = min(1.0, float(args.preview_max_side) / float(max(ph, pw)))
    if pscale < 1.0:
        preview = cv2.resize(mosaic, (int(round(pw * pscale)), int(round(ph * pscale))), interpolation=cv2.INTER_AREA)
    else:
        preview = mosaic
    save_image(preview_path, preview)

    manifest = {
        "anchor": poses_payload.get("anchor"),
        "requested_nodes": len(nodes),
        "placed_nodes": placed,
        "skipped_nodes": len(skipped),
        "skipped": skipped,
        "base_canvas_width": base_w,
        "base_canvas_height": base_h,
        "base_canvas_area": base_area,
        "render_scale": render_scale,
        "final_canvas_width": final_w,
        "final_canvas_height": final_h,
        "mosaic_path": str(tif_path),
        "preview_path": str(preview_path),
        "alpha_path": str(alpha_path),
    }
    save_json(manifest_path, manifest)

    print("Global render finished.")
    print(f"Placed images: {placed}/{len(nodes)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
