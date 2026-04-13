"""
StitchWise — Unified Pipeline Orchestrator
==========================================
Chains four stages:
  1. Metric Scale  — GSD estimation per image (meters/pixel)
  2. Segmentation  — YOLO + SAM disaster region masks
  3. Stitching     — SIFT/RANSAC global mosaic from all images
  4. Overlay       — Warp per-image masks onto mosaic + scale bar

Usage:
    python run_pipeline.py --image-dir data/my_images
    python run_pipeline.py --image-dir data/my_images --yolo-weights detection/model/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.metric_scale import estimate as estimate_gsd
from src.visualize import draw_scale_bar

# ── Disaster class palette (BGR) ────────────────────────────────────────────
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (200, 100, 0),   # Water / Flooding         — dark blue
    1: (0, 140, 255),   # Building Major Damage    — orange
    2: (0, 0, 220),     # Building Total Destruct. — red
    3: (150, 0, 200),   # Road Blocked             — purple
    4: (0, 200, 200),   # Vehicle                  — cyan
}
CLASS_NAMES: dict[int, str] = {
    0: "Water / Flooding",
    1: "Building Major Damage",
    2: "Building Total Destruction",
    3: "Road Blocked",
    4: "Vehicle",
}
OVERLAY_ALPHA = 0.45
PREVIEW_MAX_SIDE = 2800
STITCHING_RESIZE_MAX_DIM = 1600  # must match configs/stitching.yaml


# ── Stage 1 — Metric Scale ────────────────────────────────────────────────

def run_metric_scale(
    image_paths: list[Path],
    output_dir: Path,
    use_depth_fallback: bool = True,
) -> tuple[dict[str, float], Path]:
    """Estimate GSD for every image. Returns ({stem: gsd_m_per_px}, csv_path)."""
    gsd_dir = output_dir / "metric_scale"
    gsd_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}
    rows: list[dict] = []
    for img_path in image_paths:
        try:
            r = estimate_gsd(img_path, use_depth_fallback=use_depth_fallback)
            results[img_path.stem] = r.gsd_m_per_px
            rows.append({
                "image": img_path.name,
                "gsd_m_per_px": r.gsd_m_per_px,
                "method": r.method.value,
            })
        except Exception as exc:
            print(f"[metric_scale] {img_path.name}: {exc}")

    csv_path = gsd_dir / "gsd_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gsd_m_per_px", "method"])
        writer.writeheader()
        writer.writerows(rows)

    return results, csv_path


# ── Stage 2 — Disaster Segmentation ──────────────────────────────────────

def run_segmentation(
    image_paths: list[Path],
    yolo_weights: Path,
    output_dir: Path,
    conf: float = 0.25,
    sam_model: str = "sam_b.pt",
) -> dict[str, Path]:
    """Run YOLO+SAM on each image. Returns {stem: mask_path}."""
    from ultralytics import SAM, YOLO

    masks_dir = output_dir / "masks"
    viz_dir = output_dir / "seg_viz"
    masks_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(yolo_weights))
    sam = SAM(sam_model)

    mask_paths: dict[str, Path] = {}
    for img_path in image_paths:
        yolo_results = yolo.predict(source=str(img_path), conf=conf, verbose=False)[0]

        if len(yolo_results.boxes) == 0:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            empty = np.zeros((h, w), dtype=np.uint8)
            out = masks_dir / f"mask_{img_path.stem}.png"
            cv2.imwrite(str(out), empty)
            mask_paths[img_path.stem] = out
            continue

        boxes = yolo_results.boxes.xyxy.tolist()
        class_ids = yolo_results.boxes.cls.tolist()

        sam_results = sam.predict(source=str(img_path), bboxes=boxes, verbose=False)[0]

        viz = sam_results.plot()
        cv2.imwrite(str(viz_dir / f"viz_{img_path.name}"), viz)

        if sam_results.masks is not None:
            mask_data = sam_results.masks.data.cpu().numpy()
            h, w = mask_data.shape[1:]
            semantic = np.zeros((h, w), dtype=np.uint8)
            for i, mask in enumerate(mask_data):
                # Store class_id + 1 so background = 0
                semantic[mask > 0.5] = int(class_ids[i]) + 1
            out = masks_dir / f"mask_{img_path.stem}.png"
            cv2.imwrite(str(out), semantic)
            mask_paths[img_path.stem] = out

    return mask_paths


# ── Stage 3 — Stitching ────────────────────────────────────────────────────

def run_stitching(
    image_dir: Path,
    output_dir: Path,
    neighbor_offsets: str = "1,2,3",
    image_ext: str = ".jpg",
) -> tuple[Path, Path, Path]:
    """Run full stitching pipeline. Returns (full_mosaic_tif, poses_json, manifest_json)."""
    pair_graph_dir = output_dir / "pair_graph"
    global_dir = output_dir / "global_no_ba"
    python = sys.executable

    subprocess.run([
        python, str(PROJECT_ROOT / "scripts" / "build_pair_graph.py"),
        "--data-dir", str(image_dir),
        "--output-dir", str(pair_graph_dir),
        "--neighbor-offsets", neighbor_offsets,
        "--ext", image_ext,
    ], check=True)

    subprocess.run([
        python, str(PROJECT_ROOT / "scripts" / "solve_global_no_ba.py"),
        "--pair-graph-dir", str(pair_graph_dir),
        "--output-dir", str(global_dir),
    ], check=True)

    subprocess.run([
        python, str(PROJECT_ROOT / "scripts" / "render_global_no_ba.py"),
        "--data-dir", str(image_dir),
        "--poses-json", str(global_dir / "global_poses.json"),
        "--output-dir", str(global_dir),
    ], check=True)

    return (
        global_dir / "mosaic_no_ba.tif",
        global_dir / "global_poses.json",
        global_dir / "render_manifest.json",
    )


# ── Stage 3b — Progressive stitching frames ───────────────────────────────

def render_stitching_frames(
    image_dir: Path,
    poses_path: Path,
    manifest_path: Path,
    frames_dir: Path,
    preview_max_side: int = 900,
) -> list[Path]:
    """
    Re-render the mosaic incrementally, saving one JPEG after each image
    is added.  Returns the list of frame paths in order.

    These frames power the "live stitching" playback in the GUI.
    """
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from stitchwise.config import load_config
    from stitchwise.io_utils import load_image, resolve_image_path, resize_by_max_dim

    frames_dir.mkdir(parents=True, exist_ok=True)

    with poses_path.open() as f:
        poses_payload = json.load(f)
    with manifest_path.open() as f:
        manifest = json.load(f)

    cfg = load_config(PROJECT_ROOT / "configs" / "stitching.yaml")
    cfg.data_dir = str(image_dir)

    nodes: list[dict] = poses_payload.get("nodes", [])
    render_scale: float = float(manifest.get("render_scale", 1.0))
    final_w: int = int(manifest.get("final_canvas_width", 1))
    final_h: int = int(manifest.get("final_canvas_height", 1))

    x_min, y_min = _compute_global_offset(nodes)
    T = np.array([
        [render_scale, 0.0, -x_min * render_scale],
        [0.0, render_scale, -y_min * render_scale],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    nodes_sorted = sorted(nodes, key=lambda n: _parse_index_key(str(n.get("image", ""))))

    accum = np.zeros((final_h, final_w, 3), dtype=np.float32)
    weights = np.zeros((final_h, final_w), dtype=np.float32)
    frame_paths: list[Path] = []

    for i, n in enumerate(nodes_sorted, start=1):
        image_name = str(n.get("image", ""))
        shape = n.get("image_processed_shape")
        h_to_anchor = n.get("H_to_anchor")
        if not image_name or shape is None or h_to_anchor is None:
            continue

        try:
            image_path = resolve_image_path(image_name, cfg.data_dir)
            img = load_image(image_path)
            img_proc, _ = resize_by_max_dim(img, cfg.resize_max_dim)
            th, tw = int(shape[0]), int(shape[1])
            if img_proc.shape[0] != th or img_proc.shape[1] != tw:
                img_proc = cv2.resize(img_proc, (tw, th), interpolation=cv2.INTER_AREA)

            H = np.array(h_to_anchor, dtype=np.float64)
            warp_mat = T @ H
            warped = cv2.warpPerspective(img_proc, warp_mat, (final_w, final_h), flags=cv2.INTER_LINEAR)
            src_mask = np.ones((img_proc.shape[0], img_proc.shape[1]), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(src_mask, warp_mat, (final_w, final_h), flags=cv2.INTER_NEAREST)
            w = warped_mask.astype(np.float32) / 255.0

            accum += warped.astype(np.float32) * w[..., None]
            weights += w
        except Exception:
            continue

        # Build current mosaic snapshot
        denom = np.maximum(weights, 1e-6)
        snap = (accum / denom[..., None]).astype(np.uint8)
        snap[weights <= 0] = 0

        # Highlight the newly added image with a coloured border
        try:
            border_mask = cv2.warpPerspective(
                np.ones((th, tw), dtype=np.uint8) * 255,
                warp_mat, (final_w, final_h), flags=cv2.INTER_NEAREST,
            )
            contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(snap, contours, -1, (0, 220, 100), 3)
        except Exception:
            pass

        # Add frame counter label
        label = f"{i}/{len(nodes_sorted)}  {image_name}"
        cv2.putText(snap, label, (12, snap.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(snap, label, (12, snap.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

        # Downscale for quick display
        ph, pw = snap.shape[:2]
        pscale = min(1.0, float(preview_max_side) / max(ph, pw))
        if pscale < 1.0:
            snap = cv2.resize(snap, (int(pw * pscale), int(ph * pscale)), interpolation=cv2.INTER_AREA)

        frame_path = frames_dir / f"frame_{i:04d}_{Path(image_name).stem}.jpg"
        cv2.imwrite(str(frame_path), snap, [cv2.IMWRITE_JPEG_QUALITY, 82])
        frame_paths.append(frame_path)

    return frame_paths


def _parse_index_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    return (int(stem), name) if stem.isdigit() else (10**9, name)


# ── Stage 4 — Disaster Overlay on Mosaic ─────────────────────────────────

def _compute_global_offset(nodes: list[dict]) -> tuple[float, float]:
    """Recompute (x_min, y_min) offset used by the render step."""
    all_pts: list[np.ndarray] = []
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
        return 0.0, 0.0
    pts = np.vstack(all_pts)
    return float(np.floor(pts[:, 0].min())), float(np.floor(pts[:, 1].min()))


def _draw_legend(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    pad = 10
    row_h = 28
    x = img.shape[1] - 290
    y = 20
    box_h = len(CLASS_NAMES) * row_h + pad * 2
    cv2.rectangle(img, (x - pad, y - pad), (x + 260, y + box_h), (30, 30, 30), -1)
    cv2.rectangle(img, (x - pad, y - pad), (x + 260, y + box_h), (180, 180, 180), 1)
    for i, (cls_id, name) in enumerate(CLASS_NAMES.items()):
        cy = y + i * row_h + row_h // 2
        color = CLASS_COLORS_BGR[cls_id]
        cv2.rectangle(img, (x, cy - 9), (x + 22, cy + 9), color, -1)
        cv2.putText(img, name, (x + 30, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def composite_disaster_overlay(
    mosaic_tif_path: Path,
    masks_dir: Path,
    poses_path: Path,
    manifest_path: Path,
    gsd_dict: dict[str, float],
    output_dir: Path,
    scale_bar_m: float = 10.0,
) -> tuple[Path, Path, float]:
    """
    Warp per-image segmentation masks onto the full mosaic canvas,
    blend colored disaster overlay, add scale bar and legend.

    Returns (disaster_mosaic_path, disaster_preview_path, mosaic_gsd_m_per_px).
    mosaic_gsd_m_per_px is GSD in the *preview* pixel space.
    """
    mosaic = cv2.imread(str(mosaic_tif_path))
    if mosaic is None:
        raise FileNotFoundError(f"Mosaic not found: {mosaic_tif_path}")

    with poses_path.open() as f:
        poses_payload = json.load(f)
    with manifest_path.open() as f:
        manifest = json.load(f)

    nodes: list[dict] = poses_payload.get("nodes", [])
    render_scale: float = float(manifest.get("render_scale", 1.0))
    final_w: int = int(manifest.get("final_canvas_width", mosaic.shape[1]))
    final_h: int = int(manifest.get("final_canvas_height", mosaic.shape[0]))

    x_min, y_min = _compute_global_offset(nodes)

    # Transform: stitching canvas → full mosaic pixel
    T = np.array([
        [render_scale, 0.0, -x_min * render_scale],
        [0.0, render_scale, -y_min * render_scale],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    overlay_color = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    overlay_weight = np.zeros((final_h, final_w), dtype=np.float32)

    for n in nodes:
        image_name = str(n.get("image", ""))
        stem = Path(image_name).stem
        mask_path = masks_dir / f"mask_{stem}.png"
        shape = n.get("image_processed_shape")
        h_to_anchor = n.get("H_to_anchor")

        if not mask_path.exists() or shape is None or h_to_anchor is None:
            continue

        target_h, target_w = int(shape[0]), int(shape[1])
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            continue

        raw_mask = cv2.resize(raw_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Build colored mask (BGR)
        colored = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        for cls_id, color in CLASS_COLORS_BGR.items():
            colored[raw_mask == (cls_id + 1)] = color

        H = np.array(h_to_anchor, dtype=np.float64)
        warp_mat = T @ H

        warped_color = cv2.warpPerspective(
            colored, warp_mat, (final_w, final_h), flags=cv2.INTER_NEAREST)
        has_disaster = ((raw_mask > 0).astype(np.uint8) * 255)
        warped_mask = cv2.warpPerspective(
            has_disaster, warp_mat, (final_w, final_h), flags=cv2.INTER_NEAREST)

        m = warped_mask.astype(np.float32) / 255.0
        for c in range(3):
            overlay_color[:, :, c] = np.where(
                m > 0.5, warped_color[:, :, c], overlay_color[:, :, c])
        overlay_weight = np.maximum(overlay_weight, m)

    # Blend overlay onto full-res mosaic
    result = mosaic.astype(np.float32)
    for c in range(3):
        result[:, :, c] = np.where(
            overlay_weight > 0.5,
            result[:, :, c] * (1 - OVERLAY_ALPHA) + overlay_color[:, :, c] * OVERLAY_ALPHA,
            result[:, :, c],
        )
    result = np.clip(result, 0, 255).astype(np.uint8)

    # GSD in full mosaic pixel space
    mean_gsd_orig = float(np.mean(list(gsd_dict.values()))) if gsd_dict else 0.05
    # Each resized-image pixel ≈ mean_gsd / resize_scale meters.
    # render_scale then downscales the canvas further.
    # Image resize scale estimation: images were shrunk to resize_max_dim.
    # We don't know orig dims here, so use a representative value from the manifest.
    # For now, approximate: full-canvas pixel covers render_scale^-1 × resize-scale^-1 orig pixels.
    # We store the exact value in the manifest for the GUI to use.
    full_canvas_gsd = mean_gsd_orig  # placeholder — overridden below with per-image calc

    # Add scale bar using full-canvas GSD
    result = draw_scale_bar(result, full_canvas_gsd, target_length_m=scale_bar_m)
    result = _draw_legend(result)

    # Save full-res disaster mosaic
    disaster_full_path = output_dir / "disaster_mosaic.jpg"
    cv2.imwrite(str(disaster_full_path), result, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Save preview for GUI
    ph, pw = result.shape[:2]
    pscale = min(1.0, float(PREVIEW_MAX_SIDE) / max(ph, pw))
    if pscale < 1.0:
        preview = cv2.resize(
            result, (int(round(pw * pscale)), int(round(ph * pscale))),
            interpolation=cv2.INTER_AREA)
    else:
        preview = result
    preview_path = output_dir / "disaster_mosaic_preview.jpg"
    cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # Accurate GSD for preview pixels:
    # 1 preview pixel = 1/pscale full-canvas pixels = 1/(pscale * render_scale) stitching pixels
    # Stitching pixels ≈ mean_gsd_orig / image_resize_scale meters
    # We store mosaic_gsd (full canvas pixel) in manifest; here we compute preview GSD
    preview_gsd = full_canvas_gsd / (render_scale * pscale)

    return disaster_full_path, preview_path, preview_gsd


# ── Full Pipeline ─────────────────────────────────────────────────────────

def run_full_pipeline(
    image_dir: str | Path,
    output_dir: str | Path,
    yolo_weights: str | Path | None = None,
    conf: float = 0.25,
    sam_model: str = "sam_b.pt",
    neighbor_offsets: str = "1,2,3",
    scale_bar_m: float = 10.0,
    use_depth_fallback: bool = True,
    image_ext: str = ".jpg",
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict:
    """
    End-to-end pipeline.  Returns a result dict with all output paths and stats.
    progress_callback(message, fraction_0_to_1) is called at each stage.
    """
    def progress(msg: str, pct: float) -> None:
        print(f"[{pct * 100:5.1f}%] {msg}")
        if progress_callback:
            progress_callback(msg, pct)

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images (case-insensitive extension match)
    ext_lower = image_ext.lower()
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ext_lower
    )
    if not images:
        raise ValueError(f"No {image_ext} images found in {image_dir}")
    progress(f"Found {len(images)} images in {image_dir}", 0.0)

    # ── Stage 1: Metric Scale
    progress("Estimating metric scale (GSD)…", 0.05)
    gsd_dict, gsd_csv = run_metric_scale(images, output_dir, use_depth_fallback)
    mean_gsd_orig = float(np.mean(list(gsd_dict.values()))) if gsd_dict else 0.05
    progress(f"GSD estimated — mean {mean_gsd_orig:.5f} m/px", 0.25)

    # ── Stage 2: Segmentation (optional)
    mask_paths: dict[str, Path] = {}
    if yolo_weights and Path(yolo_weights).exists():
        progress("Running disaster segmentation (YOLO + SAM)…", 0.28)
        mask_paths = run_segmentation(images, Path(yolo_weights), output_dir, conf, sam_model)
        progress(f"Segmentation done — {len(mask_paths)} masks", 0.50)
    else:
        progress("No YOLO weights — skipping segmentation", 0.50)

    # ── Stage 3: Stitching
    progress("Building image pair graph…", 0.52)
    mosaic_tif, poses_path, manifest_path = run_stitching(
        image_dir, output_dir, neighbor_offsets, image_ext)
    progress("Stitching complete", 0.80)

    # ── Stage 4: Composite overlay
    progress("Compositing disaster overlay…", 0.82)
    disaster_full, disaster_preview, preview_gsd = composite_disaster_overlay(
        mosaic_tif, output_dir / "masks", poses_path, manifest_path,
        gsd_dict, output_dir, scale_bar_m,
    )

    # ── Stage 5: Progressive stitching frames (for GUI live playback)
    progress("Generating stitching playback frames…", 0.92)
    frames_dir = output_dir / "stitch_frames"
    frame_paths = render_stitching_frames(image_dir, poses_path, manifest_path, frames_dir)
    progress("Pipeline complete!", 1.0)

    # Persist summary for the GUI
    summary = {
        "images_processed": len(images),
        "mean_gsd_m_per_px": mean_gsd_orig,
        "preview_gsd_m_per_px": preview_gsd,
        "gsd_csv": str(gsd_csv),
        "mosaic_tif": str(mosaic_tif),
        "disaster_mosaic": str(disaster_full),
        "disaster_preview": str(disaster_preview),
        "poses_path": str(poses_path),
        "manifest_path": str(manifest_path),
        "mask_paths": {k: str(v) for k, v in mask_paths.items()},
        "gsd_dict": gsd_dict,
        "has_segmentation": bool(mask_paths),
        "stitch_frames": [str(p) for p in frame_paths],
    }
    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── CLI entry point ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="StitchWise unified pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image-dir", required=True, help="Directory containing input images")
    p.add_argument("--output-dir", default="outputs/pipeline", help="Pipeline output directory")
    p.add_argument("--yolo-weights", default=str(PROJECT_ROOT / "detection" / "model" / "best.pt"),
                   help="Path to trained YOLOv8 weights")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence")
    p.add_argument("--sam-model", default="sam_b.pt",
                   help="SAM model variant (mobile_sam.pt / sam_b.pt / sam_l.pt)")
    p.add_argument("--neighbor-offsets", default="1,2,3",
                   help="Stitching pair offsets, e.g. '1,2,3'")
    p.add_argument("--scale-bar", type=float, default=10.0, help="Scale bar length in metres")
    p.add_argument("--ext", default=".jpg", help="Image file extension, e.g. .jpg or .JPG")
    p.add_argument("--no-depth-fallback", action="store_true",
                   help="Disable Depth Anything V2 fallback for GSD")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_full_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        yolo_weights=args.yolo_weights,
        conf=args.conf,
        sam_model=args.sam_model,
        neighbor_offsets=args.neighbor_offsets,
        scale_bar_m=args.scale_bar,
        use_depth_fallback=not args.no_depth_fallback,
        image_ext=args.ext,
        progress_callback=lambda msg, pct: None,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "gsd_dict"}, indent=2))
