from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def compute_edge_residual(edge: dict, pose_map: dict[str, np.ndarray]) -> tuple[float, float]:
    image1 = str(edge["image1"])
    image2 = str(edge["image2"])
    if image1 not in pose_map or image2 not in pose_map:
        raise ValueError("missing_pose")

    H_2_to_1 = np.array(edge["homography_2_to_1"], dtype=np.float64)
    if H_2_to_1.shape != (3, 3):
        raise ValueError("bad_h_shape")
    if not np.isfinite(H_2_to_1).all():
        raise ValueError("bad_h_values")

    T1 = pose_map[image1]
    T2 = pose_map[image2]
    shape2 = edge.get("image2_processed_shape")
    if shape2 is None:
        raise ValueError("missing_shape2")
    h2, w2 = int(shape2[0]), int(shape2[1])
    if h2 <= 0 or w2 <= 0:
        raise ValueError("invalid_shape2")

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    via_edge = cv2.perspectiveTransform(corners2, T1 @ H_2_to_1).reshape(-1, 2)
    via_pose = cv2.perspectiveTransform(corners2, T2).reshape(-1, 2)
    d = np.linalg.norm(via_edge - via_pose, axis=1)
    return float(d.mean()), float(d.max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate global no-BA mosaic outputs.")
    parser.add_argument(
        "--pair-graph-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "pair_graph"),
    )
    parser.add_argument(
        "--global-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "global_no_ba"),
    )
    parser.add_argument("--min-placed-ratio", type=float, default=0.90)
    parser.add_argument("--max-loop-median", type=float, default=25.0)
    parser.add_argument("--max-loop-p95", type=float, default=80.0)
    parser.add_argument("--max-skipped-ratio", type=float, default=0.10)
    args = parser.parse_args()

    pair_graph_dir = Path(args.pair_graph_dir)
    global_dir = Path(args.global_dir)

    edges_path = pair_graph_dir / "accepted_edges_main_component_h.json"
    poses_path = global_dir / "global_poses.json"
    manifest_path = global_dir / "render_manifest.json"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing file: {edges_path}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing file: {poses_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing file: {manifest_path}")

    edges = load_json(edges_path)
    poses_payload = load_json(poses_path)
    manifest = load_json(manifest_path)

    pose_map: dict[str, np.ndarray] = {}
    for n in poses_payload.get("nodes", []):
        image = str(n["image"])
        H = np.array(n["H_to_anchor"], dtype=np.float64)
        if H.shape == (3, 3) and np.isfinite(H).all():
            if abs(H[2, 2]) > 1e-12:
                H = H / H[2, 2]
            pose_map[image] = H

    residual_rows: list[dict] = []
    for e in edges:
        image1 = str(e["image1"])
        image2 = str(e["image2"])
        try:
            mean_err, max_err = compute_edge_residual(e, pose_map)
            residual_rows.append(
                {
                    "image1": image1,
                    "image2": image2,
                    "residual_mean_px": mean_err,
                    "residual_max_px": max_err,
                    "status": "ok",
                    "reason": "",
                }
            )
        except Exception as exc:
            residual_rows.append(
                {
                    "image1": image1,
                    "image2": image2,
                    "residual_mean_px": 0.0,
                    "residual_max_px": 0.0,
                    "status": "skip",
                    "reason": str(exc),
                }
            )

    loop_csv_path = global_dir / "loop_residuals.csv"
    write_csv(
        loop_csv_path,
        residual_rows,
        fieldnames=["image1", "image2", "residual_mean_px", "residual_max_px", "status", "reason"],
    )

    ok_residuals = [float(r["residual_mean_px"]) for r in residual_rows if r["status"] == "ok"]
    loop_median = safe_percentile(ok_residuals, 50)
    loop_p95 = safe_percentile(ok_residuals, 95)
    loop_mean = float(np.mean(ok_residuals)) if ok_residuals else 0.0

    node_total = int(poses_payload.get("node_count_total", 0))
    pose_count = int(poses_payload.get("pose_count", 0))
    placed = int(manifest.get("placed_nodes", 0))
    requested = int(manifest.get("requested_nodes", 0))
    skipped = int(manifest.get("skipped_nodes", 0))

    placed_ratio = float(placed / max(requested, 1))
    skipped_ratio = float(skipped / max(requested, 1))
    pose_ratio = float(pose_count / max(node_total, 1))

    checks = {
        "placed_ratio_ok": placed_ratio >= args.min_placed_ratio,
        "loop_median_ok": loop_median <= args.max_loop_median,
        "loop_p95_ok": loop_p95 <= args.max_loop_p95,
        "skipped_ratio_ok": skipped_ratio <= args.max_skipped_ratio,
    }
    overall_ok = all(checks.values())

    report = {
        "overall_ok": overall_ok,
        "checks": checks,
        "node_count_total": node_total,
        "pose_count": pose_count,
        "pose_ratio": pose_ratio,
        "requested_nodes": requested,
        "placed_nodes": placed,
        "placed_ratio": placed_ratio,
        "skipped_nodes": skipped,
        "skipped_ratio": skipped_ratio,
        "loop_residual_edge_count_ok": len(ok_residuals),
        "loop_residual_mean_px": loop_mean,
        "loop_residual_median_px": loop_median,
        "loop_residual_p95_px": loop_p95,
        "files": {
            "loop_residuals_csv": str(loop_csv_path),
            "mosaic_preview": str(global_dir / "mosaic_no_ba_preview.jpg"),
            "mosaic_tif": str(global_dir / "mosaic_no_ba.tif"),
        },
    }

    report_json_path = global_dir / "validation_report.json"
    report_md_path = global_dir / "validation_report.md"
    save_json(report_json_path, report)

    lines = []
    lines.append("# Global No-BA Validation Report")
    lines.append("")
    lines.append(f"- overall_ok: {overall_ok}")
    lines.append(f"- pose_count / node_count_total: {pose_count}/{node_total} ({pose_ratio:.3f})")
    lines.append(f"- placed_nodes / requested_nodes: {placed}/{requested} ({placed_ratio:.3f})")
    lines.append(f"- skipped_nodes: {skipped} ({skipped_ratio:.3f})")
    lines.append(f"- loop_residual_mean_px: {loop_mean:.3f}")
    lines.append(f"- loop_residual_median_px: {loop_median:.3f}")
    lines.append(f"- loop_residual_p95_px: {loop_p95:.3f}")
    lines.append("")
    lines.append("## Checks")
    for k, v in checks.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- min_placed_ratio: {args.min_placed_ratio}")
    lines.append(f"- max_loop_median: {args.max_loop_median}")
    lines.append(f"- max_loop_p95: {args.max_loop_p95}")
    lines.append(f"- max_skipped_ratio: {args.max_skipped_ratio}")
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Validation finished.")
    print(f"overall_ok={overall_ok}")
    print(f"Report JSON: {report_json_path}")
    print(f"Report MD: {report_md_path}")


if __name__ == "__main__":
    main()
