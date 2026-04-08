from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stitchwise.config import load_config
from stitchwise.io_utils import ensure_dir, list_images
from stitchwise.pipeline_pairwise import stitch_pair


def parse_index(path: Path) -> int | None:
    if path.stem.isdigit():
        return int(path.stem)
    return None


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    vals = [v for v in vals if v > 0]
    return sorted(set(vals))


def build_index_map(image_paths: list[Path]) -> dict[int, Path]:
    idx_to_path: dict[int, Path] = {}
    for p in image_paths:
        idx = parse_index(p)
        if idx is not None:
            idx_to_path[idx] = p
    return idx_to_path


def select_indices(idx_to_path: dict[int, Path], start_index: int | None, end_index: int | None) -> list[int]:
    if not idx_to_path:
        return []
    all_indices = sorted(idx_to_path.keys())
    start = start_index if start_index is not None else all_indices[0]
    end = end_index if end_index is not None else all_indices[-1]
    return [i for i in all_indices if start <= i <= end]


def build_candidate_pairs(
    idx_to_path: dict[int, Path],
    indices: list[int],
    neighbor_offsets: list[int],
    max_pairs: int | None,
) -> list[tuple[int, int, int, Path, Path]]:
    pairs: list[tuple[int, int, int, Path, Path]] = []
    valid_set = set(indices)
    max_idx = max(indices) if indices else -1
    for i in indices:
        for off in neighbor_offsets:
            j = i + off
            if j > max_idx:
                continue
            if j in valid_set and i in idx_to_path and j in idx_to_path:
                pairs.append((i, j, off, idx_to_path[i], idx_to_path[j]))
    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


def load_stats_if_exists(stats_path: Path) -> dict:
    if not stats_path.exists():
        return {}
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assess_homography_validity(
    stats: dict,
    max_canvas_scale: float,
    max_canvas_side: int,
) -> dict:
    h_data = stats.get("homography_2_to_1")
    shape1 = stats.get("image1_processed_shape")
    shape2 = stats.get("image2_processed_shape")

    result = {
        "homography_invalid": True,
        "homography_invalid_reason": "missing_homography_or_shape",
        "canvas_width": 0,
        "canvas_height": 0,
        "canvas_scale": 0.0,
    }

    if h_data is None or shape1 is None or shape2 is None:
        return result

    H = np.array(h_data, dtype=np.float64)
    if H.shape != (3, 3):
        result["homography_invalid_reason"] = "invalid_homography_shape"
        return result
    if not np.isfinite(H).all():
        result["homography_invalid_reason"] = "non_finite_homography"
        return result

    h1, w1 = int(shape1[0]), int(shape1[1])
    h2, w2 = int(shape2[0]), int(shape2[1])
    if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0:
        result["homography_invalid_reason"] = "invalid_image_shape"
        return result

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    try:
        warped_corners2 = cv2.perspectiveTransform(corners2, H)
    except cv2.error:
        result["homography_invalid_reason"] = "perspective_transform_failed"
        return result

    if not np.isfinite(warped_corners2).all():
        result["homography_invalid_reason"] = "non_finite_warped_corners"
        return result

    all_corners = np.vstack((corners1, warped_corners2)).reshape(-1, 2)
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(np.float64)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(np.float64)
    canvas_w = int(max(0.0, x_max - x_min))
    canvas_h = int(max(0.0, y_max - y_min))
    canvas_area = float(canvas_w * canvas_h)
    img1_area = float(w1 * h1)
    canvas_scale = canvas_area / max(img1_area, 1.0)

    result["canvas_width"] = canvas_w
    result["canvas_height"] = canvas_h
    result["canvas_scale"] = canvas_scale

    if canvas_w <= 0 or canvas_h <= 0:
        result["homography_invalid_reason"] = "non_positive_canvas"
        return result
    if canvas_w > max_canvas_side or canvas_h > max_canvas_side:
        result["homography_invalid_reason"] = "canvas_side_explosion"
        return result
    if canvas_scale > max_canvas_scale:
        result["homography_invalid_reason"] = "canvas_area_explosion"
        return result

    result["homography_invalid"] = False
    result["homography_invalid_reason"] = ""
    return result


def compute_edge_quality_score(inlier_count: int, inlier_ratio: float, good_match_count: int, invalid_h: bool) -> float:
    # Simple bounded score in [0, 1] using inlier quantity + consistency.
    score = 0.5 * min(1.0, inlier_count / 120.0)
    score += 0.4 * min(1.0, inlier_ratio / 0.35)
    score += 0.1 * min(1.0, good_match_count / 300.0)
    if invalid_h:
        score *= 0.2
    return float(max(0.0, min(1.0, score)))


def decide_edge_acceptance(
    inlier_count: int,
    inlier_ratio: float,
    good_match_count: int,
    invalid_h: bool,
    min_inliers: int,
    min_inlier_ratio: float,
    min_good_matches: int,
) -> tuple[bool, str]:
    if invalid_h:
        return False, "invalid_homography_or_canvas"
    if inlier_count < min_inliers:
        return False, "low_inlier_count"
    if inlier_ratio < min_inlier_ratio:
        return False, "low_inlier_ratio"
    if good_match_count < min_good_matches:
        return False, "low_good_match_count"
    return True, ""


def compute_connected_components(nodes: list[str], accepted_edges: list[dict]) -> list[list[str]]:
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for e in accepted_edges:
        a = e["image1"]
        b = e["image2"]
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    visited: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in visited:
            continue
        comp: list[str] = []
        q: deque[str] = deque([node])
        visited.add(node)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        comp.sort()
        components.append(comp)
    components.sort(key=lambda c: len(c), reverse=True)
    return components


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_main_component_artifacts(
    output_root: Path,
    pair_debug_root: Path,
    accepted_edges: list[dict],
    components: list[list[str]],
) -> None:
    main_nodes = components[0] if components else []
    main_set = set(main_nodes)

    write_json(
        output_root / "main_component_nodes.json",
        {
            "node_count": len(main_nodes),
            "nodes": main_nodes,
        },
    )

    exported_edges: list[dict] = []
    export_logs: list[dict] = []
    for e in accepted_edges:
        image1 = str(e["image1"])
        image2 = str(e["image2"])
        if image1 not in main_set or image2 not in main_set:
            continue

        pair_name = f"{Path(image1).stem}_{Path(image2).stem}"
        stats_path = pair_debug_root / pair_name / "stats.json"
        if not stats_path.exists():
            export_logs.append(
                {"image1": image1, "image2": image2, "pair_name": pair_name, "status": "skip", "reason": "missing_stats"}
            )
            continue

        stats = load_stats_if_exists(stats_path)
        h_2_to_1 = stats.get("homography_2_to_1")
        shape1 = stats.get("image1_processed_shape")
        shape2 = stats.get("image2_processed_shape")
        if h_2_to_1 is None:
            export_logs.append(
                {"image1": image1, "image2": image2, "pair_name": pair_name, "status": "skip", "reason": "missing_h"}
            )
            continue
        if shape1 is None or shape2 is None:
            export_logs.append(
                {
                    "image1": image1,
                    "image2": image2,
                    "pair_name": pair_name,
                    "status": "skip",
                    "reason": "missing_processed_shape",
                }
            )
            continue

        exported_edges.append(
            {
                "image1": image1,
                "image2": image2,
                "neighbor_offset": int(e["neighbor_offset"]),
                "edge_quality_score": float(e["edge_quality_score"]),
                "inlier_count": int(e["inlier_count"]),
                "inlier_ratio": float(e["inlier_ratio"]),
                "good_match_count": int(e["good_match_count"]),
                "homography_2_to_1": h_2_to_1,
                "image1_processed_shape": shape1,
                "image2_processed_shape": shape2,
                "pair_debug_stats": str(stats_path),
            }
        )
        export_logs.append({"image1": image1, "image2": image2, "pair_name": pair_name, "status": "exported", "reason": ""})

    write_json(output_root / "accepted_edges_main_component_h.json", exported_edges)
    write_csv(
        output_root / "edge_export_log.csv",
        export_logs,
        fieldnames=["image1", "image2", "pair_name", "status", "reason"],
    )


def write_markdown_summary(
    path: Path,
    total_nodes: int,
    total_pairs: int,
    accepted_count: int,
    rejected_count: int,
    neighbor_offsets: list[int],
    rejection_counter: dict[str, int],
    components: list[list[str]],
) -> None:
    lines = []
    lines.append("# Pair Graph Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Nodes: {total_nodes}")
    lines.append(f"- Tested pairs: {total_pairs}")
    lines.append(f"- Accepted edges: {accepted_count}")
    lines.append(f"- Rejected edges: {rejected_count}")
    lines.append(f"- Neighbor offsets: {neighbor_offsets}")
    lines.append("")
    lines.append("## Rejection Reasons")
    if rejection_counter:
        for reason, cnt in sorted(rejection_counter.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {reason}: {cnt}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Connected Components")
    lines.append(f"- Component count: {len(components)}")
    top_show = min(10, len(components))
    for i in range(top_show):
        comp = components[i]
        lines.append(f"- Component {i+1}: size={len(comp)}, first_nodes={comp[:8]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local pair connectivity graph from pairwise stitching metrics.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stitching.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "pair_graph"))
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--neighbor-offsets", type=str, default="1,2,3", help="Comma-separated offsets, e.g. 1,2,3")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for tested candidate pairs.")
    parser.add_argument("--ext", type=str, default=".JPG")

    # Simple edge rejection thresholds.
    parser.add_argument("--accept-min-inliers", type=int, default=12)
    parser.add_argument("--accept-min-inlier-ratio", type=float, default=0.10)
    parser.add_argument("--accept-min-good-matches", type=int, default=30)
    parser.add_argument("--max-canvas-scale", type=float, default=25.0)
    parser.add_argument("--max-canvas-side", type=int, default=12000)
    args = parser.parse_args()

    neighbor_offsets = parse_int_list(args.neighbor_offsets)
    if not neighbor_offsets:
        raise ValueError("neighbor-offsets must contain at least one positive integer.")

    base_cfg = load_config(args.config)
    if args.data_dir is not None:
        base_cfg.data_dir = args.data_dir

    output_root = ensure_dir(Path(args.output_dir))
    pair_debug_root = ensure_dir(output_root / "pair_debug")

    data_dir = Path(base_cfg.data_dir)
    image_paths = list_images(data_dir, extensions=(args.ext.lower(),))
    idx_to_path = build_index_map(image_paths)
    indices = select_indices(idx_to_path, args.start_index, args.end_index)
    if not indices:
        raise RuntimeError("No numeric images found in selected range.")

    candidate_pairs = build_candidate_pairs(idx_to_path, indices, neighbor_offsets, args.max_pairs)
    if not candidate_pairs:
        raise RuntimeError("No candidate pairs generated. Check range/offset settings.")

    nodes = [idx_to_path[i].name for i in indices]
    all_edges: list[dict] = []

    print(f"Nodes in range: {len(nodes)}")
    print(f"Candidate pairs to evaluate: {len(candidate_pairs)}")
    for k, (i, j, off, p1, p2) in enumerate(candidate_pairs, start=1):
        print(f"[{k}/{len(candidate_pairs)}] {p1.name} <-> {p2.name} (offset={off})")

        cfg = copy.deepcopy(base_cfg)
        cfg.output_dir = str(pair_debug_root)
        cfg.image1 = p1.name
        cfg.image2 = p2.name
        cfg.min_inliers = min(cfg.min_inliers, args.accept_min_inliers)
        # Graph building needs pair quality metrics; skip heavy warping preview for speed/stability.
        cfg.compute_warp_preview = False

        pair_name = f"{p1.stem}_{p2.stem}"
        pair_dir = pair_debug_root / pair_name
        stats: dict = {}
        try:
            stats = stitch_pair(cfg)
        except Exception:
            stats = load_stats_if_exists(pair_dir / "stats.json")

        keypoints_1 = int(stats.get("keypoints_image1", 0))
        keypoints_2 = int(stats.get("keypoints_image2", 0))
        raw_match_count = int(stats.get("raw_match_count", 0))
        good_match_count = int(stats.get("good_match_count", 0))
        inlier_count = int(stats.get("inlier_count", 0))
        inlier_ratio = float(stats.get("inlier_ratio", 0.0))
        success_or_failure = str(stats.get("success_or_failure", "failure"))
        failure_reason = str(stats.get("failure_reason", "unknown_failure"))

        validity = assess_homography_validity(
            stats=stats,
            max_canvas_scale=args.max_canvas_scale,
            max_canvas_side=args.max_canvas_side,
        )
        invalid_h = bool(validity["homography_invalid"])
        invalid_reason = str(validity["homography_invalid_reason"])

        quality_score = compute_edge_quality_score(
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            good_match_count=good_match_count,
            invalid_h=invalid_h,
        )
        accepted, rejection_reason = decide_edge_acceptance(
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            good_match_count=good_match_count,
            invalid_h=invalid_h,
            min_inliers=args.accept_min_inliers,
            min_inlier_ratio=args.accept_min_inlier_ratio,
            min_good_matches=args.accept_min_good_matches,
        )

        edge_row = {
            "image1": p1.name,
            "image2": p2.name,
            "neighbor_offset": off,
            "keypoints_1": keypoints_1,
            "keypoints_2": keypoints_2,
            "raw_match_count": raw_match_count,
            "good_match_count": good_match_count,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "success_or_failure": success_or_failure,
            "failure_reason": failure_reason if success_or_failure == "failure" else "",
            "homography_invalid": invalid_h,
            "homography_invalid_reason": invalid_reason,
            "canvas_width": int(validity["canvas_width"]),
            "canvas_height": int(validity["canvas_height"]),
            "canvas_scale": float(validity["canvas_scale"]),
            "edge_quality_score": quality_score,
            "accepted_edge": accepted,
            "rejection_reason": rejection_reason if not accepted else "",
        }
        all_edges.append(edge_row)

    accepted_edges = [e for e in all_edges if e["accepted_edge"]]
    rejected_edges = [e for e in all_edges if not e["accepted_edge"]]
    components = compute_connected_components(nodes, accepted_edges)

    rejection_counter: dict[str, int] = defaultdict(int)
    for e in rejected_edges:
        rejection_counter[e["rejection_reason"]] += 1

    fieldnames = [
        "image1",
        "image2",
        "neighbor_offset",
        "keypoints_1",
        "keypoints_2",
        "raw_match_count",
        "good_match_count",
        "inlier_count",
        "inlier_ratio",
        "success_or_failure",
        "failure_reason",
        "homography_invalid",
        "homography_invalid_reason",
        "canvas_width",
        "canvas_height",
        "canvas_scale",
        "edge_quality_score",
        "accepted_edge",
        "rejection_reason",
    ]

    write_csv(output_root / "pair_graph_edges.csv", all_edges, fieldnames)
    write_json(output_root / "pair_graph_edges.json", all_edges)
    write_csv(output_root / "accepted_edges.csv", accepted_edges, fieldnames)
    write_json(output_root / "accepted_edges.json", accepted_edges)
    write_csv(output_root / "rejected_edges.csv", rejected_edges, fieldnames)
    write_json(output_root / "rejected_edges.json", rejected_edges)

    components_payload = {
        "node_count": len(nodes),
        "tested_pair_count": len(all_edges),
        "accepted_edge_count": len(accepted_edges),
        "rejected_edge_count": len(rejected_edges),
        "component_count": len(components),
        "component_sizes": [len(c) for c in components],
        "components": components,
    }
    write_json(output_root / "connected_components.json", components_payload)
    export_main_component_artifacts(
        output_root=output_root,
        pair_debug_root=pair_debug_root,
        accepted_edges=accepted_edges,
        components=components,
    )

    write_markdown_summary(
        path=output_root / "graph_summary.md",
        total_nodes=len(nodes),
        total_pairs=len(all_edges),
        accepted_count=len(accepted_edges),
        rejected_count=len(rejected_edges),
        neighbor_offsets=neighbor_offsets,
        rejection_counter=dict(rejection_counter),
        components=components,
    )

    print("Pair graph build finished.")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
