from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def to_h(mat_like) -> np.ndarray:
    h = np.array(mat_like, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError("homography must be 3x3")
    if not np.isfinite(h).all():
        raise ValueError("homography has non-finite values")
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve global image transforms from accepted pairwise edges (no BA).")
    parser.add_argument(
        "--pair-graph-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "pair_graph"),
        help="Directory containing accepted_edges_main_component_h.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "global_no_ba"),
        help="Directory to save global transforms and logs.",
    )
    parser.add_argument("--anchor", type=str, default=None, help="Optional anchor image filename, e.g. 100.JPG")
    args = parser.parse_args()

    pair_graph_dir = Path(args.pair_graph_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    edges_path = pair_graph_dir / "accepted_edges_main_component_h.json"
    nodes_path = pair_graph_dir / "main_component_nodes.json"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing file: {edges_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing file: {nodes_path}")

    edges_raw = load_json(edges_path)
    nodes_payload = load_json(nodes_path)
    nodes = list(nodes_payload.get("nodes", []))
    if not nodes:
        raise RuntimeError("No nodes found in main_component_nodes.json")

    valid_edges: list[dict] = []
    dropped_edges: list[dict] = []
    shape_map: dict[str, list[int]] = {}
    for e in edges_raw:
        image1 = str(e["image1"])
        image2 = str(e["image2"])
        try:
            h_2_to_1 = to_h(e["homography_2_to_1"])
            quality = float(e.get("edge_quality_score", 0.0))
            inlier_count = int(e.get("inlier_count", 0))
            inlier_ratio = float(e.get("inlier_ratio", 0.0))
            off = int(e.get("neighbor_offset", 0))
            shape1 = e.get("image1_processed_shape")
            shape2 = e.get("image2_processed_shape")
            if shape1 is not None:
                shape_map[image1] = shape1
            if shape2 is not None:
                shape_map[image2] = shape2
            valid_edges.append(
                {
                    "image1": image1,
                    "image2": image2,
                    "neighbor_offset": off,
                    "edge_quality_score": quality,
                    "inlier_count": inlier_count,
                    "inlier_ratio": inlier_ratio,
                    "homography_2_to_1": h_2_to_1,
                }
            )
        except Exception as exc:
            dropped_edges.append(
                {
                    "image1": image1,
                    "image2": image2,
                    "reason": f"invalid_homography: {exc}",
                }
            )

    if not valid_edges:
        raise RuntimeError("No valid edges with usable homography.")

    weighted_degree: dict[str, float] = defaultdict(float)
    for e in valid_edges:
        w = max(0.0, float(e["edge_quality_score"]))
        weighted_degree[e["image1"]] += w
        weighted_degree[e["image2"]] += w

    anchor = args.anchor
    if anchor is None or anchor not in nodes:
        anchor = max(nodes, key=lambda n: weighted_degree.get(n, 0.0))
    print(f"Anchor node: {anchor}")

    # Maximum spanning tree (Kruskal on descending weights).
    uf = UnionFind(nodes)
    sorted_edges = sorted(valid_edges, key=lambda x: x["edge_quality_score"], reverse=True)
    tree_edges: list[dict] = []
    non_tree_edges: list[dict] = []
    for e in sorted_edges:
        a = e["image1"]
        b = e["image2"]
        if uf.union(a, b):
            tree_edges.append(e)
        else:
            non_tree_edges.append(e)

    # Build adjacency with transform metadata.
    adj: dict[str, list[dict]] = defaultdict(list)
    for e in tree_edges:
        a = e["image1"]
        b = e["image2"]
        adj[a].append(e)
        adj[b].append(e)

    poses: dict[str, np.ndarray] = {anchor: np.eye(3, dtype=np.float64)}
    q: deque[str] = deque([anchor])
    dropped_nodes: list[dict] = []
    while q:
        cur = q.popleft()
        T_cur = poses[cur]
        for e in adj[cur]:
            a = e["image1"]
            b = e["image2"]
            H_2_to_1 = e["homography_2_to_1"]
            if cur == a:
                nxt = b
                T_nxt = T_cur @ H_2_to_1  # image2 -> image1 -> anchor
            else:
                nxt = a
                try:
                    H_1_to_2 = np.linalg.inv(H_2_to_1)
                except np.linalg.LinAlgError:
                    dropped_nodes.append({"image": nxt, "reason": f"non_invertible_tree_edge {a}<->{b}"})
                    continue
                T_nxt = T_cur @ H_1_to_2  # image1 -> image2 -> anchor

            if not np.isfinite(T_nxt).all():
                dropped_nodes.append({"image": nxt, "reason": f"non_finite_pose_from_edge {a}<->{b}"})
                continue
            if abs(T_nxt[2, 2]) > 1e-12:
                T_nxt = T_nxt / T_nxt[2, 2]
            if nxt not in poses:
                poses[nxt] = T_nxt
                q.append(nxt)

    unreachable = [n for n in nodes if n not in poses]
    for n in unreachable:
        dropped_nodes.append({"image": n, "reason": "unreachable_from_anchor"})

    tree_rows = []
    for e in tree_edges:
        tree_rows.append(
            {
                "image1": e["image1"],
                "image2": e["image2"],
                "neighbor_offset": e["neighbor_offset"],
                "edge_quality_score": e["edge_quality_score"],
                "inlier_count": e["inlier_count"],
                "inlier_ratio": e["inlier_ratio"],
            }
        )

    for e in non_tree_edges:
        dropped_edges.append({"image1": e["image1"], "image2": e["image2"], "reason": "non_tree_edge"})

    pose_rows = []
    for image_name in sorted(poses.keys()):
        pose_rows.append(
            {
                "image": image_name,
                "H_to_anchor": poses[image_name].tolist(),
                "image_processed_shape": shape_map.get(image_name, None),
            }
        )

    global_payload = {
        "anchor": anchor,
        "node_count_total": len(nodes),
        "pose_count": len(pose_rows),
        "tree_edge_count": len(tree_rows),
        "nodes": pose_rows,
    }

    save_json(output_dir / "global_poses.json", global_payload)
    write_csv(
        output_dir / "tree_edges.csv",
        tree_rows,
        fieldnames=["image1", "image2", "neighbor_offset", "edge_quality_score", "inlier_count", "inlier_ratio"],
    )
    write_csv(output_dir / "dropped_edges.csv", dropped_edges, fieldnames=["image1", "image2", "reason"])
    write_csv(output_dir / "dropped_nodes.csv", dropped_nodes, fieldnames=["image", "reason"])

    print(f"Global pose solving finished. Poses: {len(pose_rows)}/{len(nodes)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
