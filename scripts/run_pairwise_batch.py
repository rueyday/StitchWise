from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path


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


def build_neighbor_pairs(
    image_paths: list[Path],
    start_index: int | None,
    end_index: int | None,
    neighbor_offset: int,
    step_size: int,
    max_pairs: int | None,
) -> list[tuple[Path, Path]]:
    indexed = [(parse_index(p), p) for p in image_paths]
    indexed = [(idx, p) for idx, p in indexed if idx is not None]
    indexed.sort(key=lambda x: x[0])

    if not indexed:
        return []

    min_idx = indexed[0][0]
    max_idx = indexed[-1][0]
    start = start_index if start_index is not None else min_idx
    end = end_index if end_index is not None else max_idx

    idx_to_path = {idx: p for idx, p in indexed}
    candidates: list[tuple[Path, Path]] = []
    for idx in range(start, end + 1):
        j = idx + neighbor_offset
        if j > end:
            continue
        if idx in idx_to_path and j in idx_to_path:
            candidates.append((idx_to_path[idx], idx_to_path[j]))

    if step_size > 1:
        candidates = candidates[::step_size]
    if max_pairs is not None and max_pairs > 0:
        candidates = candidates[:max_pairs]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch pairwise stitching on neighboring image pairs.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stitching.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "pairwise_batch"),
        help="Batch output root directory.",
    )
    parser.add_argument("--start-index", type=int, default=None, help="Start numeric index, e.g. 1")
    parser.add_argument("--end-index", type=int, default=None, help="End numeric index, e.g. 236")
    parser.add_argument("--max-pairs", type=int, default=None, help="Maximum number of pairs to run.")
    parser.add_argument("--step-size", type=int, default=1, help="Take every Nth neighboring pair.")
    parser.add_argument(
        "--neighbor-offset",
        type=int,
        default=1,
        help="Neighbor offset k for pair selection: (i, i+k). Default is 1.",
    )
    parser.add_argument("--ext", type=str, default=".JPG")
    args = parser.parse_args()

    if args.step_size < 1:
        raise ValueError("step-size must be >= 1")
    if args.neighbor_offset < 1:
        raise ValueError("neighbor-offset must be >= 1")

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg.data_dir)
    batch_output_root = ensure_dir(Path(args.output_dir))
    batch_output_dir = ensure_dir(batch_output_root / f"offset_{args.neighbor_offset}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    image_paths = list_images(data_dir, extensions=(args.ext.lower(),))
    pairs = build_neighbor_pairs(
        image_paths=image_paths,
        start_index=args.start_index,
        end_index=args.end_index,
        neighbor_offset=args.neighbor_offset,
        step_size=args.step_size,
        max_pairs=args.max_pairs,
    )
    if not pairs:
        print("No neighboring pairs selected. Please check index range and extension.")
        return

    summary_rows: list[dict] = []
    for i, (img1, img2) in enumerate(pairs, start=1):
        pair_name = f"{img1.stem}_{img2.stem}"
        pair_dir = ensure_dir(batch_output_dir / pair_name)
        print(f"[{i}/{len(pairs)}] Running pair {img1.name} <-> {img2.name}")

        run_cfg = copy.deepcopy(cfg)
        run_cfg.data_dir = str(data_dir)
        run_cfg.output_dir = str(batch_output_dir)
        run_cfg.image1 = img1.name
        run_cfg.image2 = img2.name

        try:
            stats = stitch_pair(run_cfg)
            row = {
                "neighbor_offset": args.neighbor_offset,
                "image1": img1.name,
                "image2": img2.name,
                "keypoints_1": stats.get("keypoints_image1", 0),
                "keypoints_2": stats.get("keypoints_image2", 0),
                "raw_match_count": stats.get("raw_match_count", 0),
                "good_match_count": stats.get("good_match_count", 0),
                "inlier_count": stats.get("inlier_count", 0),
                "inlier_ratio": stats.get("inlier_ratio", 0.0),
                "success_or_failure": "success",
                "failure_reason": "",
            }
            summary_rows.append(row)
        except Exception as exc:
            reason = str(exc)
            stats_path = pair_dir / "stats.json"
            if stats_path.exists():
                with stats_path.open("r", encoding="utf-8") as f:
                    failure_stats = json.load(f)
            else:
                failure_stats = {
                    "keypoints_image1": 0,
                    "keypoints_image2": 0,
                    "raw_match_count": 0,
                    "good_match_count": 0,
                    "inlier_count": 0,
                    "inlier_ratio": 0.0,
                    "failure_reason": reason,
                }
            summary_rows.append(
                {
                    "neighbor_offset": args.neighbor_offset,
                    "image1": img1.name,
                    "image2": img2.name,
                    "keypoints_1": failure_stats.get("keypoints_image1", 0),
                    "keypoints_2": failure_stats.get("keypoints_image2", 0),
                    "raw_match_count": failure_stats.get("raw_match_count", 0),
                    "good_match_count": failure_stats.get("good_match_count", 0),
                    "inlier_count": failure_stats.get("inlier_count", 0),
                    "inlier_ratio": failure_stats.get("inlier_ratio", 0.0),
                    "success_or_failure": "failure",
                    "failure_reason": failure_stats.get("failure_reason", reason),
                }
            )

    summary_csv = batch_output_dir / "summary.csv"
    summary_json = batch_output_dir / "summary.json"
    fieldnames = [
        "neighbor_offset",
        "image1",
        "image2",
        "keypoints_1",
        "keypoints_2",
        "raw_match_count",
        "good_match_count",
        "inlier_count",
        "inlier_ratio",
        "success_or_failure",
        "failure_reason",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    success_count = sum(1 for r in summary_rows if r["success_or_failure"] == "success")
    print(f"Batch finished: {success_count}/{len(summary_rows)} succeeded.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
