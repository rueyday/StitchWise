from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stitchwise.config import load_config
from stitchwise.io_utils import ensure_dir
from stitchwise.pipeline_pairwise import stitch_pair


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small parameter sweep for one pairwise stitching case.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stitching.yaml"))
    parser.add_argument("--image1", type=str, default="003.JPG")
    parser.add_argument("--image2", type=str, default="004.JPG")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "pairwise_sweep"),
        help="Output root directory for sweep artifacts.",
    )
    parser.add_argument("--ratio-tests", type=str, default="0.70,0.75,0.80,0.85")
    parser.add_argument("--ransac-thresholds", type=str, default="3.0,5.0,8.0")
    parser.add_argument("--min-inliers", type=str, default="8,12,20")
    args = parser.parse_args()

    ratio_tests = parse_float_list(args.ratio_tests)
    ransac_thresholds = parse_float_list(args.ransac_thresholds)
    min_inliers_list = parse_int_list(args.min_inliers)

    base_cfg = load_config(args.config)
    if args.data_dir is not None:
        base_cfg.data_dir = args.data_dir
    base_cfg.image1 = args.image1
    base_cfg.image2 = args.image2

    sweep_root = ensure_dir(Path(args.output_dir))
    pair_name = f"{Path(args.image1).stem}_{Path(args.image2).stem}"
    rows: list[dict] = []

    combos = list(itertools.product(ratio_tests, ransac_thresholds, min_inliers_list))
    total = len(combos)
    for idx, (ratio_test, ransac_thresh, min_inliers) in enumerate(combos, start=1):
        combo_dir = ensure_dir(
            sweep_root / f"ratio_{ratio_test:.2f}_ransac_{ransac_thresh:.1f}_min_{min_inliers}"
        )
        print(
            f"[{idx}/{total}] ratio_test={ratio_test:.2f}, "
            f"ransac_thresh={ransac_thresh:.1f}, min_inliers={min_inliers}"
        )

        cfg = copy.deepcopy(base_cfg)
        cfg.output_dir = str(combo_dir)
        cfg.ratio_test = ratio_test
        cfg.ransac_reproj_threshold = ransac_thresh
        cfg.min_inliers = min_inliers

        try:
            stats = stitch_pair(cfg)
            failure_reason = ""
            success_or_failure = "success"
        except Exception as exc:
            stats_path = combo_dir / pair_name / "stats.json"
            if stats_path.exists():
                with stats_path.open("r", encoding="utf-8") as f:
                    stats = json.load(f)
            else:
                stats = {}
            failure_reason = stats.get("failure_reason", str(exc))
            success_or_failure = "failure"

        row = {
            "ratio_test": ratio_test,
            "ransac_thresh": ransac_thresh,
            "min_inliers": min_inliers,
            "keypoints_1": stats.get("keypoints_image1", 0),
            "keypoints_2": stats.get("keypoints_image2", 0),
            "raw_match_count": stats.get("raw_match_count", 0),
            "good_match_count": stats.get("good_match_count", 0),
            "inlier_count": stats.get("inlier_count", 0),
            "inlier_ratio": stats.get("inlier_ratio", 0.0),
            "success_or_failure": success_or_failure,
            "failure_reason": failure_reason,
        }
        rows.append(row)

    summary_csv = sweep_root / "summary.csv"
    summary_json = sweep_root / "summary.json"
    fields = [
        "ratio_test",
        "ransac_thresh",
        "min_inliers",
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    success_count = sum(1 for r in rows if r["success_or_failure"] == "success")
    print(f"Sweep finished: {success_count}/{len(rows)} succeeded.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
