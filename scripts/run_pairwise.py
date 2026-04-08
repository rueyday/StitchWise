from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stitchwise.config import load_config
from stitchwise.pipeline_pairwise import stitch_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pairwise aerial image stitching.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stitching.yaml"))
    parser.add_argument("--image1", type=str, default=None, help="Path or filename for first image.")
    parser.add_argument("--image2", type=str, default=None, help="Path or filename for second image.")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional override for dataset directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for output directory.")
    parser.add_argument("--resize-max-dim", type=int, default=None, help="Optional resize override.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.image1 is not None:
        cfg.image1 = args.image1
    if args.image2 is not None:
        cfg.image2 = args.image2
    if args.resize_max_dim is not None:
        cfg.resize_max_dim = args.resize_max_dim

    stats = stitch_pair(cfg)
    print(json.dumps(stats, indent=2))
    print("Pairwise stitching finished.")
    print(f"Output directory: {stats['output_dir']}")


if __name__ == "__main__":
    main()
