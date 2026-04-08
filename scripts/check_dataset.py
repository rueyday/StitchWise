from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stitchwise.config import load_config
from stitchwise.io_utils import list_images


def parse_index(filename: str) -> int | None:
    stem = Path(filename).stem
    if stem.isdigit():
        return int(stem)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset filenames for stitching.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stitching.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--ext", type=str, default=".JPG", help="Image extension filter, e.g. .JPG")
    parser.add_argument("--show", type=int, default=20, help="How many neighbor pairs to print.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg.data_dir)
    ext = args.ext.lower()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    images = list_images(data_dir, extensions=(ext,))
    if not images:
        print(f"No images found in {data_dir} with extension {ext}")
        return

    filenames = [p.name for p in images]
    indexed = sorted(i for i in (parse_index(name) for name in filenames) if i is not None)
    indexed_set = set(indexed)

    print(f"Data directory: {data_dir}")
    print(f"Extension filter: {ext}")
    print(f"Image count: {len(images)}")
    print(f"First file: {filenames[0]}")
    print(f"Last file: {filenames[-1]}")

    if indexed:
        min_idx = indexed[0]
        max_idx = indexed[-1]
        missing = [i for i in range(min_idx, max_idx + 1) if i not in indexed_set]
        neighbors = [(i, i + 1) for i in indexed if (i + 1) in indexed_set]
        width = max(len(str(max_idx)), 3)

        print(f"Numeric index range: {min_idx}..{max_idx}")
        print(f"Missing indices: {missing if missing else 'None'}")
        print(f"Neighbor pair count: {len(neighbors)}")
        print("Suggested neighboring pairs:")
        for a, b in neighbors[: args.show]:
            print(f"  {a:0{width}d}.JPG <-> {b:0{width}d}.JPG")
    else:
        print("No purely numeric filenames detected; cannot compute missing indices.")


if __name__ == "__main__":
    main()
