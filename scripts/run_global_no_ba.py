from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end global stitching pipeline without BA.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "stitching.yaml"))
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
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--neighbor-offsets", type=str, default="1,2,3")
    parser.add_argument("--run-build-graph", action="store_true", help="Rebuild pair graph before global stages.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for pair graph building.")
    args = parser.parse_args()

    python = sys.executable
    build_script = str(PROJECT_ROOT / "scripts" / "build_pair_graph.py")
    solve_script = str(PROJECT_ROOT / "scripts" / "solve_global_no_ba.py")
    render_script = str(PROJECT_ROOT / "scripts" / "render_global_no_ba.py")
    validate_script = str(PROJECT_ROOT / "scripts" / "validate_global_no_ba.py")

    if args.run_build_graph:
        cmd = [
            python,
            build_script,
            "--config",
            args.config,
            "--output-dir",
            args.pair_graph_dir,
            "--neighbor-offsets",
            args.neighbor_offsets,
        ]
        if args.data_dir is not None:
            cmd += ["--data-dir", args.data_dir]
        if args.max_pairs is not None:
            cmd += ["--max-pairs", str(args.max_pairs)]
        run_cmd(cmd)
    else:
        print("Skipping graph build step (use --run-build-graph to rebuild).")

    cmd = [
        python,
        solve_script,
        "--pair-graph-dir",
        args.pair_graph_dir,
        "--output-dir",
        args.global_dir,
    ]
    run_cmd(cmd)

    cmd = [
        python,
        render_script,
        "--config",
        args.config,
        "--poses-json",
        str(Path(args.global_dir) / "global_poses.json"),
        "--output-dir",
        args.global_dir,
    ]
    if args.data_dir is not None:
        cmd += ["--data-dir", args.data_dir]
    run_cmd(cmd)

    cmd = [
        python,
        validate_script,
        "--pair-graph-dir",
        args.pair_graph_dir,
        "--global-dir",
        args.global_dir,
    ]
    run_cmd(cmd)

    print("Global no-BA pipeline finished.")
    print(f"Results: {args.global_dir}")


if __name__ == "__main__":
    main()
