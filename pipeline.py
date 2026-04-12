"""
StitchWise — Metric Scale Pipeline for Drone Nadir Imagery
===========================================================

Usage:
    python pipeline.py [OPTIONS]

Options:
    --data-dir PATH         Local dataset directory (downloads if missing)
    --output-dir PATH       Output directory for annotated images [default: results/]
    --altitude FLOAT        Force AGL altitude in meters (overrides EXIF)
    --no-depth-fallback     Disable Depth Anything V2 fallback (requires full EXIF)
    --max-images INT        Limit number of images to process
    --scale-bar FLOAT       Scale bar length in meters to draw [default: 10.0]
    --skip-vis              Skip visualization (only output CSV)

Outputs:
    results/
      <frame>_metric.jpg   Annotated image with scale bar
      <frame>_depth.jpg    Depth map (when depth fallback used)
      results.csv          Per-frame GSD table
      gsd_timeline.png     GSD vs. frame plot
      summary.txt          Human-readable run summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
import time
from pathlib import Path

from src.dataset import load_or_download
from src.metric_scale import ScaleMethod, estimate
from src.visualize import plot_gsd_timeline, save_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="StitchWise: metric scale estimation for drone nadir image sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    p.add_argument("--data-dir", default="data/aerial234", help="Dataset cache directory")
    p.add_argument("--output-dir", default="results", help="Output directory")
    p.add_argument("--altitude", type=float, default=None, help="Force AGL altitude in meters")
    p.add_argument("--no-depth-fallback", action="store_true", help="Disable depth model fallback")
    p.add_argument("--max-images", type=int, default=None, help="Limit images processed")
    p.add_argument("--scale-bar", type=float, default=10.0, help="Scale bar length in meters")
    p.add_argument("--skip-vis", action="store_true", help="Skip image annotation")
    p.add_argument("--hf-token", default=None, help="HuggingFace access token (or set HF_TOKEN env var)")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  StitchWise — Metric Scale Pipeline")
    print(f"{'='*60}\n")

    # Load dataset
    images = load_or_download(args.data_dir, token=args.hf_token)
    if args.max_images:
        images = images[: args.max_images]

    print(f"[pipeline] Processing {len(images)} images from {args.data_dir}\n")

    # Per-frame results
    rows: list[dict] = []
    frame_names: list[str] = []
    gsd_values: list[float] = []
    failed: list[str] = []
    method_counts: dict[str, int] = {}

    t0 = time.time()

    for idx, img_path in enumerate(images):
        print(f"[{idx+1:3d}/{len(images)}] {img_path.name}", end=" ... ", flush=True)
        try:
            result = estimate(
                img_path,
                force_altitude_m=args.altitude,
                use_depth_fallback=not args.no_depth_fallback,
            )
            gsd = result.gsd_m_per_px
            method = result.method.value
            print(f"GSD={gsd*100:.2f} cm/px  alt={result.altitude_m:.1f}m  [{method}]")

            row = {
                "frame": img_path.name,
                "gsd_m_per_px": f"{gsd:.6f}",
                "gsd_cm_per_px": f"{gsd*100:.4f}",
                "altitude_m": f"{result.altitude_m:.2f}",
                "focal_px": f"{result.focal_length_px:.1f}" if result.focal_length_px else "",
                "method": method,
                "source_altitude": result.camera_params.source_altitude,
                "source_focal": result.camera_params.source_focal,
                "gps_lat": f"{result.camera_params.gps_lat:.6f}" if result.camera_params.gps_lat else "",
                "gps_lon": f"{result.camera_params.gps_lon:.6f}" if result.camera_params.gps_lon else "",
            }
            rows.append(row)
            frame_names.append(img_path.name)
            gsd_values.append(gsd)
            method_counts[method] = method_counts.get(method, 0) + 1

            if not args.skip_vis:
                save_result(
                    img_path,
                    gsd,
                    method,
                    output_dir,
                    depth_map=result.depth_map,
                    scale_bar_m=args.scale_bar,
                )

        except Exception as exc:
            print(f"FAILED: {exc}")
            failed.append(img_path.name)

    elapsed = time.time() - t0

    # Write CSV
    if rows:
        csv_path = output_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[pipeline] CSV saved -> {csv_path}")

    # GSD timeline plot
    if gsd_values and not args.skip_vis:
        plot_gsd_timeline(frame_names, gsd_values, output_dir / "gsd_timeline.png")

    # Summary stats
    if gsd_values:
        arr = [g * 100 for g in gsd_values]  # cm/px
        stats = {
            "n_frames": len(gsd_values),
            "n_failed": len(failed),
            "gsd_mean_cm_per_px": f"{sum(arr)/len(arr):.3f}",
            "gsd_min_cm_per_px": f"{min(arr):.3f}",
            "gsd_max_cm_per_px": f"{max(arr):.3f}",
            "elapsed_s": f"{elapsed:.1f}",
            "methods_used": method_counts,
        }

        summary_lines = [
            "=" * 60,
            "  StitchWise — Run Summary",
            "=" * 60,
            f"  Frames processed : {stats['n_frames']}",
            f"  Failed           : {stats['n_failed']}",
            f"  GSD mean         : {stats['gsd_mean_cm_per_px']} cm/px",
            f"  GSD range        : {stats['gsd_min_cm_per_px']} – {stats['gsd_max_cm_per_px']} cm/px",
            f"  Elapsed time     : {stats['elapsed_s']} s",
            "  Methods used     :",
        ] + [f"    {k}: {v}" for k, v in method_counts.items()]
        if failed:
            summary_lines += ["  Failed frames    :"] + [f"    {f}" for f in failed]
        summary_lines.append("=" * 60)

        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)

        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")
        print(f"[pipeline] Summary saved -> {summary_path}")

        # Also dump JSON for README update script
        json_path = output_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
