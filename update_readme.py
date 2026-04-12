"""
Update README.md with the latest pipeline run results.

Run after pipeline.py:
    python update_readme.py [--results-dir results/]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


README_PATH = Path("README.md")
RESULTS_SECTION_START = "<!-- RESULTS:START -->"
RESULTS_SECTION_END = "<!-- RESULTS:END -->"


def load_summary(results_dir: Path) -> dict | None:
    json_path = results_dir / "summary.json"
    if not json_path.exists():
        print(f"[update_readme] No summary.json found in {results_dir}")
        return None
    with open(json_path) as f:
        return json.load(f)


def load_sample_rows(results_dir: Path, n: int = 5) -> list[dict] | None:
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        return None
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return rows[:n]


def build_results_block(summary: dict, sample_rows: list[dict] | None) -> str:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    methods = summary.get("methods_used", {})
    method_str = ", ".join(f"`{k}` ({v} frames)" for k, v in methods.items())

    lines = [
        RESULTS_SECTION_START,
        "",
        f"## Results — Last Run: {run_time}",
        "",
        "### Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Frames processed | {summary['n_frames']} |",
        f"| Failed frames | {summary['n_failed']} |",
        f"| Mean GSD | {summary['gsd_mean_cm_per_px']} cm/px |",
        f"| Min GSD | {summary['gsd_min_cm_per_px']} cm/px |",
        f"| Max GSD | {summary['gsd_max_cm_per_px']} cm/px |",
        f"| Processing time | {summary['elapsed_s']} s |",
        f"| Scale methods used | {method_str} |",
        "",
    ]

    if sample_rows:
        lines += [
            "### Sample Frame Results (first 5)",
            "",
            "| Frame | GSD (cm/px) | Altitude (m) | Method |",
            "|-------|-------------|--------------|--------|",
        ]
        for r in sample_rows:
            lines.append(
                f"| `{r['frame']}` | {r['gsd_cm_per_px']} | {r['altitude_m']} | `{r['method']}` |"
            )
        lines.append("")

    lines.append("### GSD Timeline")
    lines.append("")
    lines.append("![GSD Timeline](results/gsd_timeline.png)")
    lines.append("")
    lines.append(RESULTS_SECTION_END)

    return "\n".join(lines)


def update_readme(results_dir: Path) -> None:
    summary = load_summary(results_dir)
    if summary is None:
        return

    sample_rows = load_sample_rows(results_dir)
    results_block = build_results_block(summary, sample_rows)

    content = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

    if RESULTS_SECTION_START in content and RESULTS_SECTION_END in content:
        # Replace existing results section
        before = content[: content.index(RESULTS_SECTION_START)]
        after = content[content.index(RESULTS_SECTION_END) + len(RESULTS_SECTION_END):]
        new_content = before + results_block + after
    else:
        # Append results section
        new_content = content.rstrip() + "\n\n" + results_block + "\n"

    README_PATH.write_text(new_content, encoding="utf-8")
    print(f"[update_readme] README.md updated with results from {results_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Update README with pipeline results.")
    p.add_argument("--results-dir", default="results", help="Results directory")
    args = p.parse_args()
    update_readme(Path(args.results_dir))


if __name__ == "__main__":
    main()
