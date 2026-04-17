"""
Usage:
    python evaluate.py
    python evaluate.py --weights outputs/runs/rescuenet_detect/weights/best.pt
    python evaluate.py --split val --conf 0.3 --name eval_val
    python evaluate.py --visualize --name eval_vis

Requirements:
    pip install ultralytics pyyaml opencv-python-headless
"""

import argparse
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
DATA_YAML  = REPO_ROOT / "data" / "rescuenet_yolo" / "data.yaml"
RUNS_DIR   = REPO_ROOT / "outputs" / "runs"

# Active class names — must match the order defined in prepare_rescuenet.py.
# YOLO indices 0-4 correspond to these five disaster-relevant classes.
YOLO_CLASS_NAMES = [
    "water",
    "building-major-damage",
    "building-total-destruction",
    "road-blocked",
    "vehicle",
]

# Thresholds for the per-class confusion summary
LOW_RECALL_THRESH    = 0.50  # below this → class is being missed by the model
LOW_PRECISION_THRESH = 0.50  # below this → class has too many false positives

# Max images to annotate and save when --visualize is requested
VISUALIZE_SAMPLE_N = 16

# Total number of pipeline steps shown in progress indicators
N_STEPS = 4

def parse_args() -> argparse.Namespace:
    _default_weights = str(RUNS_DIR / "rescuenet_detect" / "weights" / "best.pt")
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 checkpoint on the RescueNet dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=str, default=_default_weights,
                        help="Path to the .pt checkpoint to evaluate")
    parser.add_argument("--data", type=str, default=str(DATA_YAML),
                        help="Path to data.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size in pixels")
    parser.add_argument("--batch", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: GPU index (e.g. '0') or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for NMS")
    parser.add_argument("--name", type=str, default="eval",
                        help="Output folder name under outputs/runs/")
    parser.add_argument("--visualize", action="store_true",
                        help=f"Save up to {VISUALIZE_SAMPLE_N} annotated prediction "
                             "images to outputs/runs/{name}/visualizations/")
    return parser.parse_args()

def validate_setup(weights_path: Path, data_yaml: Path):
    """
    Confirm that the weights file and data.yaml exist, verify that the
    evaluated split's image directory is reachable, and load the model.

    Returns the loaded YOLO model.
    """
    print(f"\n[1/{N_STEPS}] Validating setup...")

    # --- weights ---
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            "Run train.py first, or pass --weights with the correct path."
        )
    print(f"  weights    : {weights_path}  ✓")

    # --- data.yaml ---
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}\n"
            "Run prepare_rescuenet.py first to build the YOLO-format dataset."
        )
    print(f"  data.yaml  : {data_yaml}  ✓")

    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    nc    = cfg.get("nc", "?")
    names = cfg.get("names", [])
    print(f"  classes    : {nc} → {names}")

    # Verify dataset root is reachable (stale absolute paths are common after
    dataset_root = Path(cfg.get("path", str(data_yaml.parent)))
    if not dataset_root.exists():
        dataset_root = data_yaml.parent
        print(f"  WARNING: data.yaml 'path' not found; using yaml directory: {dataset_root}")
    
    from ultralytics import YOLO
    print(f"\n  Loading model: {weights_path.name} ...")
    model = YOLO(str(weights_path))
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Model loaded  ✓  ({n_params:,} parameters)")

    return model

def run_evaluation(args: argparse.Namespace, model, data_yaml: Path, out_dir: Path):
    """
    Run model.val() on the requested split.

    Ultralytics saves plots (confusion matrix, PR curve) automatically to
    out_dir. Returns the Ultralytics DetMetrics object.
    """
    print(f"\n[2/{N_STEPS}] Running evaluation on '{args.split}' split...")
    print(f"  conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}  "
          f"batch={args.batch}  device={args.device}")
    print(f"  output dir : {out_dir}")
    print()

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    return metrics

def _build_report(metrics, args: argparse.Namespace, weights_path: Path) -> str:
    """
    Build the plain-text evaluation report.
    Covers overall metrics, per-class breakdown, and a confusion summary.
    Returns the full report as a single string.
    """
    box = metrics.box

    # Ultralytics stores per-class results aligned with ap_class_index.
    # Convert to plain Python lists so we can use list.index() safely.
    class_indices  = [int(i) for i in box.ap_class_index]
    per_class_p    = [float(v) for v in box.p]
    per_class_r    = [float(v) for v in box.r]
    per_class_ap50 = [float(v) for v in box.ap50]

    lines = []
    
    lines.append("=" * 62)
    lines.append("StitchWise — RescueNet Evaluation Summary")
    lines.append("=" * 62)
    lines.append(f"  weights    : {weights_path}")
    lines.append(f"  split      : {args.split}")
    lines.append(f"  conf / iou : {args.conf} / {args.iou}")
    lines.append("")
    
    lines.append("  Overall Results")
    lines.append("  " + "-" * 40)
    lines.append(f"  mAP50      : {box.map50:.4f}")
    lines.append(f"  mAP50-95   : {box.map:.4f}")
    lines.append(f"  Precision  : {box.mp:.4f}  (mean over all classes)")
    lines.append(f"  Recall     : {box.mr:.4f}  (mean over all classes)")
    lines.append("")
    
    lines.append("  Per-class Breakdown")
    lines.append("  " + "-" * 40)
    col_w = 32
    header = (f"  {'Class':<{col_w}s}  {'Precision':>9s}  "
              f"{'Recall':>9s}  {'mAP50':>9s}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for yolo_idx, class_name in enumerate(YOLO_CLASS_NAMES):
        if yolo_idx in class_indices:
            pos    = class_indices.index(yolo_idx)
            p_val  = per_class_p[pos]
            r_val  = per_class_r[pos]
            ap_val = per_class_ap50[pos]
            lines.append(
                f"  {class_name:<{col_w}s}  {p_val:>9.4f}  "
                f"{r_val:>9.4f}  {ap_val:>9.4f}"
            )
        else:
            # Class was not present in the evaluated split
            lines.append(
                f"  {class_name:<{col_w}s}  {'n/a':>9s}  "
                f"{'n/a':>9s}  {'n/a':>9s}"
            )
    lines.append("")

    # --- Confusion summary ---
    # Classify each class into two independent warning buckets.
    # A class can appear in both (low P AND low R) simultaneously.
    missed = []   # low recall  → model is failing to detect these
    noisy  = []   # low precision → model over-predicts (false positives)

    for yolo_idx, class_name in enumerate(YOLO_CLASS_NAMES):
        if yolo_idx not in class_indices:
            missed.append(f"{class_name}  [absent from '{args.split}' split]")
            continue
        pos   = class_indices.index(yolo_idx)
        p_val = per_class_p[pos]
        r_val = per_class_r[pos]
        if r_val < LOW_RECALL_THRESH:
            missed.append(f"{class_name}  (recall={r_val:.2f})")
        if p_val < LOW_PRECISION_THRESH:
            noisy.append(f"{class_name}  (precision={p_val:.2f})")

    lines.append("  Confusion Summary")
    lines.append("  " + "-" * 40)

    if missed:
        lines.append(f"  Classes being MISSED  (recall < {LOW_RECALL_THRESH}):")
        for c in missed:
            lines.append(f"    ✗  {c}")
    else:
        lines.append(f"  ✓  All classes above recall threshold ({LOW_RECALL_THRESH})")

    lines.append("")

    if noisy:
        lines.append(f"  Classes with FALSE POSITIVES  "
                     f"(precision < {LOW_PRECISION_THRESH}):")
        for c in noisy:
            lines.append(f"    ✗  {c}")
    else:
        lines.append(f"  ✓  All classes above precision threshold ({LOW_PRECISION_THRESH})")

    lines.append("")
    lines.append("=" * 62)
    return "\n".join(lines)


def report_and_save(metrics, args: argparse.Namespace,
                    weights_path: Path, out_dir: Path):
    """Print the evaluation report to the console and save it to a text file."""
    print(f"\n[3/{N_STEPS}] Reporting results...")

    report_text = _build_report(metrics, args, weights_path)
    print("\n" + report_text)
    
    summary_path = out_dir / "eval_summary.txt"
    summary_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Summary saved → {summary_path}")

def visualize_predictions(args: argparse.Namespace, model,
                          data_yaml: Path, out_dir: Path):
    """
    Run model.predict() on a random sample of images from the evaluated split
    and save the annotated (bounding-box-overlaid) images for manual inspection.

    Images are written to outputs/runs/{name}/visualizations/.
    """
    import yaml
    import cv2

    print(f"\n[4/{N_STEPS}] Generating visualizations "
          f"(up to {VISUALIZE_SAMPLE_N} images)...")
    
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("path", str(data_yaml.parent)))
    if not dataset_root.exists():
        dataset_root = data_yaml.parent

    split_rel = cfg.get(args.split, f"{args.split}/images")
    img_dir   = dataset_root / split_rel

    if not img_dir.exists():
        print(f"  WARNING: image directory not found: {img_dir}")
        print("  Skipping visualization.")
        return
    
    img_paths = (
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG")) +
        list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG"))
    )

    if not img_paths:
        print(f"  WARNING: no images found in {img_dir}. Skipping visualization.")
        return

    sample = random.sample(img_paths, min(VISUALIZE_SAMPLE_N, len(img_paths)))
    print(f"  Sampling {len(sample)} / {len(img_paths)} images from:")
    print(f"    {img_dir}")
    
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sample:
        preds = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
            verbose=False,
        )
        for pred in preds:
            annotated = pred.plot()
            cv2.imwrite(str(viz_dir / img_path.name), annotated)

    n_saved = len(list(viz_dir.glob("*")))
    print(f"  Saved {n_saved} annotated image(s) → {viz_dir}")

def main():
    args = parse_args()
    weights_path = Path(args.weights)
    data_yaml    = Path(args.data)
    out_dir      = RUNS_DIR / args.name

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  StitchWise — RescueNet YOLOv8 Evaluation               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  weights    : {weights_path}")
    print(f"  data       : {data_yaml}")
    print(f"  split      : {args.split}")
    print(f"  conf / iou : {args.conf} / {args.iou}")
    print(f"  imgsz      : {args.imgsz}  batch: {args.batch}  device: {args.device}")
    print(f"  run name   : {args.name}")
    print(f"  output dir : {out_dir}")
    print(f"  visualize  : {'yes' if args.visualize else 'no'}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    model = validate_setup(weights_path, data_yaml)
    metrics = run_evaluation(args, model, data_yaml, out_dir)
    report_and_save(metrics, args, weights_path, out_dir)
    if args.visualize:
        visualize_predictions(args, model, data_yaml, out_dir)
    else:
        print(f"\n[4/{N_STEPS}] Visualization skipped  "
              "(pass --visualize to generate annotated images)")

    print(f"\n{'='*62}")
    print("  Evaluation complete.")
    print(f"  All outputs saved to: {out_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
