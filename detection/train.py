"""
StitchWise — YOLOv8 Fine-tuning on RescueNet
=============================================

Usage:
    python train.py
    python train.py --epochs 100 --batch 8 --freeze-epochs 15
    python train.py --model yolov8m-seg.pt --name rescuenet_seg
    python train.py --resume --name rescuenet_detect
    python train.py --dry-run

Requirements:
    pip install ultralytics pyyaml
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
DATA_YAML  = REPO_ROOT / "data" / "rescuenet_yolo" / "data.yaml"
RUNS_DIR   = REPO_ROOT / "outputs" / "runs"

FREEZE_BACKBONE_LAYERS = 10
DEFAULT_FREEZE_EPOCHS = 10

AERIAL_AUGMENT = {
    "degrees":     0.0,   # disable rotation
    "perspective": 0.0,   # disable perspective warp
    "shear":       0.0,   # disable shear
    "flipud":      0.5,   # vertical flip  (50 % probability)
    "fliplr":      0.5,   # horizontal flip (50 % probability)
    "scale":       0.5,   # random scale ±50 %
    "translate":   0.1,   # random translation ±10 %
    "mosaic":      1.0,   # mosaic always on
    "hsv_h":       0.015, # hue jitter
    "hsv_s":       0.7,   # saturation jitter
    "hsv_v":       0.4,   # brightness jitter
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on RescueNet for aerial disaster detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total training epochs")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (-1 for Ultralytics auto-batch)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size in pixels")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Pretrained checkpoint to start from. "
                             "Use yolov8m-seg.pt for segmentation mode.")
    parser.add_argument("--name", type=str, default="rescuenet_detect",
                        help="Run name; controls the output subdirectory under outputs/runs/")
    parser.add_argument("--device", type=str, default="0",
                        help="Training device: GPU index (e.g. '0', '0,1') or 'cpu'")
    parser.add_argument("--freeze-epochs", type=int, default=DEFAULT_FREEZE_EPOCHS,
                        help="Epochs to train with frozen backbone before unfreezing. "
                             "Set 0 to skip phase-1 (train full model from the start).")
    parser.add_argument("--data", type=str, default=str(DATA_YAML),
                        help="Path to the YOLO data.yaml file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the most recent interrupted run for --name. "
                             "Loads outputs/runs/{name}/weights/last.pt.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data.yaml and model without starting training")
    return parser.parse_args()

def validate_setup(data_yaml: Path, model_name: str):
    print("\n[1/3] Validating setup...")
    
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at: {data_yaml}\n"
            "Run prepare_rescuenet.py first to build the YOLO-format dataset."
        )
    print(f"  data.yaml  : {data_yaml}  ✓")

    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    nc    = cfg.get("nc", "?")
    names = cfg.get("names", [])
    print(f"  classes    : {nc} → {names}")

    dataset_root = Path(cfg.get("path", data_yaml.parent))
    if not dataset_root.exists():
        dataset_root = data_yaml.parent
        print(f"  WARNING: data.yaml 'path' not found; using yaml directory instead: {dataset_root}")
    all_splits_ok = True
    for split_key in ("train", "val", "test"):
        rel = cfg.get(split_key, "")
        if not rel:
            continue
        split_path = dataset_root / rel
        ok = split_path.exists()
        status = "✓" if ok else "MISSING ✗"
        print(f"  {split_key:<6s}     : {split_path}  {status}")
        if not ok:
            all_splits_ok = False

    if not all_splits_ok:
        raise FileNotFoundError(
            "One or more dataset splits are missing. "
            "Re-run prepare_rescuenet.py to regenerate them."
        )
    
    from ultralytics import YOLO
    print(f"\n  Loading checkpoint: {model_name} ...")
    model = YOLO(model_name)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Model loaded       ✓  ({n_params:,} parameters)")

    return model

def _common_train_kwargs(args: argparse.Namespace, data_yaml: Path) -> dict:
    """Build the kwargs shared by both training phases."""
    return {
        "data":       str(data_yaml),
        "imgsz":      args.imgsz,
        "batch":      args.batch,
        "device":     args.device,
        "project":    str(RUNS_DIR),
        "exist_ok":   True,   # don't crash if the run dir already exists
        "plots":      True,   # save training-curve plots
        "verbose":    True,
        **AERIAL_AUGMENT,
    }


def train(args: argparse.Namespace, model, data_yaml: Path):
    from ultralytics import YOLO

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.resume:
        last_pt = RUNS_DIR / args.name / "weights" / "last.pt"
        if not last_pt.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {last_pt}\n"
                "Check --name matches the run you want to continue."
            )
        print(f"\n[2/3] Resuming training from: {last_pt}")
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
        best_pt = RUNS_DIR / args.name / "weights" / "best.pt"
        return results, best_pt

    kwargs = _common_train_kwargs(args, data_yaml)
    use_phased = 0 < args.freeze_epochs < args.epochs

    if use_phased:
        phase1_name = f"{args.name}_phase1"
        phase1_dir  = RUNS_DIR / phase1_name

        print(f"\n{'='*62}")
        print(f"[2/3] Phase 1 — frozen backbone "
              f"({args.freeze_epochs} / {args.epochs} epochs)")
        print(f"      Freezing first {FREEZE_BACKBONE_LAYERS} backbone layers")
        print(f"      Output: {phase1_dir}")
        print(f"{'='*62}")

        model.train(
            epochs=args.freeze_epochs,
            freeze=FREEZE_BACKBONE_LAYERS,
            name=phase1_name,
            **kwargs,
        )

        phase1_last = phase1_dir / "weights" / "last.pt"
        if not phase1_last.exists():
            raise RuntimeError(
                f"Phase 1 did not produce a checkpoint at {phase1_last}. "
                "Training may have failed — check the log above."
            )

        remaining_epochs = args.epochs - args.freeze_epochs
        phase2_dir = RUNS_DIR / args.name

        print(f"\n{'='*62}")
        print(f"[2/3] Phase 2 — full fine-tuning "
              f"({remaining_epochs} remaining epochs)")
        print(f"      Loading phase-1 weights: {phase1_last}")
        print(f"      Output: {phase2_dir}")
        print(f"{'='*62}")

        model2 = YOLO(str(phase1_last))
        results = model2.train(
            epochs=remaining_epochs,
            name=args.name,
            **kwargs,
        )
        best_pt = phase2_dir / "weights" / "best.pt"

    else:
        freeze = FREEZE_BACKBONE_LAYERS if args.freeze_epochs >= args.epochs else None

        phase_label = (
            "frozen backbone (entire run)"  if freeze else
            "full model (no backbone freeze)"
        )
        print(f"\n{'='*62}")
        print(f"[2/3] Training — {phase_label}")
        print(f"      Epochs: {args.epochs}  |  Batch: {args.batch}  "
              f"|  imgsz: {args.imgsz}")
        print(f"      Output: {RUNS_DIR / args.name}")
        print(f"{'='*62}")

        train_kwargs = dict(epochs=args.epochs, name=args.name, **kwargs)
        if freeze:
            train_kwargs["freeze"] = freeze

        results = model.train(**train_kwargs)
        best_pt = RUNS_DIR / args.name / "weights" / "best.pt"

    return results, best_pt

def report(best_pt: Path, results):
    """Print a summary of training results and the path to best weights."""
    print(f"\n{'='*62}")
    print("[3/3] Training complete")
    print(f"{'='*62}")

    if best_pt.exists():
        print(f"  Best weights : {best_pt}")
    else:
        last_pt = best_pt.parent / "last.pt"
        print(f"  WARNING: best.pt not found at {best_pt}")
        if last_pt.exists():
            print(f"  last.pt      : {last_pt}  (use this instead)")
    
    try:
        box = results.results_dict
        print(f"\n  mAP50      : {box.get('metrics/mAP50(B)',    'n/a'):.4f}")
        print(f"  mAP50-95   : {box.get('metrics/mAP50-95(B)', 'n/a'):.4f}")
        print(f"  Precision  : {box.get('metrics/precision(B)', 'n/a'):.4f}")
        print(f"  Recall     : {box.get('metrics/recall(B)',    'n/a'):.4f}")
    except Exception:
        pass  # metrics may not be available for all run types

    print(f"\n  Next step — evaluate on the test split:")
    print(f"    python evaluate.py --weights {best_pt} --data {DATA_YAML}")
    print(f"{'='*62}\n")

def main():
    args = parse_args()
    data_yaml = Path(args.data)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  StitchWise — RescueNet YOLOv8 Training                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  model        : {args.model}")
    print(f"  data         : {data_yaml}")
    print(f"  epochs       : {args.epochs}  (freeze first {args.freeze_epochs})")
    print(f"  batch / imgsz: {args.batch} / {args.imgsz}")
    print(f"  device       : {args.device}")
    print(f"  run name     : {args.name}")
    print(f"  output root  : {RUNS_DIR}")
    
    model = validate_setup(data_yaml, args.model)

    if args.dry_run:
        print("\n[Dry-run] All checks passed. Exiting without training.")
        sys.exit(0)
    
    results, best_pt = train(args, model, data_yaml)
    report(best_pt, results)


if __name__ == "__main__":
    main()
