"""YOLOv8 segmentation training script for RescueNet damage detection.

Reads all hyperparameters from config.yaml.  Pass --resume with a checkpoint
path to continue an interrupted run (replaces the old train_trial.py).

Usage:
    python scripts/train.py
    python scripts/train.py --config config.yaml
    python scripts/train.py --resume models/checkpoints/rescuenet_seg/weights/last.pt
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import yaml

from src.models.detector import DamageDetector
from src.utils.logger import setup_logger, get_logger


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg on RescueNet")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to project config.yaml (default: config.yaml)")
    parser.add_argument("--resume", metavar="CHECKPOINT", default=None,
                        help="Checkpoint path to resume training from (last.pt).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    log_cfg = cfg["logging"]
    setup_logger(
        log_level=log_cfg["log_level"],
        log_file=str(Path(log_cfg["save_dir"]) / log_cfg["log_file"]),
    )
    log = get_logger("train")

    # --- Resume mode ---
    if args.resume:
        log.info(f"Resuming training from checkpoint: {args.resume}")
        from ultralytics import YOLO
        model = YOLO(args.resume)
        model.train(resume=True)
        log.info("Resumed training complete.")
        return

    # --- Fresh training run ---
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    dataset_cfg = cfg["dataset"]

    data_yaml = Path(dataset_cfg["data_root"]) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. "
            "Run scripts/prepare_data.py first."
        )

    detector = DamageDetector(
        model_name=model_cfg["architecture"],
        pretrained=model_cfg["pretrained"],
        device=model_cfg["device"],
    )
    log.info(f"Model info: {detector.model_info}")

    detector.train(
        data_yaml=str(data_yaml.resolve()),
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        imgsz=dataset_cfg["img_size"],
        save_dir=log_cfg["save_dir"],
        name=model_cfg["name"],
        patience=train_cfg["patience"],
        # Focal loss + augmentation for class imbalance
        hsv_h=aug_cfg["hsv_h"],
        hsv_s=aug_cfg["hsv_s"],
        hsv_v=aug_cfg["hsv_v"],
        cache=True,
        plots=True,
    )
    log.info("Training complete.")


if __name__ == "__main__":
    main()
