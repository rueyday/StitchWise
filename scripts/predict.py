"""Inference script for RescueNet damage segmentation model.

Runs a trained model on one image or a directory of images, prints detections
to stdout, and saves annotated result images.

Usage:
    python scripts/predict.py --source IMAGE_OR_DIR
    python scripts/predict.py --source data/raw/rescuenet/RescueNet/test/test-org-img/10807.jpg
    python scripts/predict.py --source data/raw/ --model models/checkpoints/rescuenet_seg/weights/best.pt
    python scripts/predict.py --source data/raw/ --save-dir runs/predict
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import cv2
import yaml

from src.models.detector import DamageDetector
from src.utils.classes import build_class_map
from src.utils.logger import setup_logger, get_logger
from src.utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Run inference with the damage segmentation model")
    parser.add_argument("--source", required=True,
                        help="Image path, directory, or glob pattern.")
    parser.add_argument("--model", default=None,
                        help="Checkpoint path (.pt). Overrides config inference.model_path.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to project config.yaml (default: config.yaml)")
    parser.add_argument("--save-dir", default="runs/predict",
                        help="Directory to save annotated result images (default: runs/predict)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    log_cfg = cfg["logging"]
    setup_logger(log_level=log_cfg["log_level"])
    log = get_logger("predict")

    inf_cfg = cfg["inference"]
    model_path = args.model or inf_cfg["model_path"]

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Train first with scripts/train.py, or pass --model PATH."
        )

    active_classes = cfg["data_preparation"]["active_classes"]
    class_map = build_class_map(active_classes)

    detector = DamageDetector(
        model_name=model_path,
        device=cfg["model"]["device"],
    )
    log.info(f"Loaded model: {model_path}")

    results = detector.predict(
        source=args.source,
        conf=inf_cfg["confidence_threshold"],
        iou=inf_cfg["iou_threshold"],
        save=False,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        n_det = len(result.boxes) if result.boxes else 0
        log.info(f"Image {i + 1}: {n_det} detection(s)")

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = class_map.get(class_id, f"class_{class_id}")
            confidence = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy().astype(int)
            log.info(f"  {class_name} ({confidence * 100:.1f}%) at {coords}")

        # Build annotated image
        img = result.orig_img
        if img is None:
            continue

        if n_det > 0:
            bboxes = [box.xyxy[0].cpu().numpy() for box in result.boxes]
            class_ids = [int(box.cls[0]) for box in result.boxes]
            confs = [float(box.conf[0]) for box in result.boxes]
            annotated = Visualizer.draw_bboxes(img, bboxes, class_ids, confs, class_map)

            if result.masks is not None:
                masks_np = result.masks.data.cpu().numpy()  # (N, H, W)
                h, w = img.shape[:2]
                resized_masks = [
                    cv2.resize(masks_np[j], (w, h), interpolation=cv2.INTER_NEAREST)
                    for j in range(len(masks_np))
                ]
                annotated = Visualizer.draw_masks(annotated, resized_masks, class_ids)
        else:
            annotated = img

        out_path = save_dir / f"result_{i:04d}.jpg"
        Visualizer.save_result_image(annotated, str(out_path))
        log.info(f"  Saved annotated image: {out_path}")

    log.info(f"Inference complete. Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
