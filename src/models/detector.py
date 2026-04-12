"""YOLO-based damage detection model wrapper."""

from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO


class DamageDetector:
    """Thin wrapper around a YOLO segmentation model.

    Provides a stable interface for training and inference so that scripts
    never call ultralytics directly.  All hyperparameters come from config.yaml
    and are forwarded via explicit arguments or **kwargs.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-seg",
        pretrained: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize the detector.

        Args:
            model_name: YOLO model variant name (e.g. "yolov8n-seg") or path
                        to a checkpoint (.pt file) for inference.
            pretrained: Ignored when model_name is a checkpoint path; kept for
                        API symmetry.
            device: "cuda" or "cpu".  Auto-detected when None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.model.to(device)

    @property
    def model_info(self) -> dict:
        """Return a summary dict of model metadata."""
        return {
            "name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.model.parameters()),
        }

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        save_dir: str = "models/checkpoints",
        name: str = "run",
        patience: int = 20,
        **kwargs,
    ) -> object:
        """Train the model on a YOLO-format dataset.

        Args:
            data_yaml: Absolute or relative path to the dataset data.yaml.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            imgsz: Input image size.
            save_dir: Root directory for checkpoint output.
            name: Subdirectory name within save_dir for this run.
            patience: Early stopping patience (epochs without improvement).
            **kwargs: Any additional ultralytics train() arguments
                      (e.g. fl_gamma, hsv_h, hsv_s, hsv_v, cache).

        Returns:
            ultralytics training results object.
        """
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            project=save_dir,
            name=name,
            patience=patience,
            save=True,
            **kwargs,
        )

    def predict(
        self,
        source,
        conf: float = 0.5,
        iou: float = 0.45,
        **kwargs,
    ) -> list:
        """Run inference on one or more images.

        Args:
            source: Image path, directory, or glob pattern.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            **kwargs: Any additional ultralytics predict() arguments.

        Returns:
            List of ultralytics Results objects.
        """
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            **kwargs,
        )
