"""Model utilities for YOLO-based damage detection"""

import torch
from pathlib import Path
from typing import Optional
from ultralytics import YOLO


class DamageDetector:
    """Wrapper for YOLO damage detection model"""

    def __init__(
        self,
        model_name: str = "yolov8s-seg",
        pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize damage detector model

        Args:
            model_name: YOLO model variant (e.g., "yolov8n-seg", "yolov8s-seg")
            pretrained: Load pretrained weights
            device: Device to load model on ("cuda" or "cpu")
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.model.to(device)

    @property
    def model_info(self) -> dict:
        """Get model information"""
        return {
            "name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.model.parameters())
        }

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        save_dir: str = "./models/checkpoints",
        **kwargs
    ) -> dict:
        """
        Train the model on custom dataset

        Args:
            data_yaml: Path to YOLO data.yaml configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            imgsz: Input image size
            save_dir: Directory to save checkpoints
            **kwargs: Additional YOLO training arguments

        Returns:
            Training results dictionary
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            project=save_dir,
            name="run",
            patience=20,
            save=True,
            **kwargs
        )
        return results

    def predict(
        self,
        source,
        conf: float = 0.5,
        iou: float = 0.45,
        **kwargs
    ):
        """
        Run inference on image(s)

        Args:
            source: Image path, batch of images, or directory
            conf: Confidence threshold
            iou: IoU threshold for NMS
            **kwargs: Additional YOLO prediction arguments

        Returns:
            Prediction results
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            **kwargs
        )
        return results

    def save_checkpoint(self, save_path: str) -> None:
        """Save model weights"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights"""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path))
