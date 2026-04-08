"""Data loading utilities for RescueNet dataset"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any


class RescueNetLoader:
    """Loader for RescueNet dataset with annotation support"""

    def __init__(self, data_root: str, split: str = "train"):
        """
        Initialize RescueNet dataset loader

        Args:
            data_root: Root directory of RescueNet dataset
            split: Dataset split ("train", "val", or "test")
        """
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split

        # Get list of images
        self.image_files = sorted(list(self.split_dir.glob("*.jpg")) +
                                  list(self.split_dir.glob("*.png")))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load image and annotations

        Args:
            idx: Index in dataset

        Returns:
            Tuple of (image array, annotations dict)
        """
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))

        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations if they exist
        annotations = self._load_annotations(img_path)

        return image, annotations

    def _load_annotations(self, img_path: Path) -> Dict[str, Any]:
        """Load annotations for an image"""
        # This will be dataset-specific - adjust based on RescueNet format
        annotations = {
            "bboxes": [],
            "labels": [],
            "masks": None
        }
        return annotations


class YOLODataset:
    """PyTorch Dataset wrapper for YOLO format data"""

    def __init__(self, yaml_path: str, split: str = "train", augment: bool = False):
        """
        Initialize YOLO format dataset

        Args:
            yaml_path: Path to data.yaml file
            split: Dataset split (train/val/test)
            augment: Apply data augmentation
        """
        self.yaml_path = Path(yaml_path)
        self.split = split
        self.augment = augment

        # TODO: Load dataset paths and metadata from yaml_path
        self.image_dir = None
        self.label_dir = None

    def __len__(self) -> int:
        # TODO: Return number of images in split
        pass

    def __getitem__(self, idx: int):
        # TODO: Return image and labels for YOLO
        pass
