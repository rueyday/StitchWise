"""Convert RescueNet annotations to YOLO format"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image


class RescueNetToYOLO:
    """Converter for RescueNet annotations to YOLO format"""

    # Damage classes mapping
    DAMAGE_CLASSES = {
        "No Damage": 0,
        "Minor Damage": 1,
        "Major Damage": 2,
        "Destroyed": 3,
    }

    def __init__(self, output_format: str = "bbox"):
        """
        Initialize converter

        Args:
            output_format: "bbox" for bounding boxes, "segmentation" for masks
        """
        self.output_format = output_format

    def convert_bbox_annotation(
        self,
        annotation: dict,
        image_height: int,
        image_width: int
    ) -> List[str]:
        """
        Convert bounding box annotation to YOLO format (normalized)

        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1]

        Args:
            annotation: Annotation dict with bbox coordinates
            image_height: Height of image
            image_width: Width of image

        Returns:
            List of YOLO format strings
        """
        yolo_lines = []

        # Example structure - adjust based on actual RescueNet annotations
        if "bboxes" in annotation and annotation["bboxes"]:
            for bbox, label in zip(annotation["bboxes"], annotation.get("labels", [])):
                # Assume bbox is [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = bbox

                # Convert to center coordinates
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                width = x_max - x_min
                height = y_max - y_min

                # Normalize to [0, 1]
                x_center /= image_width
                y_center /= image_height
                width /= image_width
                height /= image_height

                # Clamp values
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                class_id = self.DAMAGE_CLASSES.get(label, 0)
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

        return yolo_lines

    def convert_segmentation_annotation(
        self,
        annotation: dict,
        image_height: int,
        image_width: int
    ) -> List[str]:
        """
        Convert segmentation mask to YOLO format (polygon)

        YOLO format: <class_id> <x1> <y1> <x2> <y2> ... (normalized)

        Args:
            annotation: Annotation dict with segmentation mask
            image_height: Height of image
            image_width: Width of image

        Returns:
            List of YOLO format strings
        """
        yolo_lines = []

        # Convert polygon/mask to normalized coordinates
        if "masks" in annotation and annotation["masks"] is not None:
            mask = annotation["masks"]
            label = annotation.get("label", "No Damage")
            class_id = self.DAMAGE_CLASSES.get(label, 0)

            # Find contours from mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) < 2 or contour.shape[0] < 3:
                    continue

                # Normalize contour coordinates
                normalized_contour = contour / np.array([image_width, image_height])
                normalized_contour = np.clip(normalized_contour, 0, 1)

                # Format as YOLO polygon
                polygon_str = " ".join(
                    f"{x:.6f} {y:.6f}"
                    for x, y in normalized_contour
                )
                yolo_lines.append(f"{class_id} {polygon_str}")

        return yolo_lines


def create_yolo_dataset_yaml(
    output_path: str,
    train_dir: str,
    val_dir: str,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Create data.yaml file for YOLO training

    Args:
        output_path: Path to save data.yaml
        train_dir: Path to training images
        val_dir: Path to validation images
        num_classes: Number of damage classes
        class_names: List of class names
    """
    if class_names is None:
        class_names = list(RescueNetToYOLO.DAMAGE_CLASSES.keys())

    yaml_content = f"""# RescueNet YOLO Dataset Configuration
path: {Path(train_dir).parent}
train: {Path(train_dir).name}
val: {Path(val_dir).name}

nc: {num_classes}
names: {class_names}
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_content)
