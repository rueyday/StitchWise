"""Visualization utilities for damage detection results"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path


class Visualizer:
    """Visualization utilities for detection and segmentation results"""

    DAMAGE_COLORS = {
        0: (0, 255, 0),        # No Damage - Green
        1: (0, 165, 255),      # Minor Damage - Orange
        2: (0, 0, 255),        # Major Damage - Red
        3: (128, 0, 128),      # Destroyed - Purple
    }

    CLASS_NAMES = {
        0: "No Damage",
        1: "Minor Damage",
        2: "Major Damage",
        3: "Destroyed",
    }

    @staticmethod
    def draw_bboxes(
        image: np.ndarray,
        bboxes: List[Tuple],
        class_ids: List[int],
        confidences: Optional[List[float]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes on image

        Args:
            image: Input image (BGR)
            bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            class_ids: List of class IDs
            confidences: Optional list of confidence scores
            thickness: Line thickness

        Returns:
            Image with drawn bboxes
        """
        output = image.copy()

        for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
            x1, y1, x2, y2 = map(int, bbox)
            color = Visualizer.DAMAGE_COLORS.get(class_id, (255, 255, 255))
            label = Visualizer.CLASS_NAMES.get(class_id, "Unknown")

            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            conf_text = f" {confidences[i]:.2f}" if confidences else ""
            text = f"{label}{conf_text}"
            cv2.putText(
                output,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return output

    @staticmethod
    def draw_masks(
        image: np.ndarray,
        masks: List[np.ndarray],
        class_ids: List[int],
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Draw segmentation masks on image

        Args:
            image: Input image (BGR)
            masks: List of binary masks
            class_ids: List of class IDs
            alpha: Transparency coefficient

        Returns:
            Image with drawn masks
        """
        output = image.copy()

        for mask, class_id in zip(masks, class_ids):
            color = Visualizer.DAMAGE_COLORS.get(class_id, (255, 255, 255))

            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color

            # Blend mask with image
            output = cv2.addWeighted(output, 1 - alpha, colored_mask, alpha, 0)

        return output

    @staticmethod
    def plot_predictions(
        image: np.ndarray,
        results,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot detection/segmentation results

        Args:
            image: Input image
            results: YOLO results object
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

        # TODO: Extract and plot bboxes/masks from results
        # This depends on YOLO results format

        ax.set_title("Damage Detection Results")
        ax.axis("off")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        else:
            plt.show()

        plt.close()

    @staticmethod
    def save_result_image(
        image: np.ndarray,
        output_path: str
    ) -> None:
        """Save result image"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
