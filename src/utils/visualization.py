"""Visualization utilities for damage detection results."""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils.classes import CLASS_COLORS, build_class_map


class Visualizer:
    """Visualization utilities for detection and segmentation results."""

    @staticmethod
    def draw_bboxes(
        image: np.ndarray,
        bboxes: list,
        class_ids: list,
        confidences: Optional[list] = None,
        class_map: Optional[dict] = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes on an image.

        Args:
            image: Input image (BGR).
            bboxes: List of bounding boxes as (x1, y1, x2, y2) arrays or tuples.
            class_ids: List of integer YOLO class IDs.
            confidences: Optional list of confidence scores.
            class_map: Optional dict mapping class ID → name (from build_class_map).
                       If None, class IDs are used as labels.
            thickness: Rectangle line thickness.

        Returns:
            Image copy with drawn boxes and labels.
        """
        output = image.copy()
        for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
            x1, y1, x2, y2 = map(int, bbox)
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            label = class_map.get(class_id, str(class_id)) if class_map else str(class_id)
            conf_text = f" {confidences[i]:.2f}" if confidences else ""
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                output,
                f"{label}{conf_text}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return output

    @staticmethod
    def draw_masks(
        image: np.ndarray,
        masks: list,
        class_ids: list,
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Overlay segmentation masks on an image.

        Args:
            image: Input image (BGR).
            masks: List of 2-D binary or float mask arrays (H x W).
            class_ids: List of integer YOLO class IDs.
            alpha: Mask transparency (0 = invisible, 1 = opaque).

        Returns:
            Image copy with blended mask overlays.
        """
        output = image.copy()
        for mask, class_id in zip(masks, class_ids):
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            output = cv2.addWeighted(output, 1 - alpha, colored_mask, alpha, 0)
        return output

    @staticmethod
    def plot_predictions(
        image: np.ndarray,
        results,
        class_map: Optional[dict] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot detection and segmentation results using matplotlib.

        Args:
            image: Input image (BGR).
            results: Single ultralytics YOLO result object (results[0]).
            class_map: Optional dict mapping class ID → name.
            save_path: If given, save the figure here instead of showing it.
        """
        annotated = image.copy()

        if results.boxes and len(results.boxes):
            bboxes = [box.xyxy[0].cpu().numpy() for box in results.boxes]
            class_ids = [int(box.cls[0]) for box in results.boxes]
            confs = [float(box.conf[0]) for box in results.boxes]
            annotated = Visualizer.draw_bboxes(annotated, bboxes, class_ids, confs, class_map)

        if results.masks is not None:
            masks_np = results.masks.data.cpu().numpy()  # (N, H, W)
            class_ids = [int(box.cls[0]) for box in results.boxes]
            # Resize each mask to match the original image dimensions
            h, w = image.shape[:2]
            resized = [
                cv2.resize(masks_np[j], (w, h), interpolation=cv2.INTER_NEAREST)
                for j in range(len(masks_np))
            ]
            annotated = Visualizer.draw_masks(annotated, resized, class_ids)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title("Damage Detection Results")
        ax.axis("off")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        else:
            plt.show()
        plt.close()

    @staticmethod
    def save_result_image(image: np.ndarray, output_path: str) -> None:
        """Save an annotated result image to disk.

        Args:
            image: Annotated image array (BGR).
            output_path: Destination file path.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
