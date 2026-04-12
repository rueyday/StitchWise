"""Canonical class definitions for StitchWise.

Single source of truth for all class mappings.  Import from here — do not
define class names or colors in any other module.

If you change which classes are trained (data_preparation.active_classes in
config.yaml), update CLASS_COLORS below to match the new class order and
retrain the model.
"""

# Full RescueNet semantic segmentation taxonomy (grayscale mask pixel value → name).
RESCUENET_CLASSES = {
    0:  "background",
    1:  "water",
    2:  "building-no-damage",
    3:  "building-minor-damage",
    4:  "building-major-damage",
    5:  "building-total-destruction",
    6:  "road-clear",
    7:  "road-blocked",
    8:  "vehicle",
    9:  "tree",
    10: "pool",
}

# BGR colors for visualization, indexed by YOLO class ID (0-based).
# The order corresponds to active_classes in config.yaml:
#   index 0 → pixel 1  (water)
#   index 1 → pixel 4  (building-major-damage)
#   index 2 → pixel 5  (building-total-destruction)
#   index 3 → pixel 7  (road-blocked)
#   index 4 → pixel 8  (vehicle)
CLASS_COLORS = {
    0: (255, 165,   0),   # water                     — orange
    1: (  0,   0, 255),   # building-major-damage      — red
    2: (128,   0, 128),   # building-total-destruction — purple
    3: (  0, 165, 255),   # road-blocked               — amber
    4: (  0, 255, 255),   # vehicle                    — yellow
}


def build_class_map(active_classes: list) -> dict:
    """Build a YOLO class-index → name mapping from an active_classes list.

    Args:
        active_classes: Ordered list of RescueNet pixel values used during
                        training, e.g. [1, 4, 5, 7, 8].  Read from
                        config.yaml data_preparation.active_classes.

    Returns:
        Dict mapping YOLO 0-based index to class name string.
    """
    return {
        yolo_idx: RESCUENET_CLASSES[orig]
        for yolo_idx, orig in enumerate(active_classes)
    }
