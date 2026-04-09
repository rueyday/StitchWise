from ultralytics import YOLO
from pathlib import Path

def train_production():
    data_yaml = Path("data/processed/data.yaml").resolve()
    
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,        
        imgsz=640,
        batch=16,          
        device=0,
        project="models/checkpoints",
        name="rescuenet_seg_500img_balanced",
        save=True,
        plots=True,
        patience=20,      
        cache=True,
        
        # --- NEW: Handle Class Imbalance & False Positives ---
        fl_gamma=2.0,      # Enables Focal Loss (default is 0.0). Reduces weight of "easy" water pixels.
        hsv_h=0.015,       # Slightly alter colors to decouple "brown" from "muddy water"
        hsv_s=0.7,         # Aggressive saturation shifts
        hsv_v=0.4          # Aggressive brightness shifts to prevent shadow confusion
    )

if __name__ == "__main__":
    train_production()