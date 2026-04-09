from ultralytics import YOLO
from pathlib import Path

def train_trial():
    # Make sure this points to your newly fixed directory!
    data_yaml = Path("data/processed/data.yaml").resolve()
    
    # Using the Nano segmentation model for the fastest possible trial
    model = YOLO("yolov8n-seg.pt")

    print("🚀 Starting 5-epoch trial run...")
    
    model.train(
        data=str(data_yaml),
        epochs=100,          # <-- TRIAL RUN: Just 5 epochs
        imgsz=640,
        batch=16,          # If you get a CUDA Out of Memory error, drop this to 8
        device=0,
        project="models/checkpoints",
        name="rescuenet_seg_production",
        save=True,
        plots=True,
        
        # --- Handle False Positives (Shadows vs Water) ---
        hsv_h=0.015,       # Alter colors to decouple "brown" from "muddy water"
        hsv_s=0.7,         # Aggressive saturation shifts
        hsv_v=0.4          # Aggressive brightness shifts
    )

if __name__ == "__main__":
    train_trial()
