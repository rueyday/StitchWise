from ultralytics import YOLO
from pathlib import Path

def train_production():
    data_yaml = Path("data/processed/data.yaml").resolve()
    
    # Using the Nano model to fit comfortably in 4GB VRAM
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,        # Increased epochs for better convergence
        imgsz=640,
        batch=16,          # Optimized for 4GB VRAM
        device=0,
        project="models/checkpoints",
        name="rescuenet_seg_500img",
        save=True,
        plots=True,
        patience=20,       # Early stopping if no improvement for 20 epochs
        cache=True         # Speeds up training if you have enough system RAM
    )

if __name__ == "__main__":
    train_production()