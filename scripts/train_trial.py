from ultralytics import YOLO

def train_production():
    # 1. Load the partially trained model checkpoint
    checkpoint_path = "runs/segment/models/checkpoints/rescuenet_seg_production3/weights/last.pt"
    model = YOLO(checkpoint_path)

    print(f"🚀 Resuming Full Production Run from {checkpoint_path}...")
    
    # 2. Resume training. YOLO automatically remembers your batch=32, workers=4, cache=False, etc.
    model.train(resume=True)

if __name__ == "__main__":
    train_production()