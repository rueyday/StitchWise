from ultralytics import YOLO
import cv2

def run_inference():
    # 1. Load your brand new custom brain
    model_path = "runs/segment/models/checkpoints/rescuenet_seg_production3/weights/best.pt"
    model = YOLO(model_path)

    # 2. Pick an image to test
    image_path = "data/raw/rescuenet/RescueNet/test/test-org-img/1001.jpg" # Change to a real test image
    
    # 3. Run the model! (conf=0.25 ignores low-confidence guesses)
    print(f"🔍 Analyzing {image_path}...")
    results = model.predict(source=image_path, save=True, conf=0.25)

    # 4. Extract the data for your teammates!
    # A single image returns a list with 1 result object
    result = results[0] 
    
    print(f"\nFound {len(result.boxes)} objects!")
    
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        
        # This is the exact [x1, y1, x2, y2] array your teammate's 
        # Traditional CV script needs to crop the image!
        coords = box.xyxy[0].cpu().numpy().astype(int) 
        
        print(f" - Detected {class_name} ({confidence*100:.1f}% confidence) at {coords}")

if __name__ == "__main__":
    run_inference()