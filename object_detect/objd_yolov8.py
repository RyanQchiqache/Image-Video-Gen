from ultralytics import YOLO
import os


trained_model = YOLO("/home/ryqc/projects/python_projects/Image-Video-Gen/object_detect/runs/detect/train/weights/best.pt")


image_path = "/home/ryqc/projects/python_projects/Image-Video-Gen/object_detect/images/test_yolo.jpg"
inference_results = trained_model(image_path, 
                                  save=True, 
                                  project="runs/detect", 
                                  name="inference_test")

print(f"\nResults saved to: {inference_results[0].save_dir}")

boxes = inference_results[0].boxes

if boxes is not None:
    print(f"\nDetected {len(boxes)} objects:")
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = trained_model.names[cls]
        print(f"  {i+1}: {label} (confidence: {conf:.2f})")
