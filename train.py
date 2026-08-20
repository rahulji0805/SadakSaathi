"""
Train YOLOv8n on pothole dataset.

Usage:
    python train.py
"""
from pyexpat import model

from ultralytics import YOLO

def main():
    # Start from pretrained nano weights -> fast, edge-friendly, matches your TFLite export plan
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="pothole_data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        project="runs/pothole",
        name="yolov8n_pothole_only_v1",   # naya naam, purana crashed run overwrite na ho
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        val=True,
        plots=True,
        workers=4,      # 8 se 4 kar diya
    )

    # Validate
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")

if __name__ == "__main__":
    main()
