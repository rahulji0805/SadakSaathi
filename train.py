"""
Train YOLOv8n on pothole dataset.

Usage:
    python train.py
"""
from ultralytics import YOLO

def main():
    # Start from pretrained nano weights -> fast, edge-friendly, matches your TFLite export plan
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="pothole_data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,           # lower to 8 if you hit VRAM limits
        patience=20,        # early stop if val doesn't improve
        device=0,           # GPU 0; use "cpu" if no GPU detected
        project="runs/pothole",
        name="yolov8n_v1",
        optimizer="AdamW",
        lr0=0.001,
        augment=True,       # built-in mosaic/flip/hsv augmentation
        val=True,
        plots=True,
    )

    # Validate
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")

if __name__ == "__main__":
    main()
