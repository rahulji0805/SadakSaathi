"""
Export best.pt -> TFLite (int8 quantized for speed on-device).

Usage:
    python export_tflite.py
"""
from ultralytics import YOLO

def main():
    model = YOLO(r"X:\SadakSaathi\weights\best.pt")

    # int8 quantized TFLite -> smallest, fastest on mobile
    # Requires a small representative calibration set; ultralytics handles this
    # automatically from your training data.
    model.export(
        format="tflite",
        imgsz=640,
        int8=True,          # quantize for speed/size on-device
        data="pothole_data.yaml",  # needed for int8 calibration
    )
    print("Exported .tflite file — check runs/pothole/yolov8n_v1/weights/")

if __name__ == "__main__":
    main()
