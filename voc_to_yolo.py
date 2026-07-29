"""
Convert Pascal VOC XML annotations to YOLO txt format.
Works for the Kaggle 'annotated-potholes-dataset' and similar VOC-style sets.

Usage:
    python voc_to_yolo.py --img_dir data/images --ann_dir data/annotations --out_dir data/yolo_labels
"""
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Single class: pothole -> id 0
CLASSES = ["pothole"]

def convert_box(size, box):
    """size=(w,h), box=(xmin,xmax,ymin,ymax) -> normalized (x_center,y_center,w,h)"""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_annotation(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls = obj.find("name").text.strip().lower()
        if cls not in CLASSES:
            cls = "pothole"  # fallback: treat any labeled damage as pothole for single-class model
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )
        bb = convert_box((w, h), b)
        lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in bb))

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

def main(img_dir, ann_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xml_files = list(Path(ann_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")
    converted, skipped = 0, 0
    for xml_path in xml_files:
        stem = xml_path.stem
        out_path = Path(out_dir) / f"{stem}.txt"
        try:
            convert_annotation(str(xml_path), str(out_path))
            converted += 1
        except Exception as e:
            print(f"Skipping {xml_path.name}: {e}")
            skipped += 1
    print(f"Converted: {converted}, Skipped: {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.img_dir, args.ann_dir, args.out_dir)
