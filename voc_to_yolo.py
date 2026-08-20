"""
Convert Pascal VOC XML annotations to YOLO txt format.
Pothole-only version (single class).

Usage (RDD2022 India):
    python voc_to_yolo.py --img_dir India/train/images --ann_dir India/train/annotations/xmls --out_dir India/train/labels

Usage (old Kaggle pothole set, unchanged behavior):
    python voc_to_yolo.py --img_dir data/images --ann_dir data/annotations --out_dir data/yolo_labels
"""
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Single class only.
CLASSES = ["pothole"]

# Maps raw XML <name> values (any casing) to entries in CLASSES above.
# Anything not listed here (cracks, etc.) is skipped.
RAW_TO_CLASS = {
    "d40": "pothole",
    "d43": "pothole",
    "d44": "pothole",
    "pothole": "pothole",  # Kaggle set already uses this literal name
}

def convert_box(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
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
    skipped_classes = set()
    for obj in root.findall("object"):
        raw = obj.find("name").text.strip().lower()
        cls = RAW_TO_CLASS.get(raw)
        if cls is None:
            skipped_classes.add(raw)
            continue  # crack / unwanted class -> skip this object only
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

    wrote_file = False
    if lines:
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        wrote_file = True
    # if no pothole objects survived, DO NOT write an empty label file —
    # main() will use this to skip copying the matching image too.
    return skipped_classes, wrote_file

def main(img_dir, ann_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xml_files = list(Path(ann_dir).glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")
    print(f"Class mapping in use: {CLASSES}")
    converted, skipped, dropped_no_pothole = 0, 0, 0
    all_skipped_classes = set()
    dropped_stems = []
    for xml_path in xml_files:
        stem = xml_path.stem
        out_path = Path(out_dir) / f"{stem}.txt"
        try:
            skipped_cls, wrote_file = convert_annotation(str(xml_path), str(out_path))
            all_skipped_classes |= skipped_cls
            if not wrote_file:
                dropped_no_pothole += 1
                dropped_stems.append(stem)
            else:
                converted += 1
        except Exception as e:
            print(f"Skipping {xml_path.name}: {e}")
            skipped += 1

    print(f"Converted (has pothole): {converted}, Failed: {skipped}, Dropped (no pothole): {dropped_no_pothole}")
    if all_skipped_classes:
        print(f"Ignored raw class names (not pothole): {sorted(all_skipped_classes)}")

    # write list of dropped stems so a follow-up step can remove matching images
    if dropped_stems:
        dropped_list_path = Path(out_dir).parent / "dropped_no_pothole.txt"
        with open(dropped_list_path, "w") as f:
            f.write("\n".join(dropped_stems))
        print(f"Wrote {len(dropped_stems)} dropped stems to {dropped_list_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.img_dir, args.ann_dir, args.out_dir)