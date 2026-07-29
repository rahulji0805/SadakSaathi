"""
Splits images+labels into YOLO's expected folder structure:

dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt

Usage:
    python split_dataset.py --img_dir data/images --label_dir data/yolo_labels --out_dir dataset --val_ratio 0.15
"""
import argparse
import random
import shutil
from pathlib import Path

def main(img_dir, label_dir, out_dir, val_ratio, seed=42):
    img_dir, label_dir, out_dir = Path(img_dir), Path(label_dir), Path(out_dir)
    images = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    random.seed(seed)
    random.shuffle(images)

    n_val = int(len(images) * val_ratio)
    val_set = set(images[:n_val])

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept, missing_labels = 0, 0
    for img_path in images:
        split = "val" if img_path in val_set else "train"
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        shutil.copy(img_path, out_dir / "images" / split / img_path.name)
        shutil.copy(label_path, out_dir / "labels" / split / label_path.name)
        kept += 1

    print(f"Total images: {len(images)}, copied: {kept}, missing labels skipped: {missing_labels}")
    print(f"Train: {len(images) - n_val}, Val: {n_val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()
    main(args.img_dir, args.label_dir, args.out_dir, args.val_ratio)
