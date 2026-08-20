"""
Merge RDD2022 (India/train) converted YOLO labels into the existing
dataset/ folder, alongside the old Kaggle pothole data. Does NOT delete
or overwrite anything already in dataset/.

Splits RDD2022 into train/val (default 90/10) and copies files with a
prefix so filenames never collide with the existing Kaggle images.

Usage:
    python merge_datasets.py --src_img India/train/images --src_lbl India/train/labels --dst dataset --val_ratio 0.1
"""
import argparse
import random
import shutil
from pathlib import Path

def main(src_img, src_lbl, dst, val_ratio, prefix):
    src_img, src_lbl, dst = Path(src_img), Path(src_lbl), Path(dst)

    label_files = sorted(src_lbl.glob("*.txt"))
    print(f"Found {len(label_files)} label files in {src_lbl}")

    random.seed(42)
    random.shuffle(label_files)
    n_val = int(len(label_files) * val_ratio)
    val_set = set(f.stem for f in label_files[:n_val])

    counts = {"train": 0, "val": 0, "missing_image": 0}

    for lbl_path in label_files:
        stem = lbl_path.stem
        split = "val" if stem in val_set else "train"

        # find matching image (try common extensions)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = src_img / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            counts["missing_image"] += 1
            continue

        new_stem = f"{prefix}_{stem}"
        dst_img_dir = dst / "images" / split
        dst_lbl_dir = dst / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, dst_img_dir / f"{new_stem}{img_path.suffix}")
        shutil.copy2(lbl_path, dst_lbl_dir / f"{new_stem}.txt")
        counts[split] += 1

    print(f"Copied -> train: {counts['train']}, val: {counts['val']}")
    if counts["missing_image"]:
        print(f"Warning: {counts['missing_image']} labels had no matching image, skipped")
    print("Old dataset/ content untouched. Delete train.cache/val.cache before next training run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_img", required=True)
    parser.add_argument("--src_lbl", required=True)
    parser.add_argument("--dst", required=True, help="existing dataset/ root (has images/ and labels/)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--prefix", default="rdd2022", help="prefix for copied filenames to avoid collisions")
    args = parser.parse_args()
    main(args.src_img, args.src_lbl, args.dst, args.val_ratio, args.prefix)
