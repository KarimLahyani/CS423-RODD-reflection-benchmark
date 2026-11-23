#!/usr/bin/env python3
"""
evaluation/make_gt.py

Builds COCO ground-truth JSON files from:
  RODD/annotations/instances_{train,val}2017.json
  data_prep/splits/{v1,v2}/split.csv

Output directory: datasets/gt/<name>.json
Always 20-class ids.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

EXPECTED_COUNTS = {
    "orig_val": {"images": 7570, "boxes": 9744},
    "orig_val_union": {"images": 7570, "boxes": 17801},
    "orig_val_records": {"images": 9744, "boxes": 9744},
    "v1_val": {"images": 1932, "boxes": 3259},
    "v1_test": {"images": 5975, "boxes": 9730},
    "v2_val": {"images": 2205, "boxes": 4511},
    "v2_test": {"images": 5797, "boxes": 8589},
}


def load_source_data(repo_root: Path):
    train_path = repo_root / "RODD" / "annotations" / "instances_train2017.json"
    val_path = repo_root / "RODD" / "annotations" / "instances_val2017.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Categories are identical and 20 classes
    categories = sorted(train_data["categories"], key=lambda c: c["id"])

    # Map image id to file_name and dimensions
    train_img_map = {img["id"]: img for img in train_data["images"]}
    val_img_map = {img["id"]: img for img in val_data["images"]}

    # Dimensions lookup by file_name
    img_dims = {}
    for img in train_data["images"]:
        img_dims[img["file_name"]] = (img["width"], img["height"])
    for img in val_data["images"]:
        img_dims[img["file_name"]] = (img["width"], img["height"])

    # Group annotations by file_name: list of (category_id, [x, y, w, h])
    train_boxes_by_fn = {}
    for ann in train_data["annotations"]:
        fn = train_img_map[ann["image_id"]]["file_name"]
        train_boxes_by_fn.setdefault(fn, []).append((ann["category_id"], ann["bbox"]))

    val_boxes_by_fn = {}
    for ann in val_data["annotations"]:
        fn = val_img_map[ann["image_id"]]["file_name"]
        val_boxes_by_fn.setdefault(fn, []).append((ann["category_id"], ann["bbox"]))

    return {
        "categories": categories,
        "img_dims": img_dims,
        "train_boxes": train_boxes_by_fn,
        "val_boxes": val_boxes_by_fn,
        "raw_val_data": val_data,
    }


def load_split_csv(split_csv_path: Path):
    split_map = {}
    with open(split_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                split_map[row[0]] = row[1]
    return split_map


def build_coco_dataset(file_names, boxes_getter, img_dims, categories):
    """
    Build standard COCO dataset:
    - Image ids are 0..n-1 over file names sorted as Python strings.
    - Annotation ids are 1..m in order of (image_id, category_id, x, y, w, h).
    - Each annotation has id, image_id, category_id, bbox, area = w*h, iscrowd = 0.
    - Each image has id, file_name, width, height.
    """
    sorted_fns = sorted(file_names)
    images = []
    annotations_raw = []

    for img_id, fn in enumerate(sorted_fns):
        w, h = img_dims[fn]
        images.append({
            "file_name": fn,
            "height": h,
            "id": img_id,
            "width": w,
        })
        boxes = boxes_getter(fn)
        for cat_id, bbox in boxes:
            annotations_raw.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": list(bbox),
                "area": int(bbox[2] * bbox[3]),
                "iscrowd": 0,
            })

    # Sort annotations by (image_id, category_id, x, y, w, h)
    annotations_raw.sort(
        key=lambda a: (
            a["image_id"],
            a["category_id"],
            a["bbox"][0],
            a["bbox"][1],
            a["bbox"][2],
            a["bbox"][3],
        )
    )

    # Assign 1..m annotation ids
    annotations = []
    for ann_id, ann in enumerate(annotations_raw, 1):
        ann_copy = dict(ann)
        ann_copy["id"] = ann_id
        annotations.append(ann_copy)

    return {
        "annotations": annotations,
        "categories": categories,
        "images": images,
    }


def build_orig_val_records(raw_val_data, categories):
    """
    Original val records with unchanged ids (0..9743).
    Each annotation has id, image_id, category_id, bbox, area = w*h, iscrowd = 0.
    Each image has id, file_name, width, height.
    """
    images = []
    for img in raw_val_data["images"]:
        images.append({
            "file_name": img["file_name"],
            "height": img["height"],
            "id": img["id"],
            "width": img["width"],
        })

    annotations = []
    for ann in raw_val_data["annotations"]:
        bbox = ann["bbox"]
        annotations.append({
            "area": int(bbox[2] * bbox[3]),
            "bbox": list(bbox),
            "category_id": ann["category_id"],
            "id": ann["id"],
            "image_id": ann["image_id"],
            "iscrowd": 0,
        })

    return {
        "annotations": annotations,
        "categories": categories,
        "images": images,
    }


def get_merged_boxes(fn, val_boxes, train_boxes):
    """
    Deduplicated set of (category_id, tuple(bbox)) from val + train.
    """
    boxes_set = set()
    if fn in val_boxes:
        for cat_id, bbox in val_boxes[fn]:
            boxes_set.add((cat_id, tuple(bbox)))
    if fn in train_boxes:
        for cat_id, bbox in train_boxes[fn]:
            boxes_set.add((cat_id, tuple(bbox)))
    return sorted(list(boxes_set), key=lambda x: (x[0], x[1]))


def generate_dataset(name: str, src_data: dict, repo_root: Path):
    categories = src_data["categories"]
    img_dims = src_data["img_dims"]
    train_boxes = src_data["train_boxes"]
    val_boxes = src_data["val_boxes"]

    if name == "orig_val":
        # 7,570 unique val file names, val-file boxes only
        fns = list(val_boxes.keys())
        def getter(fn):
            return val_boxes.get(fn, [])
        return build_coco_dataset(fns, getter, img_dims, categories)

    elif name == "orig_val_union":
        # 7,570 unique val file names, set of (category_id, bbox) from val + train
        fns = list(val_boxes.keys())
        def getter(fn):
            return get_merged_boxes(fn, val_boxes, train_boxes)
        return build_coco_dataset(fns, getter, img_dims, categories)

    elif name == "orig_val_records":
        # original val records, unchanged ids
        return build_orig_val_records(src_data["raw_val_data"], categories)

    elif name in ("v1_val", "v1_test"):
        split_map = load_split_csv(repo_root / "data_prep" / "splits" / "v1" / "split.csv")
        target_split = "val" if name == "v1_val" else "test"
        fns = [fn for fn, s in split_map.items() if s == target_split]
        def getter(fn):
            return get_merged_boxes(fn, val_boxes, train_boxes)
        return build_coco_dataset(fns, getter, img_dims, categories)

    elif name in ("v2_val", "v2_test"):
        split_map = load_split_csv(repo_root / "data_prep" / "splits" / "v2" / "split.csv")
        target_split = "val" if name == "v2_val" else "test"
        fns = [fn for fn, s in split_map.items() if s == target_split]
        def getter(fn):
            return get_merged_boxes(fn, val_boxes, train_boxes)
        return build_coco_dataset(fns, getter, img_dims, categories)

    else:
        raise ValueError(f"Unknown dataset name: {name}")


def write_json(path, ds):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ds, f, sort_keys=True, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build COCO ground-truth JSON files")
    parser.add_argument("--all", action="store_true", help="Build all 7 ground-truth files")
    parser.add_argument("--names", nargs="*", default=[], help="Specific names to build")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    target_names = list(EXPECTED_COUNTS.keys()) if (args.all or not args.names) else args.names

    out_dir = repo_root / "datasets" / "gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_data = load_source_data(repo_root)

    print(f"Building {len(target_names)} ground-truth file(s)...")

    for name in target_names:
        if name not in EXPECTED_COUNTS:
            raise ValueError(f"Unknown dataset name: {name}")

        ds = generate_dataset(name, src_data, repo_root)

        n_images = len(ds["images"])
        n_boxes = len(ds["annotations"])
        exp = EXPECTED_COUNTS[name]

        if n_images != exp["images"] or n_boxes != exp["boxes"]:
            raise ValueError(
                f"Count mismatch for {name}: got {n_images} images, {n_boxes} boxes; "
                f"expected {exp['images']} images, {exp['boxes']} boxes"
            )

        out_path = out_dir / f"{name}.json"
        write_json(out_path, ds)

        print(f"  [OK] {name:18s} -> {n_images:5d} images, {n_boxes:5d} boxes -> {out_path.name}")

    print("All requested ground-truth files built successfully.")


if __name__ == "__main__":
    main()
