#!/usr/bin/env python3
"""Export RODD dataset to YOLO format.

Supports original benchmark split (orig) and cleaned splits (v1, v2)
in both 20-class and 10-class configurations.
"""

import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMG_TIMESTAMP_PATTERN = re.compile(r"IMG_\d{8}_\d{6}")


def validate_categories(train_cats, val_cats):
    """Validate that categories match expected 0..19 schema."""
    if train_cats != val_cats:
        raise ValueError("Categories in train and val annotation files do not match.")

    cat_map = {}
    for cat in train_cats:
        cid = cat["id"]
        cname = cat["name"]
        if not (0 <= cid <= 19):
            raise ValueError(f"Category ID {cid} is out of expected range 0..19.")
        if cid in cat_map:
            raise ValueError(f"Duplicate category ID {cid} found in categories.")
        cat_map[cid] = cname

    if len(cat_map) != 20:
        raise ValueError(f"Expected 20 categories, found {len(cat_map)}.")

    for i in range(10):
        name_real = cat_map[i]
        name_refl = cat_map[i + 10]
        if not name_real.endswith("or"):
            raise ValueError(f"Category {i} ('{name_real}') does not end with 'or'.")
        if not name_refl.endswith("mi"):
            raise ValueError(f"Category {i + 10} ('{name_refl}') does not end with 'mi'.")
        obj_real = name_real[:-2]
        obj_refl = name_refl[:-2]
        if obj_real != obj_refl:
            raise ValueError(
                f"Object name mismatch between ID {i} ('{obj_real}') and ID {i + 10} ('{obj_refl}')."
            )

    names_20 = [cat_map[i] for i in range(20)]
    names_10 = [cat_map[i][:-2] for i in range(10)]
    return names_20, names_10


def load_coco_annotations(train_ann_path, val_ann_path):
    """Load train and val annotations, returning categories, boxes, and image sizes."""
    with open(train_ann_path, "r", encoding="utf-8") as f:
        train_coco = json.load(f)
    with open(val_ann_path, "r", encoding="utf-8") as f:
        val_coco = json.load(f)

    names_20, names_10 = validate_categories(
        train_coco["categories"], val_coco["categories"]
    )

    sizes = {}
    boxes = {"train": defaultdict(list), "val": defaultdict(list)}

    for source, coco in [("train", train_coco), ("val", val_coco)]:
        img_by_id = {img["id"]: img for img in coco["images"]}
        for img in coco["images"]:
            fn = img["file_name"]
            sz = (img["width"], img["height"])
            if fn in sizes and sizes[fn] != sz:
                raise ValueError(
                    f"Dimensions mismatch for {fn}: {sizes[fn]} vs {sz}"
                )
            sizes[fn] = sz

        for ann in coco["annotations"]:
            img = img_by_id[ann["image_id"]]
            fn = img["file_name"]
            cid = ann["category_id"]
            bbox = ann["bbox"]  # [x, y, w, h]
            boxes[source][fn].append((cid, *bbox))

    return names_20, names_10, boxes, sizes


def build_split_assignments(boxes, excluded_csv_path, split_v1_path, split_v2_path):
    """Build image and box mapping for orig, v1, and v2 datasets."""
    excluded = set()
    with open(excluded_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            excluded.add(row["file_name"])

    datasets_splits = {
        "orig": {
            "train": {fn: list(boxes["train"][fn]) for fn in sorted(boxes["train"].keys())},
            "val": {fn: list(boxes["val"][fn]) for fn in sorted(boxes["val"].keys())},
        },
        "v1": {"train": {}, "val": {}, "test": {}},
        "v2": {"train": {}, "val": {}, "test": {}},
    }

    for v, split_path in [("v1", split_v1_path), ("v2", split_v2_path)]:
        with open(split_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row["file_name"]
                sp = row["split"]
                if fn in excluded:
                    raise ValueError(f"Excluded image {fn} appeared in {v} split.csv!")
                all_boxes = list(boxes["train"].get(fn, [])) + list(boxes["val"].get(fn, []))
                datasets_splits[v][sp][fn] = all_boxes

    return datasets_splits


def clear_existing_datasets(datasets_dir):
    """Delete existing datasets/orig, datasets/v1, datasets/v2 folders if present."""
    def handle_readonly(func, path, exc_info):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for name in ["orig", "v1", "v2"]:
        target = datasets_dir / name
        if target.exists():
            if sys.version_info >= (3, 12):
                def on_exc(func, path, exc):
                    import stat
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                shutil.rmtree(target, onexc=on_exc)
            else:
                shutil.rmtree(target, onerror=handle_readonly)


def write_data_yaml(target_path, label_set_folder, dataset_name, names):
    """Write machine-specific data.yaml with absolute posix path."""
    resolved_path = label_set_folder.resolve().as_posix()
    lines = [
        f"path: {json.dumps(resolved_path)}",
        "train: images/train",
    ]
    if dataset_name == "orig":
        lines.append("# val is the final scoring set for orig: do not use it for tuning.")
    lines.append("val: images/val")
    if dataset_name in ("v1", "v2"):
        lines.append("test: images/test")
    lines.append(f"nc: {len(names)}")
    lines.append("names:")
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")

    content = "\n".join(lines) + "\n"
    with open(target_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def compute_label_hash(label_set_dir):
    """Compute SHA-256 for all .txt label files in a label-set directory."""
    labels_dir = label_set_dir / "labels"
    label_files = []
    for txt_file in labels_dir.rglob("*.txt"):
        if txt_file.is_file():
            rel_key = txt_file.relative_to(label_set_dir).as_posix()
            label_files.append((rel_key, txt_file))

    label_files.sort(key=lambda x: x[0])

    hasher = hashlib.sha256()
    for rel_key, file_path in label_files:
        hasher.update(rel_key.encode("utf-8") + b"\n")
        with open(file_path, "rb") as f:
            hasher.update(f.read())

    return hasher.hexdigest()


def main():
    repo_root = Path(__file__).resolve().parent.parent
    rodd_dir = repo_root / "RODD"
    ann_dir = rodd_dir / "annotations"
    images_dir = rodd_dir / "images"
    splits_dir = repo_root / "data_prep" / "splits"
    datasets_dir = repo_root / "datasets"

    train_ann_path = ann_dir / "instances_train2017.json"
    val_ann_path = ann_dir / "instances_val2017.json"
    split_v1_path = splits_dir / "v1" / "split.csv"
    split_v2_path = splits_dir / "v2" / "split.csv"
    excluded_csv_path = splits_dir / "excluded_images.csv"

    print("Step 1 & 2: Loading annotations and verifying categories...")
    names_20, names_10, boxes, sizes = load_coco_annotations(train_ann_path, val_ann_path)

    print("Step 3: Building split assignments...")
    datasets_splits = build_split_assignments(
        boxes, excluded_csv_path, split_v1_path, split_v2_path
    )

    print("Step 4: Clearing existing datasets directories...")
    clear_existing_datasets(datasets_dir)

    print("Step 5 & 6: Writing label files and linking images...")
    links_count = 0
    copies_count = 0

    stats = []
    # Track 10-class repeats per dataset and split
    ten_class_repeats = defaultdict(lambda: defaultdict(int))

    for ds_name in ["orig", "v1", "v2"]:
        splits = ["train", "val"] if ds_name == "orig" else ["train", "val", "test"]

        # Write data.yaml for both class configurations
        for cls_name, cls_names in [("20class", names_20), ("10class", names_10)]:
            cls_dir = datasets_dir / ds_name / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            write_data_yaml(cls_dir / "data.yaml", cls_dir, ds_name, cls_names)

        for sp in splits:
            file_box_dict = datasets_splits[ds_name][sp]
            total_images = len(file_box_dict)
            total_label_lines = 0
            real_boxes_count = 0
            refl_boxes_count = 0

            # Pre-create image and label directories for both 20class and 10class
            for cls_name in ["20class", "10class"]:
                (datasets_dir / ds_name / cls_name / "images" / sp).mkdir(parents=True, exist_ok=True)
                (datasets_dir / ds_name / cls_name / "labels" / sp).mkdir(parents=True, exist_ok=True)

            img_dir_20 = datasets_dir / ds_name / "20class" / "images" / sp
            lbl_dir_20 = datasets_dir / ds_name / "20class" / "labels" / sp
            img_dir_10 = datasets_dir / ds_name / "10class" / "images" / sp
            lbl_dir_10 = datasets_dir / ds_name / "10class" / "labels" / sp

            for fn, raw_boxes in file_box_dict.items():
                if len(raw_boxes) == 0:
                    raise ValueError(f"Empty label file for {fn} in {ds_name}/{sp}")

                W, H = sizes[fn]
                stem = Path(fn).stem

                # Sort boxes by (category_id, x, y, w, h)
                sorted_boxes = sorted(
                    raw_boxes, key=lambda b: (b[0], b[1], b[2], b[3], b[4])
                )

                lines_20 = []
                lines_10 = []

                for cid, x, y, w, h in sorted_boxes:
                    if cid >= 10:
                        refl_boxes_count += 1
                        if not IMG_TIMESTAMP_PATTERN.search(fn):
                            raise ValueError(
                                f"Reflected box found in image without timestamp pattern: {fn}"
                            )
                    else:
                        real_boxes_count += 1

                    total_label_lines += 1

                    xc = (x + w / 2.0) / W
                    yc = (y + h / 2.0) / H
                    bw = w / float(W)
                    bh = h / float(H)

                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                        raise ValueError(
                            f"Coordinate out of [0, 1] bounds in {fn}: xc={xc}, yc={yc}, bw={bw}, bh={bh}"
                        )

                    l20 = f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                    l10 = f"{cid % 10} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

                    lines_20.append(l20)
                    lines_10.append(l10)

                # Validation checks
                if len(lines_20) != len(set(lines_20)):
                    raise ValueError(f"Repeated line found in 20class label file for {fn}")

                rep10 = len(lines_10) - len(set(lines_10))
                if rep10 > 0:
                    ten_class_repeats[ds_name][sp] += rep10

                # Write label files
                lbl_path_20 = lbl_dir_20 / f"{stem}.txt"
                lbl_path_10 = lbl_dir_10 / f"{stem}.txt"

                content_20 = "\n".join(lines_20) + "\n"
                content_10 = "\n".join(lines_10) + "\n"

                with open(lbl_path_20, "w", encoding="utf-8", newline="\n") as f20:
                    f20.write(content_20)
                with open(lbl_path_10, "w", encoding="utf-8", newline="\n") as f10:
                    f10.write(content_10)

                # Link images for both 20class and 10class
                src_img = images_dir / fn
                dst_img_20 = img_dir_20 / fn
                dst_img_10 = img_dir_10 / fn

                for dst_img in (dst_img_20, dst_img_10):
                    try:
                        os.link(src_img, dst_img)
                        links_count += 1
                    except OSError:
                        shutil.copy2(src_img, dst_img)
                        copies_count += 1

            stats.append(
                (ds_name, sp, total_images, total_label_lines, real_boxes_count, refl_boxes_count)
            )

    print(f"Image creation summary: {links_count} hard links, {copies_count} file copies.")

    print("\nStep 7: Dataset summary table")
    header = f"{'dataset':<8} {'split':<6} {'images':>8} {'label lines':>12} {'real boxes':>11} {'reflected boxes':>16}"
    print(header)
    print("-" * len(header))
    for ds_name, sp, img_cnt, lines_cnt, real_cnt, refl_cnt in stats:
        print(f"{ds_name:<8} {sp:<6} {img_cnt:>8} {lines_cnt:>12} {real_cnt:>11} {refl_cnt:>16}")

    print("\n10-class repeated lines summary:")
    for ds_name in ["orig", "v1", "v2"]:
        splits = ["train", "val"] if ds_name == "orig" else ["train", "val", "test"]
        line_parts = [f"{sp}: {ten_class_repeats[ds_name][sp]}" for sp in splits]
        print(f"  {ds_name}: {', '.join(line_parts)}")

    print("\nStep 8: Computing SHA-256 hashes of generated label sets...")
    hash_pairs = [
        ("orig", "10class"),
        ("orig", "20class"),
        ("v1", "10class"),
        ("v1", "20class"),
        ("v2", "10class"),
        ("v2", "20class"),
    ]

    hashes = {}
    for ds_name, cls_name in hash_pairs:
        label_set_dir = datasets_dir / ds_name / cls_name
        digest = compute_label_hash(label_set_dir)
        hashes[f"{ds_name}/{cls_name}"] = digest
        print(f"{ds_name}/{cls_name} {digest}")

    return stats, hashes


if __name__ == "__main__":
    main()
