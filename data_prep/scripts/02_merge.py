"""Step 2: Merge original train and validation annotations into a clean COCO format.

Rules & Steps:
1. Load both instances_train2017.json and instances_val2017.json.
2. Group annotations by file_name across train and val.
3. Drop exact repeats of (category_id, bbox) within each image (expected: 0).
4. Detect conflicting boxes (identical bbox with different category labels).
   Exclude those images entirely and record them in out/excluded_images.csv with the reason.
5. Write out/merged_all.json (COCO format):
   - One image entry per file_name
   - Keep 20 category ids/names
   - Add "class10": id % 10, "nature": id // 10 to each annotation
   - Store "orig_split": "train", "val", or "both" on each image entry
"""
import os
import sys
import json
import csv
from collections import defaultdict, Counter

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    rodd_dir = os.path.join(project_root, "RODD")
    out_dir = os.path.join(script_dir, "..", "out")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(rodd_dir, "annotations", "instances_train2017.json")
    val_path = os.path.join(rodd_dir, "annotations", "instances_val2017.json")

    print(f"Loading {train_path}...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Loading {val_path}...")
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    categories_20 = train_data["categories"]
    cat_id_to_name = {c["id"]: c["name"] for c in categories_20}

    # Map image_id to file_name and dimensions in each file
    train_imgs = {img["id"]: img for img in train_data["images"]}
    val_imgs = {img["id"]: img for img in val_data["images"]}

    # Collect all image records and annotations by file_name
    # Structure: file_name -> {"width": w, "height": h, "splits": set(), "annotations": list}
    file_info = {}

    def ingest_records(data, split_name, img_map):
        for ann in data["annotations"]:
            img_rec = img_map[ann["image_id"]]
            fn = img_rec["file_name"]
            w = img_rec["width"]
            h = img_rec["height"]

            if fn not in file_info:
                file_info[fn] = {
                    "width": w,
                    "height": h,
                    "splits": {split_name},
                    "annotations": []
                }
            else:
                info = file_info[fn]
                info["splits"].add(split_name)
                if info["width"] != w or info["height"] != h:
                    raise ValueError(f"Dimension mismatch for {fn}: ({info['width']}, {info['height']}) vs ({w}, {h})")

            # Store annotation copy along with source split
            ann_entry = {
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
                "source_split": split_name
            }
            file_info[fn]["annotations"].append(ann_entry)

    ingest_records(train_data, "train", train_imgs)
    ingest_records(val_data, "val", val_imgs)

    total_filenames = len(file_info)
    total_raw_boxes = sum(len(info["annotations"]) for info in file_info.values())
    both_count = sum(1 for info in file_info.values() if len(info["splits"]) > 1)
    train_only = sum(1 for info in file_info.values() if info["splits"] == {"train"})
    val_only = sum(1 for info in file_info.values() if info["splits"] == {"val"})

    print(f"Total unique filenames across train and val: {total_filenames}")
    print(f"  Train only: {train_only}")
    print(f"  Val only: {val_only}")
    print(f"  Both train and val: {both_count}")
    print(f"Total raw annotations: {total_raw_boxes}")

    # Check for exact repeats of (category_id, bbox) and conflicting boxes
    exact_repeats_dropped = 0
    excluded_images = []  # list of (file_name, reason, details)
    clean_file_info = {}

    for fn in sorted(file_info.keys()):
        info = file_info[fn]
        anns = info["annotations"]

        # Step 2: Drop exact repeats of (category_id, bbox)
        unique_cat_bbox = {}
        filtered_anns = []
        for ann in anns:
            key = (ann["category_id"], tuple(ann["bbox"]))
            if key in unique_cat_bbox:
                exact_repeats_dropped += 1
            else:
                unique_cat_bbox[key] = ann
                filtered_anns.append(ann)

        # Step 3: Check for conflicting boxes (same bbox, different category)
        bbox_map = defaultdict(list)
        for ann in filtered_anns:
            bbox_key = tuple(ann["bbox"])
            bbox_map[bbox_key].append(ann)

        conflicts = [banns for banns in bbox_map.values() if len(banns) > 1]
        if conflicts:
            conflict_details = []
            for banns in conflicts:
                cats = [f"{cat_id_to_name[a['category_id']]} (id={a['category_id']}, split={a['source_split']})" for a in banns]
                conflict_details.append(f"bbox {banns[0]['bbox']} labeled as [{', '.join(cats)}]")
            reason = f"Conflicting labels for identical bounding box: {'; '.join(conflict_details)}"
            excluded_images.append({
                "file_name": fn,
                "orig_split": "both" if len(info["splits"]) > 1 else list(info["splits"])[0],
                "num_boxes_in_image": len(filtered_anns),
                "reason": reason
            })
        else:
            clean_file_info[fn] = {
                "width": info["width"],
                "height": info["height"],
                "orig_split": "both" if len(info["splits"]) > 1 else list(info["splits"])[0],
                "annotations": filtered_anns
            }

    print(f"Exact repeats dropped: {exact_repeats_dropped}")
    print(f"Conflicting images detected and excluded: {len(excluded_images)}")
    excluded_boxes = sum(ex["num_boxes_in_image"] for ex in excluded_images)
    print(f"Total boxes in excluded images: {excluded_boxes}")

    # Write out/excluded_images.csv
    excluded_csv_path = os.path.join(out_dir, "excluded_images.csv")
    with open(excluded_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "orig_split", "num_boxes_in_image", "reason"])
        writer.writeheader()
        for row in excluded_images:
            writer.writerow(row)
    print(f"Saved {excluded_csv_path}")

    # Step 4: Build out/merged_all.json
    merged_images = []
    merged_annotations = []
    ann_id = 0

    for img_id, fn in enumerate(sorted(clean_file_info.keys())):
        info = clean_file_info[fn]
        merged_images.append({
            "id": img_id,
            "file_name": fn,
            "width": info["width"],
            "height": info["height"],
            "orig_split": info["orig_split"]
        })
        for ann in info["annotations"]:
            cid = ann["category_id"]
            merged_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": ann["bbox"],
                "area": ann["area"],
                "segmentation": ann["segmentation"],
                "iscrowd": ann["iscrowd"],
                "class10": cid % 10,
                "nature": cid // 10
            })
            ann_id += 1

    merged_data = {
        "categories": categories_20,
        "images": merged_images,
        "annotations": merged_annotations
    }

    merged_json_path = os.path.join(out_dir, "merged_all.json")
    with open(merged_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)
    print(f"Saved {merged_json_path}")
    print(f"Clean merged images: {len(merged_images)}")
    print(f"Clean merged annotations: {len(merged_annotations)}")

    # Statistics
    boxes_per_img = [len(clean_file_info[fn]["annotations"]) for fn in clean_file_info]
    box_hist = Counter(boxes_per_img)
    single_box_count = box_hist[1]
    multi_box_count = sum(count for n_boxes, count in box_hist.items() if n_boxes > 1)
    print(f"Images with 1 box: {single_box_count} ({single_box_count / len(boxes_per_img) * 100:.2f}%)")
    print(f"Images with >1 box: {multi_box_count} ({multi_box_count / len(boxes_per_img) * 100:.2f}%)")
    print(f"Max boxes in single image: {max(boxes_per_img)}")

    both_nature_count = 0
    for fn, info in clean_file_info.items():
        natures = {ann["category_id"] // 10 for ann in info["annotations"]}
        if 0 in natures and 1 in natures:
            both_nature_count += 1
    print(f"Images containing both real and reflected objects: {both_nature_count}")

if __name__ == "__main__":
    main()

