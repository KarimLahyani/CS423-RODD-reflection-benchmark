"""Step 4: Create a clean, leakage-free 60/10/30 split of the RODD dataset.

Assignments are made by whole leakage groups to guarantee zero cross-split leakage.
Stratification:
- Greedy assignment: sort groups by size descending (with seed=0 shuffle for tie-breaking)
- Each group is assigned to the split whose per-(class10, nature) box counts are furthest below target,
  subject to split capacity limits (60/10/30 +/- 2%).

Outputs written to out/:
- train.json, val.json, test.json (COCO format, 20 categories)
- val_cls.json, test_cls.json (10 categories, id = class10)
- val_state.json, test_state.json (2 categories, id = nature)
- orig_val_per_image.json (original val file names, 1 record per image with all val boxes)
- split_stats.csv (per-split summary of images, boxes, category distribution)
"""
import os
import sys
import json
import csv
import random
from collections import defaultdict, Counter

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    rodd_dir = os.path.join(project_root, "RODD")
    out_dir = os.path.join(script_dir, "..", "out")
    os.makedirs(out_dir, exist_ok=True)

    merged_json_path = os.path.join(out_dir, "merged_all.json")
    groups_csv_path = os.path.join(out_dir, "groups.csv")
    orig_val_path = os.path.join(rodd_dir, "annotations", "instances_val2017.json")

    print(f"Loading merged dataset from {merged_json_path}...")
    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    categories_20 = merged_data["categories"]
    cat_names_20 = [c["name"] for c in sorted(categories_20, key=lambda x: x["id"])]
    cat_id_to_name = {c["id"]: c["name"] for c in categories_20}

    categories_10_cls = [
        {"id": 0, "name": "bowl"},
        {"id": 1, "name": "apple"},
        {"id": 2, "name": "mouse"},
        {"id": 3, "name": "keyboard"},
        {"id": 4, "name": "banana"},
        {"id": 5, "name": "carrot"},
        {"id": 6, "name": "cup"},
        {"id": 7, "name": "orange"},
        {"id": 8, "name": "chair"},
        {"id": 9, "name": "book"}
    ]

    categories_2_state = [
        {"id": 0, "name": "or"},
        {"id": 1, "name": "mi"}
    ]

    # Map file_name -> image_entry and file_name -> list of annotations
    img_by_fn = {img["file_name"]: img for img in merged_data["images"]}
    anns_by_fn = defaultdict(list)
    for ann in merged_data["annotations"]:
        img_id = ann["image_id"]
        fn = merged_data["images"][img_id]["file_name"]
        anns_by_fn[fn].append(ann)

    print(f"Loaded {len(img_by_fn)} clean images, {len(merged_data['annotations'])} clean annotations.")

    # Load groups
    print(f"Loading groups from {groups_csv_path}...")
    fn_to_group = {}
    with open(groups_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            fn_to_group[row[0]] = int(row[1])

    # Group annotated images only
    group_to_annotated_fns = defaultdict(list)
    for fn in img_by_fn:
        gid = fn_to_group[fn]
        group_to_annotated_fns[gid].append(fn)

    print(f"Annotated images span {len(group_to_annotated_fns)} leakage groups.")

    # Compute group box counts across 20 categories
    total_imgs = len(img_by_fn)
    total_cat_counts = Counter()
    group_cat_counts = {}

    for gid, fns in group_to_annotated_fns.items():
        counts = Counter()
        for fn in fns:
            for ann in anns_by_fn[fn]:
                cid = ann["category_id"]
                counts[cid] += 1
                total_cat_counts[cid] += 1
        group_cat_counts[gid] = counts

    # Targets
    target_ratios = {"train": 0.60, "val": 0.10, "test": 0.30}
    target_imgs = {s: target_ratios[s] * total_imgs for s in ["train", "val", "test"]}
    target_cats = {
        s: {c: target_ratios[s] * total_cat_counts[c] for c in range(20)}
        for s in ["train", "val", "test"]
    }

    # Deterministic sorting: seed-0 shuffle for ties, then size descending
    groups_list = list(group_to_annotated_fns.items())
    # Sort by group ID first to ensure reproducible starting order regardless of dict hash order
    groups_list.sort(key=lambda x: x[0])
    rng = random.Random(0)
    rng.shuffle(groups_list)
    groups_list.sort(key=lambda item: len(item[1]), reverse=True)

    print(f"Greedy assignment of {len(groups_list)} groups...")
    split_imgs = {"train": 0, "val": 0, "test": 0}
    split_cats = {s: Counter() for s in ["train", "val", "test"]}
    split_groups = {"train": [], "val": [], "test": []}
    split_fns = {"train": [], "val": [], "test": []}

    for gid, fns in groups_list:
        g_len = len(fns)
        g_counts = group_cat_counts[gid]

        # Valid candidate splits: enforce tolerance limits (target +/- 2%)
        # For val (10%), upper bound is 12%
        # For train (60%), upper bound is 62%
        # For test (30%), upper bound is 32%
        upper_bounds = {
            "train": 0.62 * total_imgs,
            "val": 0.12 * total_imgs,
            "test": 0.32 * total_imgs
        }

        # Check candidate splits where adding this group does not exceed upper bound
        valid = [s for s in ["train", "val", "test"] if split_imgs[s] + g_len <= upper_bounds[s]]
        if not valid:
            # Fallback: splits that are below their base target
            valid = [s for s in ["train", "val", "test"] if split_imgs[s] < target_imgs[s]]
        if not valid:
            # If all are at or above target, pick any
            valid = ["train", "val", "test"]

        # Score candidate splits: choose the one furthest below category targets
        best_s = None
        best_score = -1e18

        for s in valid:
            # Absolute deficit across categories present in this group
            cat_deficit = sum((target_cats[s][c] - split_cats[s][c]) * g_counts[c] for c in g_counts)
            # Image count deficit as tie-breaker
            img_deficit = (target_imgs[s] - split_imgs[s]) / target_imgs[s]
            score = cat_deficit + 0.01 * img_deficit

            if score > best_score:
                best_score = score
                best_s = s

        split_imgs[best_s] += g_len
        for c, count in g_counts.items():
            split_cats[best_s][c] += count
        split_groups[best_s].append(gid)
        split_fns[best_s].extend(fns)

    print("\n--- Split Image Fractions ---")
    for s in ["train", "val", "test"]:
        cnt = split_imgs[s]
        pct = cnt / total_imgs * 100
        print(f"  {s:5s}: {cnt:5d} images ({pct:.2f}%), {len(split_groups[s]):4d} groups, min class count: {min(split_cats[s][c] for c in range(20))}")

    # Build COCO datasets for each split
    def build_coco(fns, cat_list, cat_id_func):
        imgs = []
        anns = []
        ann_counter = 0

        # Sort file names for determinism
        for new_img_id, fn in enumerate(sorted(fns)):
            orig_img = img_by_fn[fn]
            imgs.append({
                "id": new_img_id,
                "file_name": fn,
                "width": orig_img["width"],
                "height": orig_img["height"],
                "orig_split": orig_img["orig_split"]
            })
            for ann in anns_by_fn[fn]:
                cid = ann["category_id"]
                mapped_cid = cat_id_func(cid)
                anns.append({
                    "id": ann_counter,
                    "image_id": new_img_id,
                    "category_id": mapped_cid,
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "segmentation": ann["segmentation"],
                    "iscrowd": ann["iscrowd"],
                    "class10": cid % 10,
                    "nature": cid // 10
                })
                ann_counter += 1

        return {
            "categories": cat_list,
            "images": imgs,
            "annotations": anns
        }

    # Save train.json, val.json, test.json (20 categories)
    for s in ["train", "val", "test"]:
        coco_obj = build_coco(split_fns[s], categories_20, lambda c: c)
        out_p = os.path.join(out_dir, f"{s}.json")
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(coco_obj, f, indent=2)
        print(f"Saved {out_p}: {len(coco_obj['images'])} images, {len(coco_obj['annotations'])} boxes")

    # Save val_cls.json, test_cls.json (10 categories, id = class10)
    for s in ["val", "test"]:
        coco_cls = build_coco(split_fns[s], categories_10_cls, lambda c: c % 10)
        out_p = os.path.join(out_dir, f"{s}_cls.json")
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(coco_cls, f, indent=2)
        print(f"Saved {out_p}: {len(coco_cls['images'])} images, {len(coco_cls['annotations'])} boxes")

    # Save val_state.json, test_state.json (2 categories, id = nature)
    for s in ["val", "test"]:
        coco_state = build_coco(split_fns[s], categories_2_state, lambda c: c // 10)
        out_p = os.path.join(out_dir, f"{s}_state.json")
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(coco_state, f, indent=2)
        print(f"Saved {out_p}: {len(coco_state['images'])} images, {len(coco_state['annotations'])} boxes")

    # -------------------------------------------------------------
    # Step 4: Write out/orig_val_per_image.json
    # Original validation filenames grouped per image (fixes scoring bug without changing split)
    # -------------------------------------------------------------
    print(f"Building {os.path.join(out_dir, 'orig_val_per_image.json')} from {orig_val_path}...")
    with open(orig_val_path, "r", encoding="utf-8") as f:
        orig_val_raw = json.load(f)

    orig_val_img_map = {img["id"]: img for img in orig_val_raw["images"]}
    orig_val_by_fn = defaultdict(list)
    for ann in orig_val_raw["annotations"]:
        fn = orig_val_img_map[ann["image_id"]]["file_name"]
        orig_val_by_fn[fn].append(ann)

    orig_val_imgs = []
    orig_val_anns = []
    orig_ann_id = 0

    for new_img_id, fn in enumerate(sorted(orig_val_by_fn.keys())):
        # Find width and height from original records
        # All records for the same file have identical width and height
        sample_rec = next(img for img in orig_val_raw["images"] if img["file_name"] == fn)
        orig_val_imgs.append({
            "id": new_img_id,
            "file_name": fn,
            "width": sample_rec["width"],
            "height": sample_rec["height"],
            "orig_split": "val"
        })
        for ann in orig_val_by_fn[fn]:
            cid = ann["category_id"]
            orig_val_anns.append({
                "id": orig_ann_id,
                "image_id": new_img_id,
                "category_id": cid,
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0),
                "class10": cid % 10,
                "nature": cid // 10
            })
            orig_ann_id += 1

    orig_val_per_image_obj = {
        "categories": categories_20,
        "images": orig_val_imgs,
        "annotations": orig_val_anns
    }

    orig_val_out_p = os.path.join(out_dir, "orig_val_per_image.json")
    with open(orig_val_out_p, "w", encoding="utf-8") as f:
        json.dump(orig_val_per_image_obj, f, indent=2)
    print(f"Saved {orig_val_out_p}: {len(orig_val_imgs)} images, {len(orig_val_anns)} boxes")

    # -------------------------------------------------------------
    # Step 5: Save out/split_stats.csv
    # -------------------------------------------------------------
    stats_csv_path = os.path.join(out_dir, "split_stats.csv")
    fieldnames = [
        "split", "images", "fraction_of_total_images", "boxes",
        "images_both_real_reflected", "fraction_images_gt_1_box"
    ] + cat_names_20

    stats_rows = []
    for s in ["train", "val", "test"]:
        fns = split_fns[s]
        n_imgs = len(fns)
        n_boxes = sum(len(anns_by_fn[fn]) for fn in fns)
        multi_box_imgs = sum(1 for fn in fns if len(anns_by_fn[fn]) > 1)
        both_nature_imgs = 0
        for fn in fns:
            natures = {ann["category_id"] // 10 for ann in anns_by_fn[fn]}
            if 0 in natures and 1 in natures:
                both_nature_imgs += 1

        row = {
            "split": s,
            "images": n_imgs,
            "fraction_of_total_images": f"{n_imgs / total_imgs:.4f}",
            "boxes": n_boxes,
            "images_both_real_reflected": both_nature_imgs,
            "fraction_images_gt_1_box": f"{multi_box_imgs / n_imgs:.4f}"
        }
        for cid in range(20):
            cname = cat_names_20[cid]
            row[cname] = split_cats[s][cid]
        stats_rows.append(row)

    # Add total row
    tot_imgs = total_imgs
    tot_boxes = sum(len(anns_by_fn[fn]) for fn in img_by_fn)
    tot_multi_box = sum(1 for fn in img_by_fn if len(anns_by_fn[fn]) > 1)
    tot_both_nature = sum(1 for fn in img_by_fn if {a["category_id"]//10 for a in anns_by_fn[fn]} == {0, 1})
    total_row = {
        "split": "total",
        "images": tot_imgs,
        "fraction_of_total_images": "1.0000",
        "boxes": tot_boxes,
        "images_both_real_reflected": tot_both_nature,
        "fraction_images_gt_1_box": f"{tot_multi_box / tot_imgs:.4f}"
    }
    for cid in range(20):
        cname = cat_names_20[cid]
        total_row[cname] = total_cat_counts[cid]
    stats_rows.append(total_row)

    with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in stats_rows:
            writer.writerow(r)
    print(f"Saved {stats_csv_path}")

if __name__ == "__main__":
    main()

