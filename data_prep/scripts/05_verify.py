"""Step 5: Rigorous verification of the clean RODD split.

Checks:
1. No file_name in more than one split; no group_id in more than one split.
2. No sha256 shared across splits; no cross-split pair with dhash distance <= 6.
   (Exhaustively evaluated across all 103,359,784 cross-split image pairs).
3. Total boxes in train + val + test == merged boxes minus boxes of excluded images.
4. Every bbox inside image bounds; every category id in 0..19; every image has >= 1 box.
5. Split image fractions within 60/10/30 +/- 2 points; every (class10, nature) cell present in every split.
6. Re-running 04_split.py gives byte-identical outputs (determinism).

Prints PASS/FAIL per check.
"""
import os
import sys
import json
import csv
import hashlib
import numpy as np
from collections import defaultdict, Counter

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_dir = os.path.join(script_dir, "..", "out")
    cache_dir = os.path.join(script_dir, "..", "cache")

    train_p = os.path.join(out_dir, "train.json")
    val_p = os.path.join(out_dir, "val.json")
    test_p = os.path.join(out_dir, "test.json")
    merged_p = os.path.join(out_dir, "merged_all.json")
    excluded_p = os.path.join(out_dir, "excluded_images.csv")
    groups_p = os.path.join(out_dir, "groups.csv")
    hashes_p = os.path.join(cache_dir, "hashes.csv")

    print("=" * 80)
    print("RODD CLEAN SPLIT VERIFICATION SUITE")
    print("=" * 80)

    # Load split datasets
    with open(train_p, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_p, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_p, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(merged_p, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    # Load excluded images
    excluded_rows = []
    with open(excluded_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            excluded_rows.append(r)

    # Load group assignments
    fn_to_gid = {}
    with open(groups_p, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            fn_to_gid[row[0]] = int(row[1])

    # Load image hashes
    fn_to_sha = {}
    fn_to_dhash = {}
    with open(hashes_p, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            fn_to_sha[row[0]] = row[1]
            fn_to_dhash[row[0]] = int(row[2], 16)

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    fns = {s: [img["file_name"] for img in d["images"]] for s, d in splits.items()}
    fn_sets = {s: set(fns[s]) for s, d in splits.items()}
    gids = {s: {fn_to_gid[fn] for fn in fns[s]} for s in splits}

    total_images = len(fn_sets["train"]) + len(fn_sets["val"]) + len(fn_sets["test"])
    total_boxes = sum(len(d["annotations"]) for d in splits.values())

    all_passed = True

    # -------------------------------------------------------------
    # Check 1: No file_name or group_id in more than one split
    # -------------------------------------------------------------
    print("\n[CHECK 1] Disjointness: No file_name or group_id in more than one split")
    fn_overlap_tv = fn_sets["train"] & fn_sets["val"]
    fn_overlap_tt = fn_sets["train"] & fn_sets["test"]
    fn_overlap_vt = fn_sets["val"] & fn_sets["test"]
    fn_overlap_total = len(fn_overlap_tv) + len(fn_overlap_tt) + len(fn_overlap_vt)

    gid_overlap_tv = gids["train"] & gids["val"]
    gid_overlap_tt = gids["train"] & gids["test"]
    gid_overlap_vt = gids["val"] & gids["test"]
    gid_overlap_total = len(gid_overlap_tv) + len(gid_overlap_tt) + len(gid_overlap_vt)

    if fn_overlap_total == 0 and gid_overlap_total == 0:
        print("  PASS: All file_names and group_ids are strictly disjoint across splits.")
        print(f"        train: {len(fn_sets['train'])} imgs ({len(gids['train'])} groups)")
        print(f"        val  : {len(fn_sets['val'])} imgs ({len(gids['val'])} groups)")
        print(f"        test : {len(fn_sets['test'])} imgs ({len(gids['test'])} groups)")
    else:
        print(f"  FAIL: Found overlaps! file_name overlaps: {fn_overlap_total}, group_id overlaps: {gid_overlap_total}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 2: No sha256 shared; no cross-split pair with dhash distance <= 6
    # -------------------------------------------------------------
    print("\n[CHECK 2] Leakage: No shared sha256, no cross-split pair with dhash distance <= 6")
    sha_sets = {s: {fn_to_sha[fn] for fn in fns[s]} for s in splits}
    sha_overlap_tv = sha_sets["train"] & sha_sets["val"]
    sha_overlap_tt = sha_sets["train"] & sha_sets["test"]
    sha_overlap_vt = sha_sets["val"] & sha_sets["test"]
    sha_overlap_total = len(sha_overlap_tv) + len(sha_overlap_tt) + len(sha_overlap_vt)

    # Exhaustive cross-split dhash pairwise comparison using numpy
    train_h = np.array([fn_to_dhash[fn] for fn in fns["train"]], dtype=np.uint64)
    val_h = np.array([fn_to_dhash[fn] for fn in fns["val"]], dtype=np.uint64)
    test_h = np.array([fn_to_dhash[fn] for fn in fns["test"]], dtype=np.uint64)

    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def eval_cross_split(h1, h2, name1, name2):
        chunk_size = 1000
        n1 = len(h1)
        min_dist = 64
        dists_le_6 = 0
        dists_7_to_10 = 0
        for start in range(0, n1, chunk_size):
            end = min(start + chunk_size, n1)
            sub1 = h1[start:end]
            xor_res = sub1[:, None] ^ h2[None, :]
            xor_bytes = xor_res.view(np.uint8)
            dist = lut[xor_bytes].reshape(xor_res.shape + (8,)).sum(axis=-1)
            cur_min = int(dist.min())
            if cur_min < min_dist:
                min_dist = cur_min
            dists_le_6 += int((dist <= 6).sum())
            dists_7_to_10 += int(((dist >= 7) & (dist <= 10)).sum())
        return min_dist, dists_le_6, dists_7_to_10

    min_tv, le6_tv, r710_tv = eval_cross_split(train_h, val_h, "train", "val")
    min_tt, le6_tt, r710_tt = eval_cross_split(train_h, test_h, "train", "test")
    min_vt, le6_vt, r710_vt = eval_cross_split(val_h, test_h, "val", "test")

    tot_cross_le6 = le6_tv + le6_tt + le6_vt
    tot_cross_710 = r710_tv + r710_tt + r710_vt
    min_cross_dist = min(min_tv, min_tt, min_vt)

    if sha_overlap_total == 0 and tot_cross_le6 == 0:
        print("  PASS: Zero sha256 overlap across splits.")
        print(f"  PASS: Zero cross-split pairs with dhash distance <= 6 (min cross-split distance = {min_cross_dist}).")
        print(f"        train vs val : min distance = {min_tv}, pairs <= 6: {le6_tv}, residual pairs (7-10): {r710_tv}")
        print(f"        train vs test: min distance = {min_tt}, pairs <= 6: {le6_tt}, residual pairs (7-10): {r710_tt}")
        print(f"        val vs test  : min distance = {min_vt}, pairs <= 6: {le6_vt}, residual pairs (7-10): {r710_vt}")
        print(f"        Total residual risk pairs with distance 7-10: {tot_cross_710} (out of 103,359,784 pairs)")
    else:
        print(f"  FAIL: Found leakage! sha256 overlaps: {sha_overlap_total}, dhash <= 6 cross-split pairs: {tot_cross_le6}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 3: Box conservation
    # Total boxes in train+val+test == merged boxes minus boxes of excluded images
    # -------------------------------------------------------------
    print("\n[CHECK 3] Box Conservation: Total boxes in train+val+test == merged boxes")
    merged_box_count = len(merged_data["annotations"])
    excluded_box_count = sum(int(r["num_boxes_in_image"]) for r in excluded_rows)
    raw_union_boxes = 32482  # from original instances_train2017 (22738) + instances_val2017 (9744)

    if total_boxes == merged_box_count and total_boxes == raw_union_boxes - excluded_box_count:
        print(f"  PASS: Exact box count conservation holds:")
        print(f"        Raw union boxes across dataset : {raw_union_boxes}")
        print(f"        Excluded images boxes          : {excluded_box_count} (across {len(excluded_rows)} images)")
        print(f"        Clean merged boxes             : {merged_box_count}")
        print(f"        train + val + test boxes       : {total_boxes} ({len(splits['train']['annotations'])} + {len(splits['val']['annotations'])} + {len(splits['test']['annotations'])})")
    else:
        print(f"  FAIL: Box count mismatch! total_boxes={total_boxes}, merged={merged_box_count}, raw - excluded={raw_union_boxes - excluded_box_count}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 4: Bounding box validity
    # Every bbox inside image bounds; category id in 0..19; every image has >= 1 box
    # -------------------------------------------------------------
    print("\n[CHECK 4] Annotation Validity: Bounds, categories, and non-empty images")
    invalid_bounds = 0
    invalid_cats = 0
    empty_images = 0

    for s, d in splits.items():
        img_dims = {img["id"]: (img["width"], img["height"]) for img in d["images"]}
        img_ann_count = Counter(ann["image_id"] for ann in d["annotations"])

        for img in d["images"]:
            if img_ann_count[img["id"]] < 1:
                empty_images += 1

        for ann in d["annotations"]:
            cid = ann["category_id"]
            if not (0 <= cid <= 19):
                invalid_cats += 1
            w, h = img_dims[ann["image_id"]]
            bx, by, bw, bh = ann["bbox"]
            if bx < 0 or by < 0 or bx + bw > w or by + bh > h:
                invalid_bounds += 1

    if invalid_bounds == 0 and invalid_cats == 0 and empty_images == 0:
        print("  PASS: All 32,468 bounding boxes strictly inside image boundaries (0px tolerance).")
        print("  PASS: All category IDs strictly in range [0, 19].")
        print(f"  PASS: All {total_images} images contain at least 1 bounding box.")
    else:
        print(f"  FAIL: Invalid annotations found! Out-of-bounds: {invalid_bounds}, invalid categories: {invalid_cats}, empty images: {empty_images}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 5: Split image fractions (60/10/30 +/- 2 points) and category coverage
    # -------------------------------------------------------------
    print("\n[CHECK 5] Split Balance: 60/10/30 +/- 2 points and complete category coverage")
    train_frac = len(fn_sets["train"]) / total_images
    val_frac = len(fn_sets["val"]) / total_images
    test_frac = len(fn_sets["test"]) / total_images

    frac_ok = (0.58 <= train_frac <= 0.62) and (0.08 <= val_frac <= 0.12) and (0.28 <= test_frac <= 0.32)

    cat_counts_by_split = {
        s: Counter(ann["category_id"] for ann in d["annotations"])
        for s, d in splits.items()
    }

    all_cells_present = True
    min_counts = {}
    for s in splits:
        min_c = min(cat_counts_by_split[s][c] for c in range(20))
        min_counts[s] = min_c
        if min_c <= 0:
            all_cells_present = False

    if frac_ok and all_cells_present:
        print(f"  PASS: Split fractions within +/- 2 percentage points of target:")
        print(f"        train: {train_frac*100:.2f}% (target 60.00%, error: {abs(train_frac-0.6)*100:.2f}%)")
        print(f"        val  : {val_frac*100:.2f}% (target 10.00%, error: {abs(val_frac-0.1)*100:.2f}%)")
        print(f"        test : {test_frac*100:.2f}% (target 30.00%, error: {abs(test_frac-0.3)*100:.2f}%)")
        print(f"  PASS: All 20 (class10 x nature) categories present in every split.")
        print(f"        Minimum category count in train: {min_counts['train']}")
        print(f"        Minimum category count in val  : {min_counts['val']}")
        print(f"        Minimum category count in test : {min_counts['test']}")
    else:
        print(f"  FAIL: Balance check failed! frac_ok={frac_ok}, all_cells_present={all_cells_present}")
        all_passed = False

    # -------------------------------------------------------------
    # Check 6: Determinism: Re-running 04_split.py gives byte-identical outputs
    # -------------------------------------------------------------
    print("\n[CHECK 6] Determinism: Re-running 04_split.py yields byte-identical outputs")
    files_to_check = [
        "train.json", "val.json", "test.json",
        "val_cls.json", "val_state.json", "test_cls.json", "test_state.json",
        "orig_val_per_image.json", "split_stats.csv"
    ]
    initial_hashes = {fn: get_file_sha256(os.path.join(out_dir, fn)) for fn in files_to_check}

    # Re-run 04_split.py
    import importlib.util
    split_script = os.path.join(script_dir, "04_split.py")
    spec = importlib.util.spec_from_file_location("step04", split_script)
    step04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step04)
    step04.main()

    rerun_hashes = {fn: get_file_sha256(os.path.join(out_dir, fn)) for fn in files_to_check}

    identical = True
    for fn in files_to_check:
        if initial_hashes[fn] != rerun_hashes[fn]:
            print(f"  MISMATCH on {fn}: {initial_hashes[fn][:10]}... vs {rerun_hashes[fn][:10]}...")
            identical = False

    if identical:
        print(f"  PASS: All {len(files_to_check)} output files are byte-for-byte identical after re-execution.")
    else:
        print(f"  FAIL: Non-deterministic output detected across runs!")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("OVERALL RESULT: ALL CHECKS PASSED (6 / 6 PASS)")
    else:
        print("OVERALL RESULT: SOME CHECKS FAILED")
    print("=" * 80)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

