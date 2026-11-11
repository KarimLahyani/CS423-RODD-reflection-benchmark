"""Build v2 split: Session-disjoint partition of RODD dataset with same-day dHash <= 4 merging.

Rules:
1. Phone photos (IMG_YYYYMMDD_HHMMSS) sorted by timestamp. A new session starts
   whenever the gap to the previous photo is > 120 seconds (26 raw sessions).
2. Session Merging: Two sessions from the same day are merged into one unit if they
   share >= 3 image pairs with dHash <= 4.
   - sess_17 <-> sess_18 (2024-09-16): 17 pairs with dHash <= 4
   - sess_20 <-> sess_21 (2024-09-17): 72 pairs with dHash <= 4
   - sess_21 <-> sess_22 (2024-09-17): 5 pairs with dHash <= 4
   - sess_21 <-> sess_23 (2024-09-17): 41 pairs with dHash <= 4
   Yielding 22 session units (and 4,066 non-phone units -> 4,088 units total).
3. Hard Constraint & Seed Search:
   In val and test, every one of the 20 class x nature cells must have >= 20 boxes.
   Because of indivisible session clusters, satisfying this constraint with val <= 12% is
   impossible (0 seeds); relaxing val's target to 10-15% of images allows valid splits.
   Evaluating seeds 0-999 finds Seed 21 as optimal (smallest max deviation: 22.95%).
4. Outputs written to out/v2/:
   - train.json, val.json, test.json (20 categories)
   - val_cls.json, test_cls.json (10 categories)
   - val_state.json, test_state.json (2 categories)
   - split_stats.csv
   - sessions.csv
5. Contact sheet saved to out/inspect/v2_cross_unit_dhash.png (12 random cross-unit pairs with dHash <= 6, Seed 0).
"""
import os
import sys
import json
import csv
import re
import random
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_font(size, bold=False):
    font_paths = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()

def render_v2_contact_sheet(pairs, out_path, img_dir):
    """
    Renders 12 cross-unit pairs with dHash <= 6 in a 6-row x 2-column layout.
    """
    title_font = get_font(20, bold=True)
    sub_font = get_font(13, bold=False)
    panel_title_font = get_font(13, bold=True)
    label_font = get_font(11, bold=False)
    bold_label_font = get_font(11, bold=True)

    panel_w = 780
    panel_h = 320
    thumb_w = 360
    thumb_h = 220
    pad_x = 25
    pad_y = 20
    top_header_h = 90

    cols = 2
    rows = 6
    sheet_w = pad_x * 3 + panel_w * cols
    sheet_h = top_header_h + pad_y * (rows + 1) + panel_h * rows

    canvas = Image.new("RGB", (sheet_w, sheet_h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    # Top Header banner
    draw.rectangle([(0, 0), (sheet_w, top_header_h - 10)], fill=(33, 43, 54))
    draw.text((pad_x, 15), "RODD v2: Cross-Unit dHash <= 6 Inspection Sheet (Seed 0)", fill=(255, 255, 255), font=title_font)
    draw.text((pad_x, 48), "12 Random Cross-Unit Image Pairs across distinct session units (Seed 0)", fill=(180, 195, 210), font=sub_font)

    for idx, (fn1, fn2, dist, u1, u2, s1, s2) in enumerate(pairs):
        r = idx // cols
        c = idx % cols
        px = pad_x + c * (panel_w + pad_x)
        py = top_header_h + r * (panel_h + pad_y)

        # Panel card background
        draw.rounded_rectangle(
            [(px, py), (px + panel_w, py + panel_h)],
            radius=8,
            fill=(255, 255, 255),
            outline=(210, 218, 226),
            width=1
        )

        header_text = f"Pair #{idx + 1:02d}  |  dHash Dist: {dist}  |  {u1} ({s1.upper()}) vs {u2} ({s2.upper()})"

        # Header banner
        draw.rounded_rectangle(
            [(px, py), (px + panel_w, py + 32)],
            radius=8,
            fill=(240, 244, 248),
            outline=(210, 218, 226),
            width=1
        )
        draw.text((px + 12, py + 7), header_text, fill=(33, 37, 41), font=panel_title_font)

        # Place images
        p1 = os.path.join(img_dir, fn1)
        p2 = os.path.join(img_dir, fn2)

        for img_idx, (p, fn, u, s) in enumerate([(p1, fn1, u1, s1), (p2, fn2, u2, s2)]):
            im_x = px + 15 + img_idx * (thumb_w + 30)
            im_y = py + 42

            try:
                with Image.open(p) as im:
                    im_copy = im.convert("RGB")
                    im_copy.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    tw, th = im_copy.size
                    ox = (thumb_w - tw) // 2
                    oy = (thumb_h - th) // 2
                    draw.rectangle([(im_x, im_y), (im_x + thumb_w, im_y + thumb_h)], fill=(235, 238, 242))
                    canvas.paste(im_copy, (im_x + ox, im_y + oy))
            except Exception as e:
                draw.rectangle([(im_x, im_y), (im_x + thumb_w, im_y + thumb_h)], fill=(230, 230, 230))
                draw.text((im_x + 10, im_y + thumb_h // 2 - 10), f"Error loading image: {e}", fill=(200, 0, 0), font=label_font)

            draw.rectangle([(im_x, im_y), (im_x + thumb_w, im_y + thumb_h)], outline=(190, 200, 210), width=1)

            lbl_y = im_y + thumb_h + 8
            split_color = (40, 167, 69) if s == "train" else ((0, 123, 255) if s == "test" else (255, 153, 0))
            draw.text((im_x, lbl_y), f"File: {fn[:38]}", fill=(50, 50, 50), font=label_font)
            draw.text((im_x, lbl_y + 16), f"Unit: {u}", fill=(80, 80, 80), font=bold_label_font)
            draw.text((im_x + 130, lbl_y + 16), f"Split: {s.upper()}", fill=split_color, font=bold_label_font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, format="PNG", quality=95)
    print(f"Saved contact sheet to {out_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    rodd_dir = os.path.join(project_root, "RODD")
    img_dir = os.path.join(rodd_dir, "images")
    clean_dir = os.path.join(script_dir, "..")
    out_dir = os.path.join(clean_dir, "out")
    v2_dir = os.path.join(out_dir, "v2")
    inspect_dir = os.path.join(out_dir, "inspect")

    os.makedirs(v2_dir, exist_ok=True)
    os.makedirs(inspect_dir, exist_ok=True)

    merged_json_path = os.path.join(out_dir, "merged_all.json")
    groups_csv_path = os.path.join(out_dir, "groups.csv")
    hashes_csv_path = os.path.join(clean_dir, "cache", "hashes.csv")

    print(f"Loading merged dataset from {merged_json_path}...")
    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    categories_20 = merged_data["categories"]
    cat_names_20 = [c["name"] for c in sorted(categories_20, key=lambda x: x["id"])]

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

    img_by_fn = {img["file_name"]: img for img in merged_data["images"]}
    anns_by_fn = defaultdict(list)
    for ann in merged_data["annotations"]:
        img_id = ann["image_id"]
        fn = merged_data["images"][img_id]["file_name"]
        anns_by_fn[fn].append(ann)

    print(f"Loaded {len(img_by_fn)} clean images, {len(merged_data['annotations'])} clean annotations.")

    # 1. Segment Phone Photos into 26 Sessions
    pat = re.compile(r"IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
    def get_dt(fn):
        m = pat.search(fn)
        y, mo, d, h, mi, s = map(int, m.groups())
        return datetime(y, mo, d, h, mi, s)

    phone_merged = sorted([fn for fn in img_by_fn if pat.search(fn)], key=lambda x: (get_dt(x), x))
    print(f"Identified {len(phone_merged)} phone photos.")

    sessions = []
    cur = [phone_merged[0]]
    for fn in phone_merged[1:]:
        prev_dt = get_dt(cur[-1])
        cur_dt = get_dt(fn)
        if (cur_dt - prev_dt).total_seconds() > 120:
            sessions.append(cur)
            cur = [fn]
        else:
            cur.append(fn)
    if cur:
        sessions.append(cur)

    print(f"Segmented phone photos into {len(sessions)} raw sessions (gap > 120 s).")

    # Load hashes
    hashes = {}
    with open(hashes_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["file_name"] in img_by_fn:
                hashes[row["file_name"]] = int(row["dhash"], 16)

    # 2. Check same-day session pairs with dHash <= 4 and merge
    print("\n" + "="*80)
    print("SESSION MERGE AUDIT (Same-day sessions sharing >= 3 pairs with dHash <= 4)")
    print("="*80)
    sess_date = {i: get_dt(s[0]).strftime("%Y-%m-%d") for i, s in enumerate(sessions)}
    parent = list(range(len(sessions)))
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    merge_edges = []
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            if sess_date[i] == sess_date[j]:
                h_i = [hashes[fn] for fn in sessions[i]]
                h_j = [hashes[fn] for fn in sessions[j]]
                cnt_le4 = 0
                for hi in h_i:
                    for hj in h_j:
                        diff = bin(hi ^ hj).count("1")
                        if diff <= 4:
                            cnt_le4 += 1
                if cnt_le4 >= 3:
                    merge_edges.append((i, j, cnt_le4))
                    union(i, j)
                    print(f"  MERGE: sess_{i:02d} <-> sess_{j:02d} ({sess_date[i]}): {cnt_le4} pairs with dHash <= 4")
                elif cnt_le4 > 0:
                    print(f"  NO MERGE (< 3 pairs): sess_{i:02d} <-> sess_{j:02d} ({sess_date[i]}): {cnt_le4} pairs with dHash <= 4")

    # Group connected components of sessions
    comp_to_sids = defaultdict(list)
    for i in range(len(sessions)):
        comp_to_sids[find(i)].append(i)

    merged_sess_units = []
    sess_to_unit_map = {}
    for p, sids in sorted(comp_to_sids.items()):
        all_fns = []
        for sid in sorted(sids):
            all_fns.extend(sessions[sid])
        uid = "sess_" + "_".join(f"{sid:02d}" for sid in sorted(sids))
        for sid in sids:
            sess_to_unit_map[sid] = uid
        merged_sess_units.append((uid, sids, all_fns))

    print(f"\nResult: 26 raw sessions merged into {len(merged_sess_units)} session units.")
    for uid, sids, fns in merged_sess_units:
        if len(sids) > 1:
            print(f"  * Merged Unit '{uid}': sessions {[f'sess_{sid:02d}' for sid in sids]}, {len(fns)} images")
    print("="*80 + "\n")

    # 3. Define Non-Phone Units
    fn_to_v1_gid = {}
    with open(groups_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            fn_to_v1_gid[row[0]] = int(row[1])

    nonphone_groups = defaultdict(list)
    for fn in img_by_fn:
        if not pat.search(fn):
            gid = fn_to_v1_gid[fn]
            nonphone_groups[gid].append(fn)
    nonphone_units = [(f"nonphone_{gid}", [gid], nonphone_groups[gid]) for gid in sorted(nonphone_groups.keys())]

    all_units = merged_sess_units + nonphone_units
    total_imgs = len(img_by_fn)
    print(f"Total units defined: {len(all_units)} ({len(merged_sess_units)} session units, {len(nonphone_units)} non-phone groups).")

    total_cat_counts = Counter()
    unit_cat_counts = {}
    unit_sizes = {}
    unit_fns_map = {}
    fn_to_unit = {}
    for uid, sids, fns in all_units:
        unit_fns_map[uid] = fns
        for fn in fns:
            fn_to_unit[fn] = uid
        cnts = Counter()
        for fn in fns:
            for ann in anns_by_fn[fn]:
                cid = ann["category_id"]
                cnts[cid] += 1
                total_cat_counts[cid] += 1
        unit_cat_counts[uid] = cnts
        unit_sizes[uid] = len(fns)

    target_ratios = {"train": 0.60, "val": 0.10, "test": 0.30}
    target_imgs = {s: target_ratios[s] * total_imgs for s in target_ratios}
    target_cats = {s: {c: target_ratios[s] * total_cat_counts[c] for c in range(20)} for s in ["train", "val", "test"]}

    sess_uids = [u[0] for u in merged_sess_units]
    nonphone_uids = [u[0] for u in nonphone_units]

    # 4. Seed search with hard constraint: in val and test, every cell has >= 20 boxes
    print("Searching seeds 0-999 with hard constraint (val & test >= 20 boxes in all 20 cells)...")
    def evaluate_seed(seed):
        rng = random.Random(seed)
        sess_order = list(sess_uids)
        rng.shuffle(sess_order)

        cur_imgs = {"train": 0, "val": 0, "test": 0}
        cur_cats = {"train": [0]*20, "val": [0]*20, "test": [0]*20}
        unit_assignment = {}

        phone_caps = {"train": 0.65 * 12842, "val": 0.16 * 12842, "test": 0.35 * 12842}
        for uid in sess_order:
            u_len = unit_sizes[uid]
            u_cats = unit_cat_counts[uid]
            valid = [s for s in ["train", "val", "test"] if cur_imgs[s] + u_len <= phone_caps[s]]
            if not valid:
                valid = [s for s in ["train", "val", "test"] if cur_imgs[s] < target_imgs[s]]
            if not valid:
                valid = ["train", "val", "test"]

            best_s, best_score = None, -1e18
            for s in valid:
                score = 0.0
                for c, cnt in u_cats.items():
                    if 10 <= c <= 19:
                        cur_c = cur_cats[s][c]
                        tar_c = target_ratios[s] * total_cat_counts[c]
                        if cur_c < 20:
                            score += 50000.0 * (cnt / total_cat_counts[c])
                        else:
                            score += 20.0 * (tar_c - cur_c) * (cnt / total_cat_counts[c])
                tar_p = {"train": 0.60 * 12842, "val": 0.10 * 12842, "test": 0.30 * 12842}
                score += 2.0 * (tar_p[s] - cur_imgs[s]) / tar_p[s]
                score += rng.uniform(-0.1, 0.1)
                if score > best_score:
                    best_score, best_s = score, s
            cur_imgs[best_s] += u_len
            unit_assignment[uid] = best_s
            for c, cnt in u_cats.items():
                cur_cats[best_s][c] += cnt

        np_order = list(nonphone_uids)
        np_order.sort(key=lambda u: unit_sizes[u] * (1.0 + rng.uniform(-0.05, 0.05)), reverse=True)
        upper_bounds = {"train": 0.62 * total_imgs, "val": 0.15 * total_imgs, "test": 0.32 * total_imgs}
        for uid in np_order:
            u_len = unit_sizes[uid]
            u_cats = unit_cat_counts[uid]
            valid = [s for s in ["train", "val", "test"] if cur_imgs[s] + u_len <= upper_bounds[s]]
            if not valid:
                valid = [s for s in ["train", "val", "test"] if cur_imgs[s] < target_imgs[s]]
            if not valid:
                valid = ["train", "val", "test"]

            best_s, best_score = None, -1e18
            for s in valid:
                score = 0.0
                for c, cnt in u_cats.items():
                    cur_c = cur_cats[s][c]
                    tar_c = target_ratios[s] * total_cat_counts[c]
                    if cur_c < 20:
                        score += 50000.0 * (cnt / total_cat_counts[c])
                    else:
                        score += 20.0 * (tar_c - cur_c) * (cnt / total_cat_counts[c])
                score += 5.0 * (target_imgs[s] - cur_imgs[s]) / target_imgs[s]
                score += rng.uniform(-0.005, 0.005)
                if score > best_score:
                    best_score, best_s = score, s
            cur_imgs[best_s] += u_len
            unit_assignment[uid] = best_s
            for c, cnt in u_cats.items():
                cur_cats[best_s][c] += cnt

        val_test_ge20 = all(cur_cats["val"][c] >= 20 and cur_cats["test"][c] >= 20 for c in range(20))
        train_ge20 = all(cur_cats["train"][c] >= 20 for c in range(20))
        val_pct = cur_imgs["val"] / total_imgs
        # Must be valid and val within 10-15%
        is_valid = val_test_ge20 and train_ge20 and (0.10 <= val_pct <= 0.15)
        max_dev = max(abs(cur_cats[s][c] / total_cat_counts[c] - target_ratios[s]) * 100.0 for s in ["train", "val", "test"] for c in range(20))
        return is_valid, max_dev, cur_imgs, cur_cats, unit_assignment

    best_seed = None
    best_dev = 999.0
    valid_seeds_count = 0
    for s in range(1000):
        ok, dev, _, _, _ = evaluate_seed(s)
        if ok:
            valid_seeds_count += 1
            if dev < best_dev:
                best_dev = dev
                best_seed = s

    print(f"Seed search complete: {valid_seeds_count} valid seeds found (satisfying val & test >= 20 in all cells and val 10-15%).")
    print(f"Optimal seed selected: Seed {best_seed} with smallest max deviation: {best_dev:.2f}%")

    ok, dev, final_imgs, final_cats, unit_assignment = evaluate_seed(best_seed)
    assert ok, "Optimal seed must satisfy hard constraints!"

    split_fns = {"train": [], "val": [], "test": []}
    for uid, split_name in unit_assignment.items():
        split_fns[split_name].extend(unit_fns_map[uid])

    for s in ["train", "val", "test"]:
        split_fns[s].sort()
        pct = len(split_fns[s]) / total_imgs * 100
        print(f"Split {s:5s}: {len(split_fns[s]):5d} images ({pct:.2f}%)")

    # 5. Print and Save Session List
    print("\n" + "="*120)
    print("SESSION LIST (v2)")
    print("="*120)
    session_rows = []
    print(f"{'ID':8s} | {'Unit':18s} | {'Date':10s} | {'Start - End':19s} | {'Dur(s)':6s} | {'Imgs':5s} | {'Split':5s} | Boxes per category (non-zero)")
    print("-" * 120)
    for sid, s_fns in enumerate(sessions):
        dts = [get_dt(fn) for fn in s_fns]
        dur = int((dts[-1] - dts[0]).total_seconds())
        date_str = dts[0].strftime("%Y-%m-%d")
        start_str = dts[0].strftime("%H:%M:%S")
        end_str = dts[-1].strftime("%H:%M:%S")
        s_raw_id = f"sess_{sid:02d}"
        u_id = sess_to_unit_map[sid]
        s_split = unit_assignment[u_id]
        box_cnts = Counter(ann["category_id"] for fn in s_fns for ann in anns_by_fn[fn])
        tot_boxes = sum(box_cnts.values())

        both_nature = 0
        for fn in s_fns:
            natures = {ann["category_id"] // 10 for ann in anns_by_fn[fn]}
            if 0 in natures and 1 in natures:
                both_nature += 1

        row = {
            "session_id": s_raw_id,
            "unit_id": u_id,
            "date": date_str,
            "start_time": start_str,
            "end_time": end_str,
            "duration_seconds": dur,
            "images": len(s_fns),
            "split": s_split,
            "boxes": tot_boxes,
            "images_both_real_reflected": both_nature
        }
        for cid in range(20):
            row[cat_names_20[cid]] = box_cnts[cid]
        session_rows.append(row)

        nz_str = ", ".join(f"{cat_names_20[cid]}:{box_cnts[cid]}" for cid in sorted(box_cnts.keys()))
        print(f"{s_raw_id:8s} | {u_id:18s} | {date_str:10s} | {start_str} - {end_str} | {dur:6d} | {len(s_fns):5d} | {s_split:5s} | {nz_str}")
    print("="*120 + "\n")

    sessions_csv_path = os.path.join(v2_dir, "sessions.csv")
    sess_fieldnames = [
        "session_id", "unit_id", "date", "start_time", "end_time", "duration_seconds",
        "images", "split", "boxes", "images_both_real_reflected"
    ] + cat_names_20

    with open(sessions_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sess_fieldnames)
        writer.writeheader()
        for r in session_rows:
            writer.writerow(r)
    print(f"Saved {sessions_csv_path}")

    # 6. Write COCO Files for v2
    def build_coco(fns, cat_list, cat_id_func):
        imgs = []
        anns = []
        ann_counter = 0
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

    for s in ["train", "val", "test"]:
        coco_obj = build_coco(split_fns[s], categories_20, lambda c: c)
        out_p = os.path.join(v2_dir, f"{s}.json")
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(coco_obj, f, indent=2)
        print(f"Saved {out_p}: {len(coco_obj['images'])} images, {len(coco_obj['annotations'])} boxes")

    for s in ["val", "test"]:
        coco_cls = build_coco(split_fns[s], categories_10_cls, lambda c: c % 10)
        out_p = os.path.join(v2_dir, f"{s}_cls.json")
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(coco_cls, f, indent=2)
        print(f"Saved {out_p}: {len(coco_cls['images'])} images, {len(coco_cls['annotations'])} boxes")

    for s in ["val", "test"]:
        coco_state = build_coco(split_fns[s], categories_2_state, lambda c: c // 10)
        out_p = os.path.join(v2_dir, f"{s}_state.json")
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(coco_state, f, indent=2)
        print(f"Saved {out_p}: {len(coco_state['images'])} images, {len(coco_state['annotations'])} boxes")

    # 7. Write split_stats.csv
    split_stats_path = os.path.join(v2_dir, "split_stats.csv")
    stats_fieldnames = [
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
            row[cname] = final_cats[s][cid]
        stats_rows.append(row)

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

    with open(split_stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
        writer.writeheader()
        for r in stats_rows:
            writer.writerow(r)
    print(f"Saved {split_stats_path}")

    # 8. Cross-Unit dHash <= 6 Analysis and Contact Sheet
    print("\nComputing cross-unit pairs with dHash <= 6...")
    all_fns = sorted(img_by_fn.keys())
    h_arr = np.array([hashes[fn] for fn in all_fns], dtype=np.uint64)
    u_arr = [fn_to_unit[fn] for fn in all_fns]

    buckets = defaultdict(list)
    for idx, h in enumerate(h_arr):
        for pos in range(8):
            byte_val = int((h >> (pos * 8)) & 0xFF)
            buckets[(pos, byte_val)].append(idx)

    seen = set()
    cross_unit_le6 = []
    for b_indices in buckets.values():
        m_len = len(b_indices)
        if m_len > 1:
            for i in range(m_len):
                idx1 = b_indices[i]
                for j in range(i + 1, m_len):
                    idx2 = b_indices[j]
                    pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
                    if pair not in seen:
                        seen.add(pair)
                        if u_arr[idx1] != u_arr[idx2]:
                            diff = int((h_arr[idx1] ^ h_arr[idx2]).bit_count())
                            if diff <= 6:
                                cross_unit_le6.append((all_fns[idx1], all_fns[idx2], diff, u_arr[idx1], u_arr[idx2]))

    cross_unit_le6.sort(key=lambda x: (x[0], x[1]))
    print(f"Total cross-unit pairs with dHash <= 6: {len(cross_unit_le6)}")

    rng_sample = random.Random(0)
    sampled_12_raw = rng_sample.sample(cross_unit_le6, 12)
    sampled_12 = []
    for fn1, fn2, dist, u1, u2 in sampled_12_raw:
        s1 = unit_assignment[u1]
        s2 = unit_assignment[u2]
        sampled_12.append((fn1, fn2, dist, u1, u2, s1, s2))

    contact_sheet_path = os.path.join(inspect_dir, "v2_cross_unit_dhash.png")
    render_v2_contact_sheet(sampled_12, contact_sheet_path, img_dir)

    # 9. Verification of Cross-Split Phone Photos within 60 s
    print("\nVerifying cross-split phone photo temporal distance...")
    seq = sorted([(get_dt(fn), fn) for fn in phone_merged], key=lambda x: x[0])
    close_pairs_60s = 0
    close_pairs_120s = 0
    for i in range(len(seq) - 1):
        t1, f1 = seq[i]
        t2, f2 = seq[i+1]
        s1 = unit_assignment[fn_to_unit[f1]]
        s2 = unit_assignment[fn_to_unit[f2]]
        if s1 != s2:
            gap = (t2 - t1).total_seconds()
            if gap <= 60:
                close_pairs_60s += 1
            if gap <= 120:
                close_pairs_120s += 1

    print(f"Cross-split phone photo pairs within 60 s: {close_pairs_60s} (Requirement: strictly 0)")
    print(f"Cross-split phone photo pairs within 120 s: {close_pairs_120s}")
    assert close_pairs_60s == 0, "Cross-split phone pairs within 60s must be 0!"

    # 10. Print Per-Cell Deviation & Min Boxes Table
    print("\n" + "="*110)
    print("PER-CELL DEVIATION & MIN BOXES TABLE (v2)")
    print("="*110)
    print(f"{'Category':12s} | {'Train Boxes':11s} {'Train %':8s} {'Dev':8s} | {'Val Boxes':10s} {'Val %':8s} {'Dev':8s} | {'Test Boxes':10s} {'Test %':8s} {'Dev':8s}")
    print("-" * 110)
    min_train = 10**9
    min_val = 10**9
    min_test = 10**9
    for cid in range(20):
        cname = cat_names_20[cid]
        tot = total_cat_counts[cid]
        tr_cnt = final_cats['train'][cid]
        tr_pct = tr_cnt / tot * 100.0
        tr_dev = tr_pct - 60.0
        min_train = min(min_train, tr_cnt)

        va_cnt = final_cats['val'][cid]
        va_pct = va_cnt / tot * 100.0
        va_dev = va_pct - 10.0
        min_val = min(min_val, va_cnt)

        te_cnt = final_cats['test'][cid]
        te_pct = te_cnt / tot * 100.0
        te_dev = te_pct - 30.0
        min_test = min(min_test, te_cnt)

        print(f"{cname:12s} | {tr_cnt:6d}/{tot:<4d}  {tr_pct:6.2f}% {tr_dev:+6.2f}% | {va_cnt:5d}/{tot:<4d}  {va_pct:6.2f}% {va_dev:+6.2f}% | {te_cnt:6d}/{tot:<4d}  {te_pct:6.2f}% {te_dev:+6.2f}%")
    print("="*110)
    print(f"Minimum box count: Train = {min_train}, Val = {min_val} (>= 20: {min_val >= 20}), Test = {min_test} (>= 20: {min_test >= 20})")
    print(f"Overall Max Deviation: {best_dev:.2f}% (on mousemi train: 37.05% vs 60.00% target)")
    print("All checks completed successfully!")

if __name__ == "__main__":
    main()
