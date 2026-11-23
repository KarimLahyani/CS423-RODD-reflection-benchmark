#!/usr/bin/env python3
"""
evaluation/score.py

Evaluates detection results against COCO ground truth for RODD.
Usage:
  score.py --gt GT.json --dets DETS.json --classes {20,10} --out OUT.json
           [--images LIST.txt] [--record-wise] [--min-score S] [--only-images-with-dets] [--authors]
           [--conf T | --pick-conf]
"""

import argparse
import collections
import contextlib
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_iou(box1, box2):
    """
    Compute IoU between two [x, y, w, h] boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, w1) * max(0.0, h1)
    area2 = max(0.0, w2) * max(0.0, h2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_collapsed_pairs(dets_by_fn):
    """
    Unordered pairs of detections in one photo with the same id % 10,
    different id // 10, and IoU > 0.7.
    """
    total_images_with_pairs = 0
    total_pairs = 0

    for fn, fn_dets in dets_by_fn.items():
        pairs_in_img = 0
        n = len(fn_dets)
        for i in range(n):
            for j in range(i + 1, n):
                d1 = fn_dets[i]
                d2 = fn_dets[j]
                c1 = d1["category_id"]
                c2 = d2["category_id"]
                if (c1 % 10 == c2 % 10) and (c1 // 10 != c2 // 10):
                    if compute_iou(d1["bbox"], d2["bbox"]) > 0.7:
                        pairs_in_img += 1
        if pairs_in_img > 0:
            total_images_with_pairs += 1
            total_pairs += pairs_in_img

    return {
        "images_with_pairs": total_images_with_pairs,
        "pairs": total_pairs,
    }


def compute_pr_at_threshold(gt_boxes_by_image, dets_by_image, threshold):
    """
    Precision, recall, F1 matching rule for 20 classes at threshold T:
    - Per image, process detections by descending score (stable sort).
    - Each detection matches the still-unmatched GT box with same 20-class id and highest IoU >= 0.5.
    - Matched detection is TP. Unmatched detection is FP for its nature. Unmatched GT box is FN for its nature.
    """
    tp_real = 0
    fp_real = 0
    fn_real = 0

    tp_refl = 0
    fp_refl = 0
    fn_refl = 0

    all_img_ids = sorted(set(gt_boxes_by_image.keys()) | set(dets_by_image.keys()))

    for img_id in all_img_ids:
        gt_list = gt_boxes_by_image.get(img_id, [])
        matched_gt_indices = set()
        det_list = [d for d in dets_by_image.get(img_id, []) if d["score"] >= threshold]
        # Python sort is stable Timsort
        det_list.sort(key=lambda d: d["score"], reverse=True)

        for det in det_list:
            det_cat = det["category_id"]
            det_nature = det_cat // 10
            best_iou = -1.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_list):
                if gt_idx in matched_gt_indices:
                    continue
                if gt["category_id"] != det_cat:
                    continue
                iou = compute_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= 0.5:
                matched_gt_indices.add(best_gt_idx)
                if det_nature == 0:
                    tp_real += 1
                else:
                    tp_refl += 1
            else:
                if det_nature == 0:
                    fp_real += 1
                else:
                    fp_refl += 1

        for gt_idx, gt in enumerate(gt_list):
            if gt_idx not in matched_gt_indices:
                gt_nature = gt["category_id"] // 10
                if gt_nature == 0:
                    fn_real += 1
                else:
                    fn_refl += 1

    def make_stats(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "precision": float(p),
            "recall": float(r),
            "F1": float(f1),
        }

    return {
        "real": make_stats(tp_real, fp_real, fn_real),
        "reflected": make_stats(tp_refl, fp_refl, fn_refl),
    }


def evaluate_coco(gt_images, gt_annotations, dt_annotations, categories, cat_ids):
    """
    Evaluates detections against ground truth using pycocotools COCOeval.
    cat_ids are categories that have at least one GT box in the evaluated images.
    """
    if len(cat_ids) == 0:
        return {
            "AP50": 0.0,
            "AP75": 0.0,
            "mAP": 0.0,
            "per_class_AP50": {},
            "per_class_mAP": {},
            "n_classes_scored": 0,
        }

    if len(dt_annotations) == 0:
        return {
            "AP50": 0.0,
            "AP75": 0.0,
            "mAP": 0.0,
            "per_class_AP50": {str(c): 0.0 for c in cat_ids},
            "per_class_mAP": {str(c): 0.0 for c in cat_ids},
            "n_classes_scored": len(cat_ids),
        }

    # Ensure annotations have area and iscrowd
    gt_anns_clean = []
    for ann in gt_annotations:
        ann_copy = dict(ann)
        if "area" not in ann_copy:
            ann_copy["area"] = ann_copy["bbox"][2] * ann_copy["bbox"][3]
        if "iscrowd" not in ann_copy:
            ann_copy["iscrowd"] = 0
        gt_anns_clean.append(ann_copy)

    dt_anns_clean = []
    for idx, ann in enumerate(dt_annotations, 1):
        ann_copy = dict(ann)
        ann_copy["id"] = idx
        if "area" not in ann_copy:
            ann_copy["area"] = ann_copy["bbox"][2] * ann_copy["bbox"][3]
        if "iscrowd" not in ann_copy:
            ann_copy["iscrowd"] = 0
        dt_anns_clean.append(ann_copy)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": img["id"]} for img in gt_images],
            "annotations": gt_anns_clean,
            "categories": categories,
        }
        coco_gt.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = {
            "images": [{"id": img["id"]} for img in gt_images],
            "annotations": dt_anns_clean,
            "categories": categories,
        }
        coco_dt.createIndex()

        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.catIds = list(cat_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # precision shape: (T, R, K, A, M)
    # Area 'all' is index 0, maxDets 100 is index 2
    p = ev.eval["precision"][:, :, :, 0, 2]  # shape: (10, 101, len(cat_ids))

    valid_all = p[p > -1]
    overall_map = float(np.mean(valid_all)) if len(valid_all) > 0 else 0.0

    p_50 = p[0, :, :]
    valid_50 = p_50[p_50 > -1]
    overall_ap50 = float(np.mean(valid_50)) if len(valid_50) > 0 else 0.0

    p_75 = p[5, :, :]
    valid_75 = p_75[p_75 > -1]
    overall_ap75 = float(np.mean(valid_75)) if len(valid_75) > 0 else 0.0

    per_class_map = {}
    per_class_ap50 = {}

    for idx, c in enumerate(cat_ids):
        p_c = p[:, :, idx]
        valid_c = p_c[p_c > -1]
        per_class_map[str(c)] = float(np.mean(valid_c)) if len(valid_c) > 0 else 0.0

        p_c_50 = p[0, :, idx]
        valid_c_50 = p_c_50[p_c_50 > -1]
        per_class_ap50[str(c)] = float(np.mean(valid_c_50)) if len(valid_c_50) > 0 else 0.0

    return {
        "AP50": overall_ap50,
        "AP75": overall_ap75,
        "mAP": overall_map,
        "n_classes_scored": len(cat_ids),
        "per_class_AP50": per_class_ap50,
        "per_class_mAP": per_class_map,
    }


def main():
    parser = argparse.ArgumentParser(description="RODD detection scorer")
    parser.add_argument("--gt", required=True, help="Path to ground-truth JSON")
    parser.add_argument("--dets", required=True, help="Path to detections JSON")
    parser.add_argument("--classes", type=int, choices=[10, 20], required=True, help="Number of classes (10 or 20)")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--images", default=None, help="Path to text file with subset of image file names")
    parser.add_argument("--record-wise", action="store_true", help="Score per record instead of per photo")
    parser.add_argument("--min-score", type=float, default=None, help="Keep only detections with score > S")
    parser.add_argument("--only-images-with-dets", action="store_true", help="Evaluate only images with at least one remaining detection")
    parser.add_argument("--authors", action="store_true", help="Exact authors' scoring settings (--record-wise --min-score 0.3 --only-images-with-dets)")

    conf_group = parser.add_mutually_exclusive_group()
    conf_group.add_argument("--conf", type=float, default=None, help="Fixed confidence threshold T for precision/recall")
    conf_group.add_argument("--pick-conf", action="store_true", help="Pick threshold T on validation maximizing mean(real F1, reflected F1)")

    args = parser.parse_args()

    record_wise = args.record_wise
    min_score = args.min_score
    only_images_with_dets = args.only_images_with_dets

    if args.authors:
        record_wise = True
        min_score = 0.3
        only_images_with_dets = True

    # 1. Load GT and hash files
    gt_bytes = Path(args.gt).read_bytes()
    dets_bytes = Path(args.dets).read_bytes()
    gt_sha256 = hashlib.sha256(gt_bytes).hexdigest()
    dets_sha256 = hashlib.sha256(dets_bytes).hexdigest()

    gt_data = json.loads(gt_bytes.decode("utf-8"))
    dets_data = json.loads(dets_bytes.decode("utf-8"))

    if not isinstance(dets_data, list):
        raise ValueError("Detections file must be a JSON list of detection objects")

    all_gt_images = gt_data["images"]
    all_gt_annotations = gt_data["annotations"]
    all_gt_categories = gt_data.get("categories", [])

    gt_dims_by_fn = {img["file_name"]: (img["width"], img["height"]) for img in all_gt_images}

    # Validate input detections format and content
    det_counts_by_fn = collections.Counter()

    for det in dets_data:
        fn = det.get("file_name")
        if fn not in gt_dims_by_fn:
            raise ValueError(f"Unknown file name in detections: '{fn}'")

        cat_id = det.get("category_id")
        if args.classes == 20:
            if not (isinstance(cat_id, int) and 0 <= cat_id <= 19):
                raise ValueError(f"Category id {cat_id} out of range for 20 classes (0-19)")
        elif args.classes == 10:
            if not (isinstance(cat_id, int) and 0 <= cat_id <= 9):
                raise ValueError(f"Category id {cat_id} out of range for 10 classes (0-9)")

        score = det.get("score")
        if score is None or not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError(f"Detection score must be a finite number: score={score} in {fn}")

        bbox = det.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValueError(f"Invalid bbox format: {bbox}")

        for val in bbox:
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise ValueError(f"Bbox coordinate must be a finite number: bbox={bbox} in {fn}")

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"Non-positive box width or height: bbox={bbox} in {fn}")

        img_w, img_h = gt_dims_by_fn[fn]
        if (x < -1.0) or (y < -1.0) or (x + w > img_w + 1.0) or (y + h > img_h + 1.0):
            raise ValueError(
                f"Box beyond image boundaries by >1px: bbox={bbox}, image size=({img_w}, {img_h}) in {fn}"
            )

        det_counts_by_fn[fn] += 1
        if det_counts_by_fn[fn] > 100:
            raise ValueError(f"More than 100 detections for one file name: {fn}")

    min_det_score = None
    if len(dets_data) > 0:
        min_det_score = float(min(d["score"] for d in dets_data))

    # Step 1: Filter GT by --images if provided
    selected_image_fns = None
    if args.images:
        images_file = Path(args.images)
        if not images_file.exists():
            raise FileNotFoundError(f"--images file not found: {args.images}")
        with open(images_file, "r", encoding="utf-8") as f:
            selected_image_fns = set()
            for line in f:
                line_str = line.strip()
                if line_str:
                    if line_str not in gt_dims_by_fn:
                        raise ValueError(f"Unknown file name in --images: '{line_str}'")
                    selected_image_fns.add(line_str)

        evaluated_gt_images = [img for img in all_gt_images if img["file_name"] in selected_image_fns]
    else:
        evaluated_gt_images = list(all_gt_images)

    evaluated_img_ids = {img["id"] for img in evaluated_gt_images}
    evaluated_gt_annotations = [ann for ann in all_gt_annotations if ann["image_id"] in evaluated_img_ids]

    # Filter detections to evaluated images if --images was specified
    if selected_image_fns is not None:
        eval_dets_raw = [d for d in dets_data if d["file_name"] in selected_image_fns]
    else:
        eval_dets_raw = list(dets_data)

    # Step 2: Map detections to image ids by file_name
    gt_ids_by_fn = collections.defaultdict(list)
    for img in evaluated_gt_images:
        gt_ids_by_fn[img["file_name"]].append(img["id"])

    has_multi_records = any(len(ids) > 1 for ids in gt_ids_by_fn.values())
    if has_multi_records and not record_wise:
        raise ValueError("GT contains multiple records for one file_name; requires --record-wise")

    mapped_dets = []
    for d in eval_dets_raw:
        fn = d["file_name"]
        rec_ids = gt_ids_by_fn.get(fn, [])
        if record_wise:
            for rid in rec_ids:
                mapped_dets.append({
                    "bbox": d["bbox"],
                    "category_id": d["category_id"],
                    "file_name": fn,
                    "image_id": rid,
                    "score": float(d["score"]),
                })
        else:
            if rec_ids:
                mapped_dets.append({
                    "bbox": d["bbox"],
                    "category_id": d["category_id"],
                    "file_name": fn,
                    "image_id": rec_ids[0],
                    "score": float(d["score"]),
                })

    # Step 3: Filter by --min-score S (strictly > S)
    if min_score is not None:
        remaining_dets = [d for d in mapped_dets if d["score"] > min_score]
    else:
        remaining_dets = mapped_dets

    # Step 4: Filter by --only-images-with-dets
    if only_images_with_dets:
        dets_img_ids = {d["image_id"] for d in remaining_dets}
        evaluated_gt_images = [img for img in evaluated_gt_images if img["id"] in dets_img_ids]
        evaluated_img_ids = {img["id"] for img in evaluated_gt_images}
        evaluated_gt_annotations = [ann for ann in evaluated_gt_annotations if ann["image_id"] in evaluated_img_ids]

    gt_ids_renumbered = not args.authors
    if gt_ids_renumbered:
        renumbered_gt_annotations = []
        for new_id, ann in enumerate(evaluated_gt_annotations, 1):
            ann_copy = dict(ann)
            ann_copy["id"] = new_id
            renumbered_gt_annotations.append(ann_copy)
        evaluated_gt_annotations = renumbered_gt_annotations

    # Group evaluated GT annotations and remaining detections for evaluation
    gt_boxes_by_img = collections.defaultdict(list)
    for ann in evaluated_gt_annotations:
        gt_boxes_by_img[ann["image_id"]].append(ann)

    dets_by_img = collections.defaultdict(list)
    for d in remaining_dets:
        dets_by_img[d["image_id"]].append(d)

    # For collapsed pairs: group detections by photo (file_name)
    # Count each photo once
    dets_by_fn_evaluated = collections.defaultdict(list)
    for d in remaining_dets:
        # Note: in record-wise mode, d was replicated per record. To count each detection once per photo:
        dets_by_fn_evaluated[d["file_name"]].append(d)

    # Deduplicate detections per photo in record-wise mode if they were replicated
    dets_by_fn_deduped = {}
    if record_wise:
        for fn, dlist in dets_by_fn_evaluated.items():
            # A photo's detections were copied across records; take the unique detections
            # Each original detection had unique (category_id, tuple(bbox), score)
            seen = set()
            unique_dlist = []
            for d in dlist:
                key = (d["category_id"], tuple(d["bbox"]), d["score"])
                if key not in seen:
                    seen.add(key)
                    unique_dlist.append(d)
            dets_by_fn_deduped[fn] = unique_dlist
    else:
        dets_by_fn_deduped = dets_by_fn_evaluated

    # Step 6: Evaluate COCO metrics
    metrics = {"c": None, "cn": None, "n": None}

    if args.classes == 20:
        # 1. cn: original ids (0-19)
        cats_cn = [{"id": i, "name": f"cat_{i}"} for i in range(20)]
        gt_cat_ids_cn = sorted(list({ann["category_id"] for ann in evaluated_gt_annotations}))
        metrics["cn"] = evaluate_coco(
            evaluated_gt_images,
            evaluated_gt_annotations,
            remaining_dets,
            cats_cn,
            gt_cat_ids_cn,
        )

        # 2. c: ids % 10
        cats_c = [{"id": i, "name": f"cat_{i}"} for i in range(10)]
        gt_anns_c = []
        for ann in evaluated_gt_annotations:
            ann_c = dict(ann)
            ann_c["category_id"] = ann["category_id"] % 10
            gt_anns_c.append(ann_c)
        dets_c = []
        for d in remaining_dets:
            dc = dict(d)
            dc["category_id"] = d["category_id"] % 10
            dets_c.append(dc)
        gt_cat_ids_c = sorted(list({ann["category_id"] for ann in gt_anns_c}))
        metrics["c"] = evaluate_coco(
            evaluated_gt_images,
            gt_anns_c,
            dets_c,
            cats_c,
            gt_cat_ids_c,
        )

        # 3. n: ids // 10
        cats_n = [{"id": 0, "name": "real"}, {"id": 1, "name": "reflected"}]
        gt_anns_n = []
        for ann in evaluated_gt_annotations:
            ann_n = dict(ann)
            ann_n["category_id"] = ann["category_id"] // 10
            gt_anns_n.append(ann_n)
        dets_n = []
        for d in remaining_dets:
            dn = dict(d)
            dn["category_id"] = d["category_id"] // 10
            dets_n.append(dn)
        gt_cat_ids_n = sorted(list({ann["category_id"] for ann in gt_anns_n}))
        metrics["n"] = evaluate_coco(
            evaluated_gt_images,
            gt_anns_n,
            dets_n,
            cats_n,
            gt_cat_ids_n,
        )

    elif args.classes == 10:
        # For 10 classes: GT ids become id % 10, detections stay as they are, cn and n are null
        cats_c = [{"id": i, "name": f"cat_{i}"} for i in range(10)]
        gt_anns_c = []
        for ann in evaluated_gt_annotations:
            ann_c = dict(ann)
            ann_c["category_id"] = ann["category_id"] % 10
            gt_anns_c.append(ann_c)
        gt_cat_ids_c = sorted(list({ann["category_id"] for ann in gt_anns_c}))
        metrics["c"] = evaluate_coco(
            evaluated_gt_images,
            gt_anns_c,
            remaining_dets,
            cats_c,
            gt_cat_ids_c,
        )
        metrics["cn"] = None
        metrics["n"] = None

    # Precision, Recall, F1 and threshold tuning (20 classes only)
    pr = None
    conf = None
    pr_status = None

    if args.classes == 20:
        if args.pick_conf:
            best_t = None
            best_mean_f1 = -1.0
            best_pr = None

            for step in range(1, 100):
                t = round(step * 0.01, 2)
                cur_pr = compute_pr_at_threshold(gt_boxes_by_img, dets_by_img, t)
                mean_f1 = (cur_pr["real"]["F1"] + cur_pr["reflected"]["F1"]) / 2.0
                if mean_f1 > best_mean_f1:
                    best_mean_f1 = mean_f1
                    best_t = t
                    best_pr = cur_pr

            conf = best_t
            pr = best_pr
            pr_status = "tuned_and_scored_on_same_set"

        elif args.conf is not None:
            conf = float(args.conf)
            pr = compute_pr_at_threshold(gt_boxes_by_img, dets_by_img, conf)
            pr_status = "fixed_threshold"

    # Collapsed pairs (20 classes only)
    collapsed_pairs = None
    if args.classes == 20:
        cp_all = compute_collapsed_pairs(dets_by_fn_deduped)
        cp_at_conf = None
        if conf is not None:
            dets_by_fn_conf = {
                fn: [d for d in dlist if d["score"] >= conf]
                for fn, dlist in dets_by_fn_deduped.items()
            }
            cp_at_conf = compute_collapsed_pairs(dets_by_fn_conf)

        collapsed_pairs = {
            "all": cp_all,
            "at_conf": cp_at_conf,
            "images_with_pairs": cp_all["images_with_pairs"],
            "pairs": cp_all["pairs"],
        }

    out_obj = {
        "collapsed_pairs": collapsed_pairs,
        "conf": conf,
        "dets_sha256": dets_sha256,
        "gt_ids_renumbered": gt_ids_renumbered,
        "gt_sha256": gt_sha256,
        "metrics": metrics,
        "min_det_score": min_det_score,
        "n_detections": len(remaining_dets),
        "n_gt_boxes": len(evaluated_gt_annotations),
        "n_images": len(evaluated_gt_images),
        "options": {
            "authors": args.authors,
            "classes": args.classes,
            "conf": args.conf,
            "dets": Path(args.dets).name,
            "gt": Path(args.gt).name,
            "images": Path(args.images).name if args.images else None,
            "min_score": min_score,
            "only_images_with_dets": only_images_with_dets,
            "out": Path(args.out).name,
            "pick_conf": args.pick_conf,
            "record_wise": record_wise,
        },
        "pr": pr,
        "pr_status": pr_status,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out_obj, f, sort_keys=True, indent=2)

    # Print a short summary
    print("=" * 60)
    print(f"Scoring complete: {out_path.name}")
    print(f"Images: {len(evaluated_gt_images)}, GT boxes: {len(evaluated_gt_annotations)}, Detections: {len(remaining_dets)}")
    if metrics["c"]:
        print(f"  c  : mAP={metrics['c']['mAP']:.4f}, AP50={metrics['c']['AP50']:.4f}, AP75={metrics['c']['AP75']:.4f}")
    if metrics["cn"]:
        print(f"  cn : mAP={metrics['cn']['mAP']:.4f}, AP50={metrics['cn']['AP50']:.4f}, AP75={metrics['cn']['AP75']:.4f}")
    if metrics["n"]:
        print(f"  n  : mAP={metrics['n']['mAP']:.4f}, AP50={metrics['n']['AP50']:.4f}, AP75={metrics['n']['AP75']:.4f}")
    if pr:
        print(f"  PR (conf={conf}, status={pr_status}):")
        print(f"    Real     : P={pr['real']['precision']:.4f}, R={pr['real']['recall']:.4f}, F1={pr['real']['F1']:.4f} (TP={pr['real']['TP']}, FP={pr['real']['FP']}, FN={pr['real']['FN']})")
        print(f"    Reflected: P={pr['reflected']['precision']:.4f}, R={pr['reflected']['recall']:.4f}, F1={pr['reflected']['F1']:.4f} (TP={pr['reflected']['TP']}, FP={pr['reflected']['FP']}, FN={pr['reflected']['FN']})")
    if collapsed_pairs:
        print(f"  Collapsed pairs (all): {collapsed_pairs['pairs']} pairs across {collapsed_pairs['images_with_pairs']} images")
        if collapsed_pairs["at_conf"]:
            print(f"  Collapsed pairs (at conf): {collapsed_pairs['at_conf']['pairs']} pairs across {collapsed_pairs['at_conf']['images_with_pairs']} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
