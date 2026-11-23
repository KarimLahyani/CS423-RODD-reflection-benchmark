#!/usr/bin/env python3
"""
evaluation/test_score.py

Unit and integration tests for scoring logic and make_gt union logic.
Uses a tiny synthetic ground truth in a temporary directory.
Prints PASS or FAIL per case and exits non-zero on any failure.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Add repo root to sys.path so we can import helper modules if needed
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PYTHON_EXE = sys.executable
SCORE_SCRIPT = str(REPO_ROOT / "evaluation" / "score.py")


def run_scorer(gt_path, dets_path, out_path, classes=20, extra_args=None):
    cmd = [
        PYTHON_EXE,
        SCORE_SCRIPT,
        "--gt", str(gt_path),
        "--dets", str(dets_path),
        "--classes", str(classes),
        "--out", str(out_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc


def create_gt(images, annotations, categories=None):
    if categories is None:
        categories = [{"id": i, "name": f"cat_{i}"} for i in range(20)]
    return {
        "annotations": annotations,
        "categories": categories,
        "images": images,
    }


def main():
    cases_passed = 0
    cases_failed = 0

    def record_result(name, passed, detail=""):
        nonlocal cases_passed, cases_failed
        if passed:
            print(f"[PASS] {name}")
            cases_passed += 1
        else:
            print(f"[FAIL] {name} - {detail}")
            cases_failed += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ---------------------------------------------------------
        # Case 1: Perfect detections: every AP50, AP75 and mAP is 1.0
        # ---------------------------------------------------------
        gt1 = create_gt(
            images=[
                {"id": 0, "file_name": "im1.jpg", "width": 200, "height": 200},
                {"id": 1, "file_name": "im2.jpg", "width": 200, "height": 200},
            ],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
                {"id": 2, "image_id": 0, "category_id": 11, "bbox": [80, 80, 40, 40], "area": 1600, "iscrowd": 0},
                {"id": 3, "image_id": 1, "category_id": 5, "bbox": [20, 20, 30, 30], "area": 900, "iscrowd": 0},
                {"id": 4, "image_id": 1, "category_id": 16, "bbox": [70, 70, 50, 50], "area": 2500, "iscrowd": 0},
            ],
        )
        gt1_path = tmp / "gt1.json"
        gt1_path.write_text(json.dumps(gt1), encoding="utf-8")

        dets1 = [
            {"file_name": "im1.jpg", "category_id": 0, "bbox": [10, 10, 50, 50], "score": 0.99},
            {"file_name": "im1.jpg", "category_id": 11, "bbox": [80, 80, 40, 40], "score": 0.98},
            {"file_name": "im2.jpg", "category_id": 5, "bbox": [20, 20, 30, 30], "score": 0.97},
            {"file_name": "im2.jpg", "category_id": 16, "bbox": [70, 70, 50, 50], "score": 0.96},
        ]
        dets1_path = tmp / "dets1.json"
        dets1_path.write_text(json.dumps(dets1), encoding="utf-8")
        out1_path = tmp / "out1.json"

        res1 = run_scorer(gt1_path, dets1_path, out1_path, classes=20)
        c1_ok = False
        if res1.returncode == 0:
            data1 = json.loads(out1_path.read_text(encoding="utf-8"))
            m = data1["metrics"]
            c1_ok = (
                abs(m["c"]["mAP"] - 1.0) < 1e-4 and abs(m["c"]["AP50"] - 1.0) < 1e-4 and abs(m["c"]["AP75"] - 1.0) < 1e-4 and
                abs(m["cn"]["mAP"] - 1.0) < 1e-4 and abs(m["cn"]["AP50"] - 1.0) < 1e-4 and abs(m["cn"]["AP75"] - 1.0) < 1e-4 and
                abs(m["n"]["mAP"] - 1.0) < 1e-4 and abs(m["n"]["AP50"] - 1.0) < 1e-4 and abs(m["n"]["AP75"] - 1.0) < 1e-4
            )
        record_result("Case 1: Perfect detections (AP50, AP75, mAP == 1.0)", c1_ok, res1.stderr)

        # ---------------------------------------------------------
        # Case 2: Perfect boxes with nature flipped: mAP_c = 1.0, mAP_cn = 0, mAP_n = 0
        # ---------------------------------------------------------
        dets2 = [
            {"file_name": "im1.jpg", "category_id": 10, "bbox": [10, 10, 50, 50], "score": 0.99},
            {"file_name": "im1.jpg", "category_id": 1, "bbox": [80, 80, 40, 40], "score": 0.98},
            {"file_name": "im2.jpg", "category_id": 15, "bbox": [20, 20, 30, 30], "score": 0.97},
            {"file_name": "im2.jpg", "category_id": 6, "bbox": [70, 70, 50, 50], "score": 0.96},
        ]
        dets2_path = tmp / "dets2.json"
        dets2_path.write_text(json.dumps(dets2), encoding="utf-8")
        out2_path = tmp / "out2.json"

        res2 = run_scorer(gt1_path, dets2_path, out2_path, classes=20)
        c2_ok = False
        if res2.returncode == 0:
            data2 = json.loads(out2_path.read_text(encoding="utf-8"))
            m = data2["metrics"]
            c2_ok = (
                abs(m["c"]["mAP"] - 1.0) < 1e-4 and
                abs(m["cn"]["mAP"] - 0.0) < 1e-4 and
                abs(m["n"]["mAP"] - 0.0) < 1e-4
            )
        record_result("Case 2: Nature flipped (mAP_c=1.0, mAP_cn=0, mAP_n=0)", c2_ok, res2.stderr)

        # ---------------------------------------------------------
        # Case 3: Right nature, wrong category: n stays 1.0, c and cn are 0.0
        # ---------------------------------------------------------
        dets3 = [
            # im1 had cat 0 (nature 0) and cat 11 (nature 1)
            # give cat 2 (nature 0) and cat 13 (nature 1)
            {"file_name": "im1.jpg", "category_id": 2, "bbox": [10, 10, 50, 50], "score": 0.99},
            {"file_name": "im1.jpg", "category_id": 13, "bbox": [80, 80, 40, 40], "score": 0.98},
            # im2 had cat 5 (nature 0) and cat 16 (nature 1)
            # give cat 4 (nature 0) and cat 17 (nature 1)
            {"file_name": "im2.jpg", "category_id": 4, "bbox": [20, 20, 30, 30], "score": 0.97},
            {"file_name": "im2.jpg", "category_id": 17, "bbox": [70, 70, 50, 50], "score": 0.96},
        ]
        dets3_path = tmp / "dets3.json"
        dets3_path.write_text(json.dumps(dets3), encoding="utf-8")
        out3_path = tmp / "out3.json"

        res3 = run_scorer(gt1_path, dets3_path, out3_path, classes=20)
        c3_ok = False
        if res3.returncode == 0:
            data3 = json.loads(out3_path.read_text(encoding="utf-8"))
            m = data3["metrics"]
            c3_ok = (
                abs(m["n"]["mAP"] - 1.0) < 1e-4 and
                abs(m["c"]["mAP"] - 0.0) < 1e-4 and
                abs(m["cn"]["mAP"] - 0.0) < 1e-4
            )
        record_result("Case 3: Right nature, wrong category (mAP_n=1.0, mAP_c=0, mAP_cn=0)", c3_ok, res3.stderr)

        # ---------------------------------------------------------
        # Case 4: --classes 10: only c is computed; cn and n are null
        # ---------------------------------------------------------
        dets4 = [
            {"file_name": "im1.jpg", "category_id": 0, "bbox": [10, 10, 50, 50], "score": 0.99},
            {"file_name": "im1.jpg", "category_id": 1, "bbox": [80, 80, 40, 40], "score": 0.98},
            {"file_name": "im2.jpg", "category_id": 5, "bbox": [20, 20, 30, 30], "score": 0.97},
            {"file_name": "im2.jpg", "category_id": 6, "bbox": [70, 70, 50, 50], "score": 0.96},
        ]
        dets4_path = tmp / "dets4.json"
        dets4_path.write_text(json.dumps(dets4), encoding="utf-8")
        out4_path = tmp / "out4.json"

        res4 = run_scorer(gt1_path, dets4_path, out4_path, classes=10)
        c4_ok = False
        if res4.returncode == 0:
            data4 = json.loads(out4_path.read_text(encoding="utf-8"))
            m = data4["metrics"]
            c4_ok = (
                m["c"] is not None and
                abs(m["c"]["mAP"] - 1.0) < 1e-4 and
                m["cn"] is None and
                m["n"] is None and
                data4["pr"] is None and
                data4["collapsed_pairs"] is None
            )
        record_result("Case 4: --classes 10 (only c computed, cn/n are null)", c4_ok, res4.stderr)

        # ---------------------------------------------------------
        # Case 5: --images: scores only the listed images
        # ---------------------------------------------------------
        imgs_list_path = tmp / "images.txt"
        imgs_list_path.write_text("im1.jpg\n", encoding="utf-8")
        out5_path = tmp / "out5.json"

        res5 = run_scorer(gt1_path, dets1_path, out5_path, classes=20, extra_args=["--images", str(imgs_list_path)])
        c5_ok = False
        if res5.returncode == 0:
            data5 = json.loads(out5_path.read_text(encoding="utf-8"))
            c5_ok = (
                data5["n_images"] == 1 and
                data5["n_gt_boxes"] == 2 and
                abs(data5["metrics"]["cn"]["mAP"] - 1.0) < 1e-4
            )
        record_result("Case 5: --images scores only listed images", c5_ok, res5.stderr)

        # ---------------------------------------------------------
        # Case 6: Record-wise: photo with two same-class boxes stored as two records
        # ---------------------------------------------------------
        # Standard per-image GT
        gt6_single = create_gt(
            images=[{"id": 0, "file_name": "photo.jpg", "width": 200, "height": 200}],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
                {"id": 2, "image_id": 0, "category_id": 0, "bbox": [100, 100, 40, 40], "area": 1600, "iscrowd": 0},
            ],
        )
        # Record-wise GT: two records for photo.jpg
        gt6_records = create_gt(
            images=[
                {"id": 0, "file_name": "photo.jpg", "width": 200, "height": 200},
                {"id": 1, "file_name": "photo.jpg", "width": 200, "height": 200},
            ],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 0, "bbox": [100, 100, 40, 40], "area": 1600, "iscrowd": 0},
            ],
        )
        gt6_s_path = tmp / "gt6_single.json"
        gt6_r_path = tmp / "gt6_records.json"
        gt6_s_path.write_text(json.dumps(gt6_single), encoding="utf-8")
        gt6_r_path.write_text(json.dumps(gt6_records), encoding="utf-8")

        dets6 = [
            {"file_name": "photo.jpg", "category_id": 0, "bbox": [10, 10, 40, 40], "score": 0.99},
            {"file_name": "photo.jpg", "category_id": 0, "bbox": [100, 100, 40, 40], "score": 0.98},
        ]
        dets6_path = tmp / "dets6.json"
        dets6_path.write_text(json.dumps(dets6), encoding="utf-8")

        out6_s_path = tmp / "out6_s.json"
        out6_r_path = tmp / "out6_r.json"

        res6_s = run_scorer(gt6_s_path, dets6_path, out6_s_path, classes=20)
        res6_r = run_scorer(gt6_r_path, dets6_path, out6_r_path, classes=20, extra_args=["--record-wise"])

        c6_ok = False
        if res6_s.returncode == 0 and res6_r.returncode == 0:
            d6_s = json.loads(out6_s_path.read_text(encoding="utf-8"))
            d6_r = json.loads(out6_r_path.read_text(encoding="utf-8"))
            map_s = d6_s["metrics"]["c"]["mAP"]
            map_r = d6_r["metrics"]["c"]["mAP"]
            c6_ok = (abs(map_s - 1.0) < 1e-4 and map_r < 1.0)
        record_result("Case 6: Record-wise drops mAP below 1.0 for multi-box image", c6_ok)

        # ---------------------------------------------------------
        # Case 7: --only-images-with-dets: missing detection image lowers AP without flag, not with it
        # ---------------------------------------------------------
        gt7 = create_gt(
            images=[
                {"id": 0, "file_name": "has_det.jpg", "width": 100, "height": 100},
                {"id": 1, "file_name": "no_det.jpg", "width": 100, "height": 100},
            ],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 0, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            ],
        )
        gt7_path = tmp / "gt7.json"
        gt7_path.write_text(json.dumps(gt7), encoding="utf-8")

        dets7 = [
            {"file_name": "has_det.jpg", "category_id": 0, "bbox": [10, 10, 30, 30], "score": 0.99},
        ]
        dets7_path = tmp / "dets7.json"
        dets7_path.write_text(json.dumps(dets7), encoding="utf-8")

        out7_without = tmp / "out7_without.json"
        out7_with = tmp / "out7_with.json"

        res7_without = run_scorer(gt7_path, dets7_path, out7_without, classes=20)
        res7_with = run_scorer(gt7_path, dets7_path, out7_with, classes=20, extra_args=["--only-images-with-dets"])

        c7_ok = False
        if res7_without.returncode == 0 and res7_with.returncode == 0:
            d7_wo = json.loads(out7_without.read_text(encoding="utf-8"))
            d7_w = json.loads(out7_with.read_text(encoding="utf-8"))
            map_wo = d7_wo["metrics"]["c"]["mAP"]
            map_w = d7_w["metrics"]["c"]["mAP"]
            c7_ok = (map_wo < 1.0 and abs(map_w - 1.0) < 1e-4)
        record_result("Case 7: --only-images-with-dets drops zero-det GT images", c7_ok)

        # ---------------------------------------------------------
        # Case 8: --min-score: removes detections at or below value
        # ---------------------------------------------------------
        dets8 = [
            {"file_name": "has_det.jpg", "category_id": 0, "bbox": [10, 10, 30, 30], "score": 0.20},
            {"file_name": "has_det.jpg", "category_id": 0, "bbox": [10, 10, 30, 30], "score": 0.30},
            {"file_name": "has_det.jpg", "category_id": 0, "bbox": [10, 10, 30, 30], "score": 0.40},
        ]
        dets8_path = tmp / "dets8.json"
        dets8_path.write_text(json.dumps(dets8), encoding="utf-8")
        out8_path = tmp / "out8.json"

        res8 = run_scorer(gt7_path, dets8_path, out8_path, classes=20, extra_args=["--min-score", "0.3"])
        c8_ok = False
        if res8.returncode == 0:
            d8 = json.loads(out8_path.read_text(encoding="utf-8"))
            c8_ok = (d8["n_detections"] == 1)
        record_result("Case 8: --min-score removes detections <= S (keeps strictly > S)", c8_ok)

        # ---------------------------------------------------------
        # Case 9: Precision/recall by hand: 2 TP, 1 FP, 1 FN gives 2/3 each; flipped nature gives 1 FP and 1 FN
        # ---------------------------------------------------------
        # Subcase 9A: 3 GT boxes real, 3 detections real (2 matching, 1 FP, 1 GT missed)
        gt9a = create_gt(
            images=[{"id": 0, "file_name": "im9a.jpg", "width": 300, "height": 300}],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 2, "image_id": 0, "category_id": 0, "bbox": [40, 10, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 3, "image_id": 0, "category_id": 0, "bbox": [70, 10, 20, 20], "area": 400, "iscrowd": 0},
            ],
        )
        gt9a_path = tmp / "gt9a.json"
        gt9a_path.write_text(json.dumps(gt9a), encoding="utf-8")

        dets9a = [
            {"file_name": "im9a.jpg", "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.90},
            {"file_name": "im9a.jpg", "category_id": 0, "bbox": [40, 10, 20, 20], "score": 0.80},
            {"file_name": "im9a.jpg", "category_id": 0, "bbox": [150, 10, 20, 20], "score": 0.70},
        ]
        dets9a_path = tmp / "dets9a.json"
        dets9a_path.write_text(json.dumps(dets9a), encoding="utf-8")
        out9a_path = tmp / "out9a.json"

        res9a = run_scorer(gt9a_path, dets9a_path, out9a_path, classes=20, extra_args=["--conf", "0.5"])

        # Subcase 9B: Right box with wrong nature: GT has 1 real box, detection has 1 reflected box at same bbox
        gt9b = create_gt(
            images=[{"id": 0, "file_name": "im9b.jpg", "width": 200, "height": 200}],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            ],
        )
        gt9b_path = tmp / "gt9b.json"
        gt9b_path.write_text(json.dumps(gt9b), encoding="utf-8")

        dets9b = [
            {"file_name": "im9b.jpg", "category_id": 10, "bbox": [10, 10, 20, 20], "score": 0.90},
        ]
        dets9b_path = tmp / "dets9b.json"
        dets9b_path.write_text(json.dumps(dets9b), encoding="utf-8")
        out9b_path = tmp / "out9b.json"

        res9b = run_scorer(gt9b_path, dets9b_path, out9b_path, classes=20, extra_args=["--conf", "0.5"])

        c9_ok = False
        if res9a.returncode == 0 and res9b.returncode == 0:
            d9a = json.loads(out9a_path.read_text(encoding="utf-8"))["pr"]["real"]
            d9b_real = json.loads(out9b_path.read_text(encoding="utf-8"))["pr"]["real"]
            d9b_refl = json.loads(out9b_path.read_text(encoding="utf-8"))["pr"]["reflected"]

            c9a_ok = (
                d9a["TP"] == 2 and d9a["FP"] == 1 and d9a["FN"] == 1 and
                abs(d9a["precision"] - 2.0 / 3.0) < 1e-4 and
                abs(d9a["recall"] - 2.0 / 3.0) < 1e-4 and
                abs(d9a["F1"] - 2.0 / 3.0) < 1e-4
            )
            c9b_ok = (
                d9b_real["TP"] == 0 and d9b_real["FP"] == 0 and d9b_real["FN"] == 1 and
                d9b_refl["TP"] == 0 and d9b_refl["FP"] == 1 and d9b_refl["FN"] == 0
            )
            c9_ok = c9a_ok and c9b_ok
        record_result("Case 9: Precision/recall by hand (2/3 F1 and wrong nature 1 FP + 1 FN)", c9_ok)

        # ---------------------------------------------------------
        # Case 10: Empty detections: all AP 0.0, no crash
        # ---------------------------------------------------------
        empty_dets_path = tmp / "empty_dets.json"
        empty_dets_path.write_text("[]", encoding="utf-8")
        out10_path = tmp / "out10.json"

        res10 = run_scorer(gt1_path, empty_dets_path, out10_path, classes=20)
        c10_ok = False
        if res10.returncode == 0:
            d10 = json.loads(out10_path.read_text(encoding="utf-8"))
            m = d10["metrics"]
            c10_ok = (
                m["c"]["mAP"] == 0.0 and m["c"]["AP50"] == 0.0 and m["c"]["AP75"] == 0.0 and
                m["cn"]["mAP"] == 0.0 and m["cn"]["AP50"] == 0.0 and m["cn"]["AP75"] == 0.0 and
                m["n"]["mAP"] == 0.0 and m["n"]["AP50"] == 0.0 and m["n"]["AP75"] == 0.0
            )
        record_result("Case 10: Empty detections (all AP 0.0, no crash)", c10_ok, res10.stderr)

        # ---------------------------------------------------------
        # Case 11: Invalid input error handling
        # ---------------------------------------------------------
        # 11a: unknown file name
        dets11a = [{"file_name": "ghost.jpg", "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.5}]
        p11a = tmp / "d11a.json"
        p11a.write_text(json.dumps(dets11a), encoding="utf-8")
        r11a = run_scorer(gt1_path, p11a, tmp / "o11a.json", classes=20)

        # 11b: out-of-range category
        dets11b = [{"file_name": "im1.jpg", "category_id": 25, "bbox": [10, 10, 20, 20], "score": 0.5}]
        p11b = tmp / "d11b.json"
        p11b.write_text(json.dumps(dets11b), encoding="utf-8")
        r11b = run_scorer(gt1_path, p11b, tmp / "o11b.json", classes=20)

        # 11c: box beyond image by > 1px
        dets11c = [{"file_name": "im1.jpg", "category_id": 0, "bbox": [190, 10, 20, 20], "score": 0.5}]
        p11c = tmp / "d11c.json"
        p11c.write_text(json.dumps(dets11c), encoding="utf-8")
        r11c = run_scorer(gt1_path, p11c, tmp / "o11c.json", classes=20)

        # 11d: non-positive box width/height
        dets11d = [{"file_name": "im1.jpg", "category_id": 0, "bbox": [10, 10, 0, 20], "score": 0.5}]
        p11d = tmp / "d11d.json"
        p11d.write_text(json.dumps(dets11d), encoding="utf-8")
        r11d = run_scorer(gt1_path, p11d, tmp / "o11d.json", classes=20)

        # 11e: more than 100 detections for one image
        dets11e = [{"file_name": "im1.jpg", "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.5} for _ in range(101)]
        p11e = tmp / "d11e.json"
        p11e.write_text(json.dumps(dets11e), encoding="utf-8")
        r11e = run_scorer(gt1_path, p11e, tmp / "o11e.json", classes=20)

        c11_ok = (
            r11a.returncode != 0 and
            r11b.returncode != 0 and
            r11c.returncode != 0 and
            r11d.returncode != 0 and
            r11e.returncode != 0
        )
        record_result("Case 11: Invalid inputs raise errors", c11_ok)

        # ---------------------------------------------------------
        # Case 12: Slightly shifted boxes: 100x100 GT boxes shifted 20px in x (IoU = 0.667)
        # AP50 = 1.0, AP75 = 0.0, mAP = 0.4
        # ---------------------------------------------------------
        gt12 = create_gt(
            images=[{"id": 0, "file_name": "shift.jpg", "width": 400, "height": 400}],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [50, 50, 100, 100], "area": 10000, "iscrowd": 0},
            ],
        )
        gt12_path = tmp / "gt12.json"
        gt12_path.write_text(json.dumps(gt12), encoding="utf-8")

        dets12 = [
            # Shifted by 20 px in x: IoU = 8000 / 12000 = 0.6667
            {"file_name": "shift.jpg", "category_id": 0, "bbox": [70, 50, 100, 100], "score": 0.99},
        ]
        dets12_path = tmp / "dets12.json"
        dets12_path.write_text(json.dumps(dets12), encoding="utf-8")
        out12_path = tmp / "out12.json"

        res12 = run_scorer(gt12_path, dets12_path, out12_path, classes=20)
        c12_ok = False
        if res12.returncode == 0:
            d12 = json.loads(out12_path.read_text(encoding="utf-8"))
            m = d12["metrics"]["c"]
            c12_ok = (
                abs(m["AP50"] - 1.0) < 1e-4 and
                abs(m["AP75"] - 0.0) < 1e-4 and
                abs(m["mAP"] - 0.4) < 1e-4
            )
        record_result("Case 12: Shifted box IoU 0.667 gives AP50=1.0, AP75=0.0, mAP=0.4", c12_ok, res12.stderr)

        # ---------------------------------------------------------
        # Case 13: collapsed_pairs: apple-real + apple-reflected counts 1 pair; IoU < 0.7 counts 0; different cat counts 0
        # ---------------------------------------------------------
        gt13 = create_gt(
            images=[
                {"id": 0, "file_name": "p1.jpg", "width": 300, "height": 300},
                {"id": 1, "file_name": "p2.jpg", "width": 300, "height": 300},
                {"id": 2, "file_name": "p3.jpg", "width": 300, "height": 300},
            ],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 100, 100], "area": 10000, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100], "area": 10000, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [10, 10, 100, 100], "area": 10000, "iscrowd": 0},
            ],
        )
        gt13_path = tmp / "gt13.json"
        gt13_path.write_text(json.dumps(gt13), encoding="utf-8")

        dets13 = [
            # p1: apple-real (1) + apple-reflected (11) on same box (IoU 1.0 > 0.7) -> 1 pair
            {"file_name": "p1.jpg", "category_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9},
            {"file_name": "p1.jpg", "category_id": 11, "bbox": [10, 10, 100, 100], "score": 0.8},
            # p2: apple-real (1) + apple-reflected (11) with IoU < 0.7 (shifted 50px: IoU = 5000/15000 = 0.33) -> 0 pairs
            {"file_name": "p2.jpg", "category_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9},
            {"file_name": "p2.jpg", "category_id": 11, "bbox": [60, 10, 100, 100], "score": 0.8},
            # p3: apple-real (1) + chair-reflected (18) on same box (different category) -> 0 pairs
            {"file_name": "p3.jpg", "category_id": 1, "bbox": [10, 10, 100, 100], "score": 0.9},
            {"file_name": "p3.jpg", "category_id": 18, "bbox": [10, 10, 100, 100], "score": 0.8},
        ]
        dets13_path = tmp / "dets13.json"
        dets13_path.write_text(json.dumps(dets13), encoding="utf-8")
        out13_path = tmp / "out13.json"

        res13 = run_scorer(gt13_path, dets13_path, out13_path, classes=20)
        c13_ok = False
        if res13.returncode == 0:
            d13 = json.loads(out13_path.read_text(encoding="utf-8"))
            cp = d13["collapsed_pairs"]
            c13_ok = (cp["pairs"] == 1 and cp["images_with_pairs"] == 1)
        record_result("Case 13: collapsed_pairs counts only same-object real+reflected pairs", c13_ok)

        # ---------------------------------------------------------
        # Case 14: make_gt union logic: same class + coordinates counted once
        # ---------------------------------------------------------
        from evaluation.make_gt import get_merged_boxes
        val_b = {"im.jpg": [(1, [10, 10, 50, 50]), (2, [100, 100, 20, 20])]}
        train_b = {"im.jpg": [(1, [10, 10, 50, 50]), (3, [60, 60, 30, 30])]}
        merged = get_merged_boxes("im.jpg", val_b, train_b)
        # Should contain 3 boxes: (1, [10,10,50,50]), (2, [100,100,20,20]), (3, [60,60,30,30])
        c14_ok = len(merged) == 3 and (1, (10, 10, 50, 50)) in merged
        record_result("Case 14: make_gt union logic deduplicates identical class+coords", c14_ok)

        # ---------------------------------------------------------
        # Case 15: GT box with id = 0: mAP_cn = 1.0 by default, mAP_cn = 0.0 with --authors
        # ---------------------------------------------------------
        gt15 = create_gt(
            images=[{"id": 0, "file_name": "im15.jpg", "width": 200, "height": 200}],
            annotations=[
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
            ],
        )
        gt15_path = tmp / "gt15.json"
        gt15_path.write_text(json.dumps(gt15), encoding="utf-8")

        dets15 = [
            {"file_name": "im15.jpg", "category_id": 0, "bbox": [10, 10, 50, 50], "score": 0.99},
        ]
        dets15_path = tmp / "dets15.json"
        dets15_path.write_text(json.dumps(dets15), encoding="utf-8")

        out15_def_path = tmp / "out15_default.json"
        out15_auth_path = tmp / "out15_authors.json"

        res15_def = run_scorer(gt15_path, dets15_path, out15_def_path, classes=20)
        res15_auth = run_scorer(gt15_path, dets15_path, out15_auth_path, classes=20, extra_args=["--authors"])

        c15_ok = False
        if res15_def.returncode == 0 and res15_auth.returncode == 0:
            d15_def = json.loads(out15_def_path.read_text(encoding="utf-8"))
            d15_auth = json.loads(out15_auth_path.read_text(encoding="utf-8"))
            c15_ok = (
                abs(d15_def["metrics"]["cn"]["mAP"] - 1.0) < 1e-4 and
                d15_def.get("gt_ids_renumbered") is True and
                abs(d15_auth["metrics"]["cn"]["mAP"] - 0.0) < 1e-4 and
                d15_auth.get("gt_ids_renumbered") is False
            )
        record_result("Case 15: GT id 0 renumbered by default (mAP_cn=1.0), kept with --authors (mAP_cn=0.0)", c15_ok)

        # ---------------------------------------------------------
        # Case 16: One TP on image with GT and one det on image with no GT at --conf 0.5: real TP = 1, FP = 1
        # ---------------------------------------------------------
        gt16 = create_gt(
            images=[
                {"id": 0, "file_name": "has_gt.jpg", "width": 200, "height": 200},
                {"id": 1, "file_name": "no_gt.jpg", "width": 200, "height": 200},
            ],
            annotations=[
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
            ],
        )
        gt16_path = tmp / "gt16.json"
        gt16_path.write_text(json.dumps(gt16), encoding="utf-8")

        dets16 = [
            {"file_name": "has_gt.jpg", "category_id": 0, "bbox": [10, 10, 50, 50], "score": 0.90},
            {"file_name": "no_gt.jpg", "category_id": 0, "bbox": [20, 20, 40, 40], "score": 0.80},
        ]
        dets16_path = tmp / "dets16.json"
        dets16_path.write_text(json.dumps(dets16), encoding="utf-8")
        out16_path = tmp / "out16.json"

        res16 = run_scorer(gt16_path, dets16_path, out16_path, classes=20, extra_args=["--conf", "0.5"])
        c16_ok = False
        if res16.returncode == 0:
            d16 = json.loads(out16_path.read_text(encoding="utf-8"))
            pr_real = d16["pr"]["real"]
            c16_ok = (pr_real["TP"] == 1 and pr_real["FP"] == 1)
        record_result("Case 16: Detection on image with no GT counts as FP (TP=1, FP=1)", c16_ok)

        # ---------------------------------------------------------
        # Case 17: options.gt, options.dets, options.out and options.images contain no / or \
        # ---------------------------------------------------------
        sub17 = tmp / "subdir17"
        sub17.mkdir(parents=True, exist_ok=True)
        gt17_path = sub17 / "custom_gt.json"
        gt17_path.write_text(json.dumps(gt1), encoding="utf-8")
        dets17_path = sub17 / "custom_dets.json"
        dets17_path.write_text(json.dumps(dets1), encoding="utf-8")
        imgs17_path = sub17 / "custom_images.txt"
        imgs17_path.write_text("im1.jpg\nim2.jpg\n", encoding="utf-8")
        out17_path = sub17 / "custom_out.json"

        res17 = run_scorer(gt17_path, dets17_path, out17_path, classes=20, extra_args=["--images", str(imgs17_path)])
        c17_ok = False
        if res17.returncode == 0:
            d17 = json.loads(out17_path.read_text(encoding="utf-8"))
            opts = d17["options"]
            c17_ok = (
                opts["gt"] == "custom_gt.json" and
                opts["dets"] == "custom_dets.json" and
                opts["out"] == "custom_out.json" and
                opts["images"] == "custom_images.txt" and
                all("/" not in str(opts[k]) and "\\" not in str(opts[k]) for k in ["gt", "dets", "out", "images"])
            )
        record_result("Case 17: options.gt, dets, out, images contain no '/' or '\\'", c17_ok)

        # ---------------------------------------------------------
        # Case 18: make_gt.write_json to a temp file produces bytes with no \r
        # ---------------------------------------------------------
        from evaluation.make_gt import write_json
        sample_ds = {"images": [{"id": 0}], "annotations": [], "categories": []}
        file18_path = tmp / "no_cr_test.json"
        write_json(file18_path, sample_ds)
        file18_bytes = file18_path.read_bytes()
        c18_ok = (b"\r" not in file18_bytes) and (len(file18_bytes) > 0)
        record_result("Case 18: make_gt.write_json produces bytes with no '\\r'", c18_ok)

        # ---------------------------------------------------------
        # Case 19: NaN in a bbox and infinity in a score each raise an error (written with allow_nan=True)
        # ---------------------------------------------------------
        dets19_nan = [
            {"file_name": "im1.jpg", "category_id": 0, "bbox": [float("nan"), 10, 50, 50], "score": 0.99},
        ]
        p19_nan = tmp / "dets19_nan.json"
        p19_nan.write_text(json.dumps(dets19_nan, allow_nan=True), encoding="utf-8")
        res19_nan = run_scorer(gt1_path, p19_nan, tmp / "out19_nan.json", classes=20)

        dets19_inf = [
            {"file_name": "im1.jpg", "category_id": 0, "bbox": [10, 10, 50, 50], "score": float("inf")},
        ]
        p19_inf = tmp / "dets19_inf.json"
        p19_inf.write_text(json.dumps(dets19_inf, allow_nan=True), encoding="utf-8")
        res19_inf = run_scorer(gt1_path, p19_inf, tmp / "out19_inf.json", classes=20)

        c19_ok = (res19_nan.returncode != 0 and res19_inf.returncode != 0)
        record_result("Case 19: NaN in bbox and infinity in score raise errors", c19_ok)

    print("\nTest Summary:")
    print(f"Passed: {cases_passed} / {cases_passed + cases_failed}")
    if cases_failed > 0:
        print(f"FAILED {cases_failed} case(s).")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
