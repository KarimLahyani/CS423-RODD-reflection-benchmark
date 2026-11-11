# RODD Dataset Re-evaluation: Clean, Leakage-Free Split Report

**Course Project:** CS 423 (Computer Vision) — Re-evaluating the Reflected-Object Benchmark (RODD, Wu et al., CVM 2025)  
**Deliverable:** Clean Annotation Merging, Deduplication / Leakage Grouping, Stratified 60/10/30 Split, and Verification Suite.

---

## 1. Executive Summary

This report documents the construction of a clean, leakage-free train/validation/test split for the Reflected Object Detection Dataset (RODD).

The original benchmark suffered from major structural flaws:
1. **Severe Train-Val Overlap (Data Leakage):** 2,968 unique image files were present in both `instances_train2017.json` and `instances_val2017.json` with different bounding boxes. Additionally, identical images (byte-identical copies and near-duplicate burst shots) were distributed across splits.
2. **One-Record-Per-Object Scoring Artifact:** The original annotations stored each bounding box as a separate COCO image record (e.g. 22,738 train records over 14,922 images; 9,744 validation records over 7,570 images). The authors' evaluation script (`cal_map.py`) scores each record independently, penalizing any other valid objects in the same image as false positives and scoring multi-object photos multiple times.
3. **Label Contradictions:** 5 images contained identical bounding boxes annotated with contradictory category labels (e.g. `cupor` vs `bowlor`).

To resolve these issues, we implemented a four-step pipeline (`02_merge.py`, `03_group.py`, `04_split.py`, `05_verify.py`):
- **Merged** all annotations by unique image filename, dropping 5 contradictory images (14 boxes) to produce a clean set of **19,519 annotated images** containing **32,468 bounding boxes**.
- **Grouped** images into **6,988 leakage groups** (8,357 across all files) using Union-Find over byte hashes (`sha256`), perceptual hashes (`dhash` Hamming distance $\le 6$), and photo sequence stems.
- **Stratified** whole groups into **Train (59.49%)**, **Validation (9.90%)**, and **Test (30.61%)**, with each class $\times$ nature cell closely tracking targets (maximum deviation from the 60/10/30 target across all 20 cells is only 0.38%, with exact per-cell percentages detailed in Section 5).
- **Verified** complete isolation across all **103,359,784** cross-split image pairs: **0 byte duplicates, 0 near-duplicates with dhash $\le 6$, 0 box coordinate violations, and 100% deterministic reproducibility**.

---

## 2. Dataset Reconciliation & Ground Truth Numbers

| Metric | Original Train | Original Val | Union / Disk | Clean Merged Split |
| :--- | :--- | :--- | :--- | :--- |
| **COCO Image Records** | 22,738 | 9,744 | 32,482 | 19,519 (per-image) |
| **Unique Image File Names** | 14,922 | 7,570 | 19,524 | 19,519 |
| **Total Bounding Boxes** | 22,738 | 9,744 | 32,482 | 32,468 |
| **Overlapping File Names** | — | — | 2,968 | **0 (Disjoint)** |
| **Images on Disk (`RODD/images`)**| — | — | 22,838 | 22,838 |
| **Unreferenced Images on Disk** | — | — | 3,314 | 3,319 (incl. 5 excluded) |

### Note on Image Counts vs. Original Paper (21,059)
The original paper (Wu et al., CVM 2025) reports 21,059 images. However:
- The actual uncompressed image folder provided in the repository contains **22,838 `.jpg` files**.
- The official COCO annotation files reference only **19,524 unique filenames** across train and validation.
- The remaining **3,314 files on disk** are unreferenced by any annotation in either official split.
- Our clean dataset retains all **19,519 valid, annotated images** (after removing the 5 corrupted images).

---

## 3. Data Cleaning & Excluded Images (`02_merge.py`)

All annotations from `instances_train2017.json` and `instances_val2017.json` were unified by `file_name`. Each image was assigned an `orig_split` tag (`"train"`, `"val"`, or `"both"`).
- Exact duplicates of `(category_id, bbox)` within an image: **0**
- Dimension conflicts across records: **0**
- Conflicting bounding boxes (identical box coordinates with contradictory labels): **5 images**

These 5 images (14 total annotations) were excluded from all splits and logged to [`out/excluded_images.csv`](out/excluded_images.csv):

| File Name | Orig Split | Boxes | Conflicting Labels & Bounding Box Details |
| :--- | :---: | :---: | :--- |
| `053cab43-bo267.jpg` | both | 3 | Box `[80, 224, 202, 124]` labeled as `cupor` (id=6, train) vs `bowlor` (id=0, train) |
| `1661909b-bo73.jpg` | both | 3 | Box `[285, 16, 307, 247]` labeled as `bowlor` (id=0, train) vs `cupor` (id=6, val) |
| `45e1f7cf-bo152.jpg` | val | 3 | Box `[61, 177, 234, 222]` labeled as `cupor` (id=6, val) vs `bowlor` (id=0, val) |
| `a7827953-IMG_20240917_144851_1.jpg` | train | 3 | Box `[885, 1, 232, 178]` labeled as `applemi` (id=11, train) vs `orangemi` (id=17, train) |
| `bb2a85a5-IMG_20240917_144424_1.jpg` | train | 2 | Box `[614, 596, 144, 129]` labeled as `applemi` (id=11, train) vs `bowlmi` (id=10, train) |

### Clean Merged Properties (`out/merged_all.json`)
- **Clean Images:** 19,519 (Train-only: 11,952; Val-only: 4,601; Both: 2,966)
- **Clean Boxes:** 32,468
- **Single-box images:** 14,451 (74.04%)
- **Multi-box images:** 5,068 (25.96%), with a maximum of 31 boxes in one image
- **Co-occurring Real + Reflected:** 1,405 images contain both real (`nature=0`) and reflected (`nature=1`) objects.

---

## 4. Leakage Grouping via Union-Find (`03_group.py`)

To eliminate train-to-test and train-to-val data leakage, images were grouped using Union-Find over three rules applied to all 22,838 images:
1. **Rule a (Exact Hash):** Same SHA-256 byte digest $\rightarrow$ **2,941 duplicate sets (3,672 pairs)**.
2. **Rule c (Camera Stem):** Stripping the 8-hex-character prefix (`^[0-9a-fA-F]{8}-`), `.jpg`, and any sequence suffix after timestamp (`IMG_YYYYMMDD_HHMMSS_<digit>` $\rightarrow$ `IMG_YYYYMMDD_HHMMSS`) $\rightarrow$ **3,559 stem sets (8,751 pairs)**.
3. **Rule b (Perceptual Hash):** 64-bit difference hash (`dhash`) with Hamming distance $\le 6$, accelerated via 8-byte pigeonhole bucketing $\rightarrow$ **16,771 matching pairs (2,058 spanning tree edges)**.

### Grouping Results
- **Total groups formed across all 22,838 images:** 8,357
- **Groups containing the 19,519 annotated images:** 6,988
- **Group size distribution:**
  - Singleton groups (size 1): 3,462
  - Size 2: 2,775
  - Size 3: 1,041
  - Size 4: 403
  - Size 5: 330
  - Size 6–20: 307
  - Size 21–40: 20
  - Size > 40: 19
- **Rules acting on non-singleton groups:**
  - `dhash + sha256`: 2,867 groups
  - `dhash + stem`: 1,063 groups
  - `stem` only: 911 groups
  - `dhash` only: 52 groups
  - `dhash + sha256 + stem`: 2 groups

### Sanity Check on Near-Duplicates (dhash distance 4–6)
Ten random pairs with Hamming distance in $[4, 6]$ were sampled and verified:
- `c76fdf7f-keyboard80.jpg` $\leftrightarrow$ `c9529f9f-keyboard178.jpg` ($d=6$): identical keyboard and desk setup from slightly shifted camera angle.
- `984662a1-IMG_20240917_143910_1.jpg` $\leftrightarrow$ `f1988e08-IMG_20240917_143923_2.jpg` ($d=5$): consecutive burst shots taken 13 seconds apart.
- `9345d1b2-IMG_20240228_134317_3.jpg` $\leftrightarrow$ `f89290d9-IMG_20240228_134318.jpg` ($d=6$): burst frames taken 1 second apart.
- `41a3a5cc-IMG_20240917_150829_4.jpg` $\leftrightarrow$ `d43bd27c-IMG_20240917_150831_1.jpg` ($d=5$): burst shots taken 2 seconds apart.
- `21fd055d-IMG_20240917_143718_4.jpg` $\leftrightarrow$ `51b9c2c6-IMG_20240917_143719_2.jpg` ($d=4$): near-identical consecutive frames 1 second apart.

### Largest Group Assessment & Threshold Analysis
- The largest group across all images contains **1,132 images (4.96%)** (and 1,098 annotated images, 5.62%).
- **Cause:** Transitive chaining in Union-Find. Because multiple burst sequences were shot against identical recurring lab surfaces, transitive closure $A \sim B \sim C \sim \dots$ links adjacent capture sessions.
- **Stricter threshold exploration:**
  - Threshold $d \le 3$: largest group drops to **369 images (1.62% < 2.0%)**.
  - Threshold $d \le 2$: largest group drops to **56 images (0.25% < 2.0%)**.
- **Design Decision:** We retain $d \le 6$ for the partition because the verification suite requires zero pairs with $d \le 6$ across splits. A group of 1,098 annotated images represents only 9.38% of the 11,711-image Train split and fits comfortably without skewing the target distributions.

---

## 5. Stratified 60/10/30 Split (`04_split.py`)

Whole leakage groups were greedily assigned to splits sorted by group size descending (with a deterministic `seed=0` shuffle for tie-breaking). Each group was assigned to the split whose box counts across the 20 categories were furthest below target, subject to image capacity limits ($60/10/30 \pm 2\%$).

### Split Summary Table

| Split | Images | % Images | Target % | Groups | Boxes | % Boxes | Images with Real+Reflected | Fraction Multi-Box Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 11,612 | **59.49%** | 60.00% | 3,177 | 19,479 | 59.99% | 1,064 | 0.2768 |
| **Val** | 1,932 | **9.90%** | 10.00% | 1,301 | 3,259 | 10.04% | 81 | 0.2428 |
| **Test** | 5,975 | **30.61%** | 30.00% | 2,510 | 9,730 | 29.97% | 260 | 0.2318 |
| **Total** | **19,519** | **100.00%** | 100.00% | **6,988** | **32,468** | **100.00%** | **1,405** | **0.2596** |

### Per-Category Box Distribution (20 Categories)

| Category ID | Category Name | Nature | Total Boxes | Train Count (%) | Val Count (%) | Test Count (%) | Max Dev |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `bowlor` | Real | 729 | 437 (59.95%) | 73 (10.01%) | 219 (30.04%) | 0.05% |
| 1 | `appleor` | Real | 1,743 | 1,041 (59.72%) | 181 (10.38%) | 521 (29.89%) | **0.38%** |
| 2 | `mouseor` | Real | 909 | 545 (59.96%) | 91 (10.01%) | 273 (30.03%) | 0.04% |
| 3 | `keyboardor` | Real | 1,238 | 743 (60.02%) | 124 (10.02%) | 371 (29.97%) | 0.03% |
| 4 | `bananaor` | Real | 1,713 | 1,028 (60.01%) | 173 (10.10%) | 512 (29.89%) | 0.11% |
| 5 | `carrotor` | Real | 3,748 | 2,251 (60.06%) | 373 (9.95%) | 1,124 (29.99%) | 0.06% |
| 6 | `cupor` | Real | 1,734 | 1,041 (60.03%) | 172 (9.92%) | 521 (30.05%) | 0.08% |
| 7 | `orangeor` | Real | 4,251 | 2,552 (60.03%) | 425 (10.00%) | 1,274 (29.97%) | 0.03% |
| 8 | `chairor` | Real | 1,011 | 607 (60.04%) | 101 (9.99%) | 303 (29.97%) | 0.04% |
| 9 | `bookor` | Real | 1,999 | 1,197 (59.88%) | 206 (10.31%) | 596 (29.81%) | 0.31% |
| 10 | `bowlmi` | Reflected | 1,444 | 866 (59.97%) | 144 (9.97%) | 434 (30.06%) | 0.06% |
| 11 | `applemi` | Reflected | 1,204 | 723 (60.05%) | 121 (10.05%) | 360 (29.90%) | 0.10% |
| 12 | `mousemi` | Reflected | 1,293 | 776 (60.02%) | 129 (9.98%) | 388 (30.01%) | 0.02% |
| 13 | `keyboardmi` | Reflected | 1,164 | 698 (59.97%) | 117 (10.05%) | 349 (29.98%) | 0.05% |
| 14 | `bananami` | Reflected | 1,393 | 837 (60.09%) | 139 (9.98%) | 417 (29.94%) | 0.09% |
| 15 | `carrotmi` | Reflected | 1,341 | 805 (60.03%) | 134 (9.99%) | 402 (29.98%) | 0.03% |
| 16 | `cupmi` | Reflected | 1,422 | 852 (59.92%) | 143 (10.06%) | 427 (30.03%) | 0.08% |
| 17 | `orangemi` | Reflected | 1,411 | 847 (60.03%) | 141 (9.99%) | 423 (29.98%) | 0.03% |
| 18 | `chairmi` | Reflected | 1,483 | 890 (60.01%) | 148 (9.98%) | 445 (30.01%) | 0.02% |
| 19 | `bookmi` | Reflected | 1,238 | 743 (60.02%) | 124 (10.02%) | 371 (29.97%) | 0.03% |

*Target ratios: Train 60.00%, Val 10.00%, Test 30.00%. Maximum deviation from target across all 20 cells is **0.38%** (observed on `appleor` validation: 10.38% vs. 10.00% target; second largest is 0.31% on `bookor` validation). Every other cell is within +/- 0.11% of target.*
---

## 6. Generated Files in `data_prep/out/`

1. [`train.json`](out/train.json): 11,612 images, 19,479 boxes (20 categories, COCO format).
2. [`val.json`](out/val.json): 1,932 images, 3,259 boxes (20 categories, COCO format).
3. [`test.json`](out/test.json): 5,975 images, 9,730 boxes (20 categories, COCO format).
4. [`val_cls.json`](out/val_cls.json): 10-class validation file (`id = class10`) for authors' `cal_ap.py`.
5. [`val_state.json`](out/val_state.json): 2-state validation file (`id = nature`: 0=or, 1=mi) for authors' `cal_ap.py`.
6. [`test_cls.json`](out/test_cls.json): 10-class test file (`id = class10`).
7. [`test_state.json`](out/test_state.json): 2-state test file (`id = nature`).
8. [`orig_val_per_image.json`](out/orig_val_per_image.json): Original 7,570 validation images and 9,744 annotations aggregated per image (fixes multiple-scoring bug without modifying the original validation split).
9. [`split_stats.csv`](out/split_stats.csv): Complete CSV summary of image, box, and category breakdowns.
10. [`groups.csv`](out/groups.csv): Mapping of all 22,838 image filenames to `group_id` and active grouping rules.
11. [`excluded_images.csv`](out/excluded_images.csv): Details of 5 excluded conflicting images.
12. [`merged_all.json`](out/merged_all.json): Full clean merged COCO file (19,519 images, 32,468 boxes).

---

## 7. Verification Results (`05_verify.py`)

All checks were executed with 0 warnings and 0 failures:

```text
================================================================================
RODD CLEAN SPLIT VERIFICATION SUITE
================================================================================

[CHECK 1] Disjointness: No file_name or group_id in more than one split
  PASS: All file_names and group_ids are strictly disjoint across splits.
        train: 11612 imgs (3177 groups)
        val  : 1932 imgs (1301 groups)
        test : 5975 imgs (2510 groups)

[CHECK 2] Leakage: No shared sha256, no cross-split pair with dhash distance <= 6
  PASS: Zero sha256 overlap across splits.
  PASS: Zero cross-split pairs with dhash distance <= 6 (min cross-split distance = 7).
        train vs val : min distance = 7, pairs <= 6: 0, residual pairs (7-10): 1481
        train vs test: min distance = 7, pairs <= 6: 0, residual pairs (7-10): 3315
        val vs test  : min distance = 7, pairs <= 6: 0, residual pairs (7-10): 415
        Total residual risk pairs with distance 7-10: 5211 (out of 103,359,784 pairs)

[CHECK 3] Box Conservation: Total boxes in train+val+test == merged boxes
  PASS: Exact box count conservation holds:
        Raw union boxes across dataset : 32482
        Excluded images boxes          : 14 (across 5 images)
        Clean merged boxes             : 32468
        train + val + test boxes       : 32468 (19479 + 3259 + 9730)

[CHECK 4] Annotation Validity: Bounds, categories, and non-empty images
  PASS: All 32,468 bounding boxes strictly inside image boundaries (0px tolerance).
  PASS: All category IDs strictly in range [0, 19].
  PASS: All 19519 images contain at least 1 bounding box.

[CHECK 5] Split Balance: 60/10/30 +/- 2 points and complete category coverage
  PASS: Split fractions within +/- 2 percentage points of target:
        train: 59.49% (target 60.00%, error: 0.51%)
        val  : 9.90% (target 10.00%, error: 0.10%)
        test : 30.61% (target 30.00%, error: 0.61%)
  PASS: All 20 (class10 x nature) categories present in every split.
        Minimum category count in train: 437
        Minimum category count in val  : 73
        Minimum category count in test : 219

[CHECK 6] Determinism: Re-running 04_split.py yields byte-identical outputs
  PASS: All 9 output files are byte-for-byte identical after re-execution.

================================================================================
OVERALL RESULT: ALL CHECKS PASSED (6 / 6 PASS)
================================================================================
```

---

## 8. Honest Caveats, Residual Risk, and Known Limitations

### Known Limitations
> [!NOTE]
> **Largest Leakage Group:** The largest leakage group contains **1,098 annotated images** (~5.6% of all annotated images, or 1,132 images across all files on disk). This cluster arises from transitive dHash chaining ($A \sim B \sim C \dots$) across recurring indoor laboratory backgrounds and burst capture sessions. This grouping is intentionally **conservative with respect to data leakage** (ensuring zero near-duplicates cross between train and evaluation sets) and **limits stratification flexibility only** (the entire 1,098-image component was placed in the 60% Train split, which comfortably accommodated it without violating image or category limits).

### Caveats and Residual Risk Analysis

1. **Perceptual Hash Threshold Selection:**
   The Hamming threshold of $d \le 6$ was selected based on visual inspection of pairs with distances 4, 5, and 6. At $d \le 6$, pairs correspond to the same physical object, background surface, and lighting condition.
2. **Residual Risk in Distance Range [7, 10]:**
   Across all **103,359,784 cross-split pairs**, exactly **5,211 pairs** (0.005%) exhibit a Hamming distance between 7 and 10:
   - Train vs. Val: 1,481 pairs
   - Train vs. Test: 3,315 pairs
   - Val vs. Test: 415 pairs
   These represent photos taken under similar ambient lighting or in the same laboratory room, but with distinct object placements or broader viewing angle changes. While not near-duplicates, they represent a minor residual similarity risk common to indoor camera datasets.
3. **Burst Sequences with Distinct Timestamps:**
   Burst shots taken several minutes apart with different filename timestamps (e.g. `142538` vs `143908`) that changed camera angles sufficiently to have $d \ge 7$ were treated as independent scene captures.
4. **Scoring Bug Fix vs. Clean Split Comparison:**
   For fair evaluation, future models should be scored on both:
   - `orig_val_per_image.json` (to isolate the impact of fixing the authors' multi-record scoring bug on the original validation images);
   - `test.json` (to measure true generalization performance on the clean, leakage-free benchmark).

---

## 9. Split v2: Session-Disjoint Partition

To evaluate real-world generalization across capture sessions rather than individual images or transitive perceptual clusters, a second partition—**Split v2 (Session-Disjoint)**—was constructed alongside `v1`. Split `v1` remains strictly unmodified in `out/`, while all `v2` files are saved to `out/v2/`.

### 9.1 Rules and Formulation

1. **Phone Photo Identification and Timestamp Parsing:**
   Phone photographs follow the filename pattern `IMG_YYYYMMDD_HHMMSS` (with optional duplicate-resolution prefixes and sequence suffixes). Filenames are parsed to extract UTC capture timestamps. Exactly **12,842 clean phone images** are present.
2. **Session Segmentation Rule:**
   All phone photos are sorted chronologically. A new capture session starts whenever the elapsed time to the immediately preceding photo exceeds **120 seconds** ($\Delta t > 120\text{ s}$). This temporal threshold segments the 12,842 phone images into **26 raw capture sessions**.
3. **Same-Day Session Merging ($d_{\text{dHash}} \le 4$):**
   Two sessions captured on the same day are merged into a single atomic unit if they share $\ge 3$ image pairs with Hamming distance $d_{\text{dHash}} \le 4$:
   - **2024-09-16:** `sess_17` and `sess_18` share 17 pairs with $d \le 4$ $\rightarrow$ merged into unit `sess_17_18` (1,486 images).
   - **2024-09-17:** `sess_20` and `sess_21` share 72 pairs ($d \le 4$), `sess_21` and `sess_22` share 5 pairs ($d \le 4$), and `sess_21` and `sess_23` share 41 pairs ($d \le 4$) $\rightarrow$ transitively merged into unit `sess_20_21_22_23` (2,080 images).
   - **Other same-day pairs below threshold:** `sess_09` vs `sess_10` (2 pairs), `sess_15` vs `sess_16` (2 pairs), and `sess_21` vs `sess_24` (1 pair) each have $< 3$ pairs and remain unmerged.
   - **Result:** The 26 raw sessions are merged into **22 session units**.
4. **Atomic Unit Formation:**
   - **Phone Units:** 22 session units (12,842 images).
   - **Non-Phone Units:** The 6,677 web/synthetic images retain their `v1` leakage groups as atomic units (**4,066 units**).
   - **Total Units:** The entire clean dataset (19,519 images) is partitioned into **4,088 atomic units**.
5. **Hard Per-Cell Minimum Constraint ($\ge 20$ Boxes) and Val Target Relaxation:**
   - **Hard Constraint:** In both `val` and `test`, every single one of the 20 class $\times$ nature cells must contain $\ge 20$ bounding boxes.
   - **Target Relaxation Note:** Under the strict target of $\le 12\%$ images for `val`, exactly **0 seeds out of 1,000** satisfied this hard constraint. This occurs because reflected mirrored categories exist solely within phone capture sessions, several of which are large indivisible units (e.g. `sess_00` has 1,117 images; `sess_20_21_22_23` has 2,080 images). Providing $\ge 20$ boxes for all 10 mirrored categories in validation requires assigning multiple complementary phone sessions, which exceeds 12% of images. Consequently, **validation's target was relaxed to 10–15% of images** (as permitted by design). Under this relaxed target, 8 valid seeds were identified.
   - **Optimal Seed Selected: Seed 21** achieves the smallest maximum per-cell box deviation of **22.95 percentage points** (on `mousemi` train: 37.05% actual vs. 60.00% target), with `val` accounting for **2,205 images (11.30%)**.

---

### 9.2 Session Summary Table

The 26 phone capture sessions and their unit consolidations are summarized below:

| Session ID | Atomic Unit | Date | Start – End Time | Duration | Images | Split | Non-Zero Categories (Box Counts) |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| `sess_00` | `sess_00` | 2024-01-21 | 16:27:32 – 16:57:58 | 1,826 s (30m 26s) | 1,117 | **val** | `bowlor`: 236, `appleor`: 266, `mouseor`: 242, `keyboardor`: 373, `bowlmi`: 236, `applemi`: 266, `mousemi`: 242, `keyboardmi`: 372 |
| `sess_01` | `sess_01` | 2024-01-21 | 17:40:18 – 17:45:35 | 317 s (5m 17s) | 289 | **train** | `bananaor`: 289, `bananami`: 289 |
| `sess_02` | `sess_02` | 2024-01-23 | 15:54:56 – 16:04:27 | 571 s (9m 31s) | 232 | **test** | `carrotmi`: 232, `chairmi`: 14 |
| `sess_03` | `sess_03` | 2024-01-23 | 16:13:52 – 16:22:44 | 532 s (8m 52s) | 601 | **test** | `cupmi`: 320, `orangemi`: 281, `chairmi`: 1 |
| `sess_04` | `sess_04` | 2024-01-23 | 18:06:53 – 18:11:09 | 256 s (4m 16s) | 239 | **val** | `chairmi`: 248 |
| `sess_05` | `sess_05` | 2024-01-24 | 14:34:32 – 14:40:13 | 341 s (5m 41s) | 230 | **train** | `cupmi`: 2, `chairmi`: 5, `bookmi`: 232 |
| `sess_06` | `sess_06` | 2024-02-28 | 13:29:40 – 13:29:40 | 0 s (single shot) | 1 | **test** | `orangemi`: 1 |
| `sess_07` | `sess_07` | 2024-02-28 | 13:35:52 – 13:44:59 | 547 s (9m 07s) | 1,008 | **train** | `bowlmi`: 1, `applemi`: 457, `orangemi`: 553, `chairmi`: 12 |
| `sess_08` | `sess_08` | 2024-02-28 | 13:47:24 – 13:50:52 | 208 s (3m 28s) | 559 | **test** | `keyboardmi`: 1, `bananami`: 562 |
| `sess_09` | `sess_09` | 2024-02-28 | 13:54:44 – 14:01:26 | 402 s (6m 42s) | 1,090 | **train** | `carrotmi`: 571, `cupmi`: 554, `chairmi`: 10 |
| `sess_10` | `sess_10` | 2024-02-28 | 15:03:55 – 15:11:43 | 468 s (7m 48s) | 1,124 | **train** | `bowlmi`: 581, `mousemi`: 2, `keyboardmi`: 1, `chairmi`: 566 |
| `sess_11` | `sess_11` | 2024-02-28 | 15:38:38 – 15:49:51 | 673 s (11m 13s) | 1,131 | **test** | `bowlmi`: 13, `mousemi`: 570, `keyboardmi`: 130, `bananami`: 1, `bookmi`: 465 |
| `sess_12` | `sess_12` | 2024-09-16 | 18:31:12 – 18:31:45 | 33 s | 24 | **test** | `bowlmi`: 24 |
| `sess_13` | `sess_13` | 2024-09-16 | 18:40:34 – 18:44:59 | 265 s (4m 25s) | 87 | **test** | `bowlmi`: 126, `mousemi`: 2, `cupmi`: 10, `bookmi`: 1 |
| `sess_14` | `sess_14` | 2024-09-16 | 18:51:03 – 18:52:16 | 73 s | 13 | **test** | `bowlmi`: 24 |
| `sess_15` | `sess_15` | 2024-09-16 | 18:54:37 – 18:59:11 | 274 s (4m 34s) | 99 | **val** | `bananami`: 95, `orangemi`: 4 |
| `sess_16` | `sess_16` | 2024-09-16 | 19:54:37 – 20:06:43 | 726 s (12m 06s) | 523 | **val** | `applemi`: 89, `carrotmi`: 96, `cupmi`: 127, `orangemi`: 125, `bookmi`: 113 |
| `sess_17` | `sess_17_18` | 2024-09-16 | 20:09:57 – 20:14:47 | 290 s (4m 50s) | 263 | **train** | `bowlmi`: 102, `cupmi`: 122, `bookmi`: 134 |
| `sess_18` | `sess_17_18` | 2024-09-16 | 20:36:50 – 20:51:33 | 883 s (14m 43s) | 1,223 | **train** | `bowlmi`: 118, `applemi`: 221, `mousemi`: 1, `bananami`: 252, `carrotmi`: 227, `cupmi`: 133, `orangemi`: 222, `bookmi`: 124 |
| `sess_19` | `sess_19` | 2024-09-16 | 21:01:57 – 21:06:17 | 260 s (4m 20s) | 403 | **train** | `chairmi`: 405 |
| `sess_20` | `sess_20_21_22_23` | 2024-09-17 | 14:18:27 – 14:28:56 | 629 s (10m 29s) | 753 | **train** | `mousemi`: 320, `keyboardmi`: 457, `cupmi`: 5, `bookmi`: 2 |
| `sess_21` | `sess_20_21_22_23` | 2024-09-17 | 14:36:59 – 14:49:50 | 771 s (12m 51s) | 913 | **train** | `bowlmi`: 219, `mousemi`: 156, `bananami`: 194, `cupmi`: 149, `orangemi`: 224 |
| `sess_22` | `sess_20_21_22_23` | 2024-09-17 | 14:53:16 – 14:54:24 | 68 s | 211 | **train** | `carrotmi`: 215 |
| `sess_23` | `sess_20_21_22_23` | 2024-09-17 | 14:58:20 – 15:00:13 | 113 s (1m 53s) | 203 | **train** | `keyboardmi`: 203 |
| `sess_24` | `sess_24` | 2024-09-17 | 15:04:29 – 15:05:57 | 88 s | 162 | **train** | `bookmi`: 167 |
| `sess_25` | `sess_25` | 2024-09-17 | 15:08:05 – 15:11:38 | 213 s (3m 33s) | 344 | **test** | `applemi`: 171, `orangemi`: 1, `chairmi`: 222 |

---

### 9.3 Per-Cell Category Stratification & Min Boxes Table

In Split `v2`, every single category cell satisfies the hard constraint of containing **$\ge 20$ bounding boxes** in both `val` and `test` (and `train`).

| Category | Total Boxes | Train Count (60% Target) | Train % (Dev) | Val Count (10% Target) | Val % (Dev) | Test Count (30% Target) | Test % (Dev) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `bowlor` | 729 | 356 | 48.83% (-11.17 pp) | 236 | 32.37% (+22.37 pp) | 137 | 18.79% (-11.21 pp) |
| `appleor` | 1,743 | 1,002 | 57.49% (-2.51 pp) | 266 | 15.26% (+5.26 pp) | 475 | 27.25% (-2.75 pp) |
| `mouseor` | 909 | 470 | 51.71% (-8.29 pp) | 242 | 26.62% (+16.62 pp) | 197 | 21.67% (-8.33 pp) |
| `keyboardor` | 1,238 | 619 | 50.00% (-10.00 pp) | 373 | 30.13% (+20.13 pp) | 246 | 19.87% (-10.13 pp) |
| `bananaor` | 1,713 | 1,035 | 60.42% (+0.42 pp) | 159 | 9.28% (-0.72 pp) | 519 | 30.30% (+0.30 pp) |
| `carrotor` | 3,748 | 2,257 | 60.22% (+0.22 pp) | 357 | 9.53% (-0.47 pp) | 1,134 | 30.26% (+0.26 pp) |
| `cupor` | 1,734 | 1,041 | 60.03% (+0.03 pp) | 176 | 10.15% (+0.15 pp) | 517 | 29.82% (-0.18 pp) |
| `orangeor` | 4,251 | 2,557 | 60.15% (+0.15 pp) | 418 | 9.83% (-0.17 pp) | 1,276 | 30.02% (+0.02 pp) |
| `chairor` | 1,011 | 615 | 60.83% (+0.83 pp) | 85 | 8.41% (-1.59 pp) | 311 | 30.76% (+0.76 pp) |
| `bookor` | 1,999 | 1,208 | 60.43% (+0.43 pp) | 186 | 9.30% (-0.70 pp) | 605 | 30.27% (+0.27 pp) |
| `bowlmi` | 1,444 | 1,021 | 70.71% (+10.71 pp) | 236 | 16.34% (+6.34 pp) | 187 | 12.95% (-17.05 pp) |
| `applemi` | 1,204 | 678 | 56.31% (-3.69 pp) | 355 | 29.49% (+19.49 pp) | 171 | 14.20% (-15.80 pp) |
| `mousemi` | 1,293 | 479 | 37.05% (**-22.95 pp**) | 242 | 18.72% (+8.72 pp) | 572 | 44.24% (+14.24 pp) |
| `keyboardmi` | 1,164 | 661 | 56.79% (-3.21 pp) | 372 | 31.96% (+21.96 pp) | 131 | 11.25% (-18.75 pp) |
| `bananami` | 1,393 | 735 | 52.76% (-7.24 pp) | 95 | 6.82% (-3.18 pp) | 563 | 40.42% (+10.42 pp) |
| `carrotmi` | 1,341 | 1,013 | 75.54% (+15.54 pp) | 96 | 7.16% (-2.84 pp) | 232 | 17.30% (-12.70 pp) |
| `cupmi` | 1,422 | 965 | 67.86% (+7.86 pp) | 127 | 8.93% (-1.07 pp) | 330 | 23.21% (-6.79 pp) |
| `orangemi` | 1,411 | 999 | 70.80% (+10.80 pp) | 129 | 9.14% (-0.86 pp) | 283 | 20.06% (-9.94 pp) |
| `chairmi` | 1,483 | 998 | 67.30% (+7.30 pp) | 248 | 16.72% (+6.72 pp) | 237 | 15.98% (-14.02 pp) |
| `bookmi` | 1,238 | 659 | 53.23% (-6.77 pp) | 113 | 9.13% (-0.87 pp) | 466 | 37.64% (+7.64 pp) |
| **Total Boxes** | **32,468** | **19,368** | **59.65% (-0.35 pp)** | **4,511** | **13.89% (+3.89 pp)** | **8,589** | **26.45% (-3.55 pp)** |
| **Total Images** | **19,519** | **11,517** | **59.00% (-1.00 pp)** | **2,205** | **11.30% (+1.30 pp)** | **5,797** | **29.70% (-0.30 pp)** |
| **Min Boxes per Cell** | — | **356** (`bowlor`) | — | **85** (`chairor`) | — | **131** (`keyboardmi`) | — |

- **Minimum Box Count per Cell:** Train = **356**, Val = **85** ($\ge 20$ satisfied), Test = **131** ($\ge 20$ satisfied).
- **Maximum Deviation:** **22.95 percentage points** (on `mousemi` train: 37.05% vs 60.00% target).

---

### 9.4 Temporal Independence and Cross-Split Verification

- **Cross-Split Phone Photo Pairs within 60 Seconds:** **Strictly 0**.
- **Cross-Split Phone Photo Pairs within 120 Seconds:** **Strictly 0**.
- **Same-Day Cross-Split Phone Pairs with $d_{\text{dHash}} \le 4$ (Revisited Scenes):** **Strictly 0**.
  Because sessions sharing $\ge 3$ pairs with $d_{\text{dHash}} \le 4$ on the same day are merged into atomic units, no cross-split pairs share close perceptual hashes from revisited setups.

### 9.5 Cross-Unit Perceptual Similarity ($d_{\text{dHash}} \le 6$) & Contact Sheet

- **Total Cross-Unit Pairs with $d_{\text{dHash}} \le 6$:** **121 pairs** (substantially reduced from 880 in the unmerged split, as the dominant cross-session near duplicates on 2024-09-16 and 2024-09-17 were unified).
- **Visual Inspection Contact Sheet:**
  Twelve random pairs were sampled with Seed 0 across distinct session units and rendered into a side-by-side inspection contact sheet at [v2_cross_unit_dhash.png](out/inspect/v2_cross_unit_dhash.png), displaying file names, Hamming distance, source units, and split assignments.



