# RODD Evaluation Suite

Scoring scripts and ground-truth generation for the RODD reflected-object benchmark.

## Commands

### Ground-truth generation
```bash
.venv/Scripts/python.exe evaluation/make_gt.py --all
```
Builds all 7 ground-truth JSON files under `datasets/gt/` (`orig_val.json`, `orig_val_union.json`, `orig_val_records.json`, `v1_val.json`, `v1_test.json`, `v2_val.json`, `v2_test.json`). Always uses 20-class IDs.

### Scoring
```bash
.venv/Scripts/python.exe evaluation/score.py --gt GT.json --dets DETS.json --classes {20,10} --out OUT.json \
    [--images LIST.txt] [--record-wise] [--min-score S] [--only-images-with-dets] [--authors] \
    [--conf T | --pick-conf]
```

### Tests
```bash
.venv/Scripts/python.exe evaluation/test_score.py
```
Runs 19 synthetic test cases covering all scoring metrics, matching rules, error conditions, and edge cases.

---

## Detections Format and `predict.py` Requirements

The detections file must be a JSON list of objects:
```json
[
  {
    "file_name": "image_name.jpg",
    "category_id": 0,
    "bbox": [x, y, w, h],
    "score": 0.8542
  }
]
```
- `file_name`: Image file name matching the ground truth.
- `category_id`: YOLO class integer ID: `0..19` for 20 classes (`--classes 20`), or `0..9` for 10 classes (`--classes 10`).
- `bbox`: `[x, y, w, h]` bounding box in absolute pixels of the original image (`x, y` are top-left corner coordinates).
- `score`: Detection confidence score (float).

**`predict.py` requirements:**
- Must export with confidence threshold `conf=0.001` and at most 100 detections per image, because mAP calculation requires the full score distribution.
- Boxes must not exceed image boundaries by more than 1 pixel (`x < -1`, `y < -1`, `x + w > W + 1`, `y + h > H + 1`).
- Width and height must be strictly positive (`w > 0`, `h > 0`).

---

## Metrics and IoU Levels

Evaluated via `pycocotools` `COCOeval(iouType="bbox")`:
- **`cn` (Category and Nature)**: Evaluated on original 20 classes (0-9 real, 10-19 reflected).
- **`c` (Category only)**: Evaluated by replacing both ground-truth and detection IDs with `id % 10` (10 classes: 0-9).
- **`n` (Nature only)**: Evaluated by replacing both ground-truth and detection IDs with `id // 10` (2 classes: 0 real, 1 reflected).
- When `--classes 10` is passed, ground-truth IDs become `id % 10`, detection IDs remain 0-9, only `c` is evaluated, and `cn` and `n` are reported as `null`.

For each computed metric, `score.py` extracts from `eval["precision"]` with area `all` and max detections 100:
- **`AP50`**: Average Precision at IoU threshold 0.50.
- **`AP75`**: Average Precision at IoU threshold 0.75.
- **`mAP`**: Mean Average Precision averaged over 10 IoU thresholds `0.50:0.05:0.95`.
- **`per_class_mAP`** and **`per_class_AP50`**: Per-class AP at IoU 0.50:0.95 and 0.50.
- **`n_classes_scored`**: Number of classes with at least one ground-truth box in the evaluated images.

---

## No-Merging Default and `collapsed_pairs`

When collapsing 20-class detections to category-only (`c`), detections are not merged and NMS is not re-run.

`collapsed_pairs` reports unordered pairs of detections in the same photo that share the same object category (`id % 10`) but have different nature (`id // 10`) and overlap with IoU `> 0.7` (e.g., apple-real + apple-reflected on the same object). These pairs become same-class duplicate predictions in `c`.

The output reports both `pairs` (total pair count) and `images_with_pairs` (number of photos containing such pairs) for all evaluated detections, and for detections with `score >= T` when a confidence threshold is specified. In `--record-wise` mode, each photo is counted only once.

---

## Precision, Recall, and F1 Matching Rule

Evaluated for 20 classes only on detections with `score >= T`:
1. Per image, detections are sorted by confidence score in descending order (stable sort).
2. Each detection is greedily matched to the unmatched ground-truth box with the **exact same 20-class ID** and the highest IoU, provided IoU `>= 0.5`. Each ground-truth box matches at most one detection.
3. A matched detection is a True Positive (**TP**).
4. An unmatched detection is a False Positive (**FP**) for the nature of its class (`id // 10`).
5. An unmatched ground-truth box is a False Negative (**FN**) for the nature of its class (`id // 10`). A box with the correct coordinates but wrong nature produces 1 FP and 1 FN.

### Threshold Procedure
- `--conf T`: Evaluates precision, recall, and F1 at fixed threshold `T`, reported with `pr_status = "fixed_threshold"`.
- `--pick-conf`: Sweeps `T = 0.01, 0.02, ..., 0.99` and selects the threshold that maximizes the mean of real F1 and reflected F1 (breaking ties by choosing the lowest `T`). Reported with `pr_status = "tuned_and_scored_on_same_set"`. Tune on `val`, then pass the resulting `T` to `test` using `--conf`.
- If neither flag is provided, `pr`, `conf`, and `pr_status` are `null`.
- **Orig caveat**: `orig` has no separate test split. Its numbers are produced by `--pick-conf` directly on `orig_val` and remain labeled `tuned_and_scored_on_same_set`.

---

## Ground-Truth Union Logic (`orig_val_union`: 17,801 Boxes)

In the original benchmark, 2,968 images are shared between the train and val annotation files.
`orig_val_union` merges all boxes for the 7,570 val images by taking the **set** of `(category_id, bbox)` across the val and train files. A bounding box with the identical class and pixel coordinates on both sides is counted exactly once:
$$\text{9,744 val boxes} + \text{8,057 train boxes of shared images} - \text{0 exact duplicates} = \text{17,801 boxes}.$$

Conflicting annotations on shared images are preserved: for example, image `1661909b-bo73.jpg` contains the same coordinates labeled `bowlor` (class 0) in train and `cupor` (class 6) in val. Since their class IDs differ, both entries remain in `orig_val_union`.

---

## Authors' Scorer Differences (`--authors`)

Passing `--authors` applies `--record-wise --min-score 0.3 --only-images-with-dets`, matching the published code (`cal_ap.py`, `cal_map.py`):
1. **Record-wise scoring**: Each original annotation record is scored independently. For photos containing multiple ground-truth boxes, each detection is copied to every record of the photo, creating false-positive penalties.
2. **Score thresholding (`min-score 0.3`)**: Keeps only detections with confidence `score > 0.3`.
3. **Dropped un-detected images (`only-images-with-dets`)**: Ground-truth records with zero detections above 0.3 are removed from evaluation (`params.imgIds` is set to images with detections), so missed ground-truth boxes in those images are never penalized.
4. **Annotation ID 0 unmatched**: The original val file has one box with annotation ID 0, which `pycocotools` can never match. `--authors` keeps this behaviour to reproduce the published code, and every other mode renumbers IDs so that box is scored normally.
