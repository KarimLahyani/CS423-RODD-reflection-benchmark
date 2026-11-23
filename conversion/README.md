# RODD to YOLO Export

Converts the RODD (Reflected-Object Detection Dataset) annotations and images into YOLO format for training and evaluation.

## Prerequisites & Inputs

To run this export, ensure you have a fresh clone of the repository and the original RODD dataset files in place:
- `RODD/annotations/instances_train2017.json`
- `RODD/annotations/instances_val2017.json`
- `RODD/images/<file_name>`
- `data_prep/splits/v1/split.csv`
- `data_prep/splits/v2/split.csv`
- `data_prep/splits/excluded_images.csv`

No files from `data_prep/out/` or `data_prep/cache/` are needed or read.

## Usage

Run the export script from anywhere in the repository:
```bash
python conversion/export_yolo.py
```

## Output Structure

The generated datasets are placed under `datasets/`:
```
datasets/{orig,v1,v2}/{20class,10class}/
    data.yaml
    images/{train,val[,test]}/<file_name>     # OS hard link to RODD/images/<file_name>
    labels/{train,val[,test]}/<stem>.txt
```

- `orig` contains `train` and `val` splits.
- `v1` and `v2` contain `train`, `val`, and `test` splits.
- Both 20-class and 10-class variants have their own dedicated `images/` directory structure to allow Ultralytics YOLO to resolve labels correctly.
- Hard links (`os.link`) are used for images to avoid duplicating disk space, with automatic fallback to file copies (`shutil.copy2`) if hard links fail.
- `data.yaml` specifies an absolute path for `path:` for the machine where the script is executed. Ultralytics resolves relative paths against the working directory rather than the YAML directory. Therefore, each person should run this script on their own machine (or Colab) rather than copying `datasets/`.

## Benchmark Decisions & Notes

- **Excluded Images:** The 5 conflicting images from `data_prep/splits/excluded_images.csv` remain in `orig` because `orig` reproduces the benchmark as originally released (resulting in 22,738 train and 9,744 val label lines). They are excluded from `v1` and `v2`.
- **Validation Set for `orig`:** `orig` `val` is the final scoring set for the original benchmark and must not be used for tuning.
- **10-Class Labels:** 10-class labels match the 20-class labels with each category ID mapped as `id % 10`.
- **Incomplete labels in `orig`:** 2,968 images are in both original annotation files, with some objects labeled only on the other side. This affects 19.9% of `orig` train images and 39.2% of `orig` val images. It is the benchmark as released and is intentionally not fixed here. `v1` and `v2` contain the boxes from both sides.
- **Ultralytics `path:` resolution (read in version 8.4.216):** a relative `path:` is tried against the working directory, then the `datasets_dir` setting, and never the yaml folder. This is why `data.yaml` stores an absolute path.
- **EXIF:** 0 of the 13,652 phone photos in RODD have an orientation tag, so they have no rotation issue.

## Hash Verification

Label hashes are computed across all label files (`labels/<split>/*.txt`) under each label set:
1. Discover all `.txt` files under `datasets/<dataset>/<class_variant>/labels/`.
2. Construct the relative key path using forward slashes (e.g. `labels/train/<stem>.txt`).
3. Sort keys as plain Python strings.
4. Update SHA-256 with `key.encode("utf-8") + b"\n"` followed by the raw bytes of the file.

## Label hashes

Run the script and compare the six lines it prints at the end with these:

```text
orig/10class 3aa0ec3d9d0cde8b6b0edc3aa2e13565d634b9269a0fce44d643c403a1b956cb
orig/20class 6148303ab881c62c7caa8e72ad3bf7953d10aeb506f474d002260a6a11422dc1
v1/10class a5f7443174cba1afaec76f619d496743ee43a386ae65a7d1a9533e6c7afad98c
v1/20class a87258357aacc3bb5964b790a4a35090896627d875dcbf98ec9b0b1407655571
v2/10class eb153a2de7a23f7a1041a7f905ebd0ccc424723d9e08faa57ebb2f510a403ca2
v2/20class bfe4c65512f3521b853994d719c55cac87df57c35021730268858050fa8a7f5b
```
