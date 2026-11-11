# Clean splits for the RODD reflected-object detection benchmark

CS 423 Computer Vision (Fall 2026), Özyeğin University.
Team: Karim Lahyani, Mohamed Elayadi, Ahmad Kosaele.

RODD (Wu et al., CVM 2025) is a benchmark where a detector must say whether each object is
**real** or **reflected**. Our audit of the released annotation files found:

- **Train/val leakage:** 2,968 file names appear in both `instances_train2017.json` and `instances_val2017.json`.
- **Duplicate images:** 2,941 groups of byte-identical files, 1,426 of them spanning train and val.
- **One box per record:** a photo with k objects is stored as k records. The authors' scorer evaluates each record
  separately, so the other objects in the photo count as false positives and the photo is scored k times.
- **Label conflicts:** 5 images where the same box carries two different labels (excluded).
- **Burst-shot leakage:** many phone photos were taken seconds apart; near-identical shots end up in different splits.

This folder builds two clean splits of the 19,519 annotated images (32,468 boxes; 10 classes × {real, reflected}).

| Split | Rule | Definition |
|---|---|---|
| `v1` (deduplicated) | Same image, exact copy (SHA-256), near-duplicate (dHash ≤ 6) or same source photo → same split | `data_prep/splits/v1/split.csv` |
| `v2` (session-disjoint) | Whole capture sessions (gap > 120 s; same-day revisits merged) stay in one split | `data_prep/splits/v2/split.csv` |

Both are ~60 / 10 / 30 train / val / test. Full numbers, method and limitations: [`data_prep/REPORT.md`](data_prep/REPORT.md).

## Layout

```
data_prep/
  scripts/   01_hash_images → 02_merge → 03_group → 04_split → build_v2_split → 05_verify → 06_export_splits
  checks/    independent checks (check.py for v1, check_v2.py for v2) + SHA-256 of the original annotation files
  splits/    split definitions (CSV) and supporting tables (groups, sessions, excluded images, stats)
  figures/   inspection sheets of near-duplicate / cross-split image pairs
  REPORT.md  full data report
docs/        project proposal
```

## Reproduce

1. Download RODD from [github.com/jirouvan/ROD](https://github.com/jirouvan/ROD) and place it at `RODD/`
   (`RODD/images/`, `RODD/annotations/`).
2. Check you have the same annotation files:
   `cd RODD/annotations && sha256sum -c ../../data_prep/checks/original_annotations.sha256`
3. From the repository root:
   ```
   pip install -r requirements.txt
   python data_prep/scripts/01_hash_images.py RODD data_prep/cache/hashes.csv
   python data_prep/scripts/02_merge.py
   python data_prep/scripts/03_group.py
   python data_prep/scripts/04_split.py
   python data_prep/scripts/build_v2_split.py
   python data_prep/scripts/05_verify.py
   python data_prep/scripts/06_export_splits.py
   python data_prep/checks/check.py
   python data_prep/checks/check_v2.py
   ```
   Generated COCO files go to `data_prep/out/` (git-ignored, regenerable).

## Credits

Dataset and original code: Y. Wu, Z. Wang, Y. Wu, L. Huang, H. Zhou, S. Li,
"Towards Reflected Object Detection: A Benchmark", CVM 2025 (arXiv:2407.05575),
[github.com/jirouvan/ROD](https://github.com/jirouvan/ROD) (GPL-3.0). Used for coursework only.
