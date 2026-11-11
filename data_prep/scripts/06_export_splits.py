"""Step 6: export the split definitions as small CSVs (the files committed to git).

The COCO JSONs in out/ are large and regenerable; these CSVs fully define the splits.
Writes splits/v1/split.csv and splits/v2/split.csv (file_name,split) and copies the
supporting tables (groups, sessions, excluded images, split stats).
Usage: python data_prep/scripts/06_export_splits.py
"""
import csv, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
SPLITS = os.path.join(HERE, "..", "splits")

def export(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    rows = []
    for s in ["train", "val", "test"]:
        for im in json.load(open(os.path.join(src_dir, f"{s}.json")))["images"]:
            rows.append((im["file_name"], s))
    rows.sort()
    with open(os.path.join(dst_dir, "split.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file_name", "split"]); w.writerows(rows)
    print(dst_dir, len(rows), "images")

export(OUT, os.path.join(SPLITS, "v1"))
export(os.path.join(OUT, "v2"), os.path.join(SPLITS, "v2"))
for name, dst in [("groups.csv", "v1"), ("excluded_images.csv", ""), ("split_stats.csv", "v1"),
                  ("v2/sessions.csv", "v2"), ("v2/split_stats.csv", "v2")]:
    shutil.copy(os.path.join(OUT, name), os.path.join(SPLITS, dst, os.path.basename(name)))
