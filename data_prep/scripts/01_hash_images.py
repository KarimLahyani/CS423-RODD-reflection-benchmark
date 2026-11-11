"""Step 1: hash every image file in RODD/images.

Writes cache/hashes.csv with: file_name, sha256, dhash (64-bit hex), width, height.
- sha256 finds byte-identical copies.
- dhash (difference hash on a 9x8 grayscale thumbnail) finds near-duplicates
  (re-encoded, resized or slightly shifted copies of the same photo).
Read-only on the dataset. Usage: python 01_hash_images.py <RODD_root> <out_csv>
"""
import csv, hashlib, os, sys
from multiprocessing import Pool
from PIL import Image

def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return f"{bits:016x}"

def work(path):
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with Image.open(path) as im:
        w, h = im.size
        im.draft("RGB", (128, 128))  # fast JPEG decode at reduced scale
        dh = dhash(im)
    return os.path.basename(path), sha, dh, w, h

if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    img_dir = os.path.join(root, "images")
    files = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir))
    num_workers = min(16, os.cpu_count() or 4)
    with Pool(num_workers) as p, open(out, "w", newline="") as fo:
        wr = csv.writer(fo); wr.writerow(["file_name", "sha256", "dhash", "width", "height"])
        for i, row in enumerate(p.imap(work, files, chunksize=64), 1):
            wr.writerow(row)
            if i % 2000 == 0: print(f"{i}/{len(files)}", flush=True)
    print("done", len(files))
