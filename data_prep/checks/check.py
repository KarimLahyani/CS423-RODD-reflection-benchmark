"""Independent supervisor check of Gemini's outputs (does NOT trust 05_verify.py).
Usage (from Project/):  python3 data_prep/checks/check.py
Checks whatever stage exists so far; prints PASS/FAIL/SKIP per item."""
import csv, json, os, re, hashlib, collections, subprocess
R = "RODD"; C = "data_prep"; O = f"{C}/out"
res = []
def ok(name, cond, info=""): res.append(("PASS" if cond else "FAIL", name, info))
def skip(name): res.append(("SKIP", name, "file missing"))

# 0. originals untouched
out = subprocess.run(["sha256sum", "-c", "--quiet", f"../../{C}/checks/original_annotations.sha256"],
                     cwd=f"{R}/annotations", capture_output=True, text=True)
ok("original RODD annotation files unchanged", out.returncode == 0, out.stdout.strip()[:200])
ok("RODD/images still has 22838 files", len(os.listdir(f"{R}/images")) == 22838)

# 1. hashes
H = {}
if os.path.exists(f"{C}/cache/hashes.csv"):
    for r in csv.DictReader(open(f"{C}/cache/hashes.csv")): H[r["file_name"]] = r
    ok("hashes.csv complete (22838 rows)", len(H) == 22838, f"{len(H)} rows")
    for fn in sorted(H)[::4000]:  # spot-check sha256 against the real file
        s = hashlib.sha256(open(f"{R}/images/{fn}", "rb").read()).hexdigest()
        ok(f"sha256 spot-check {fn}", s == H[fn]["sha256"])
else: skip("hashes.csv")

# ground truth recomputed from originals
gt = collections.defaultdict(set)
for f in ["instances_train2017.json", "instances_val2017.json"]:
    d = json.load(open(f"{R}/annotations/{f}")); im = {i["id"]: i["file_name"] for i in d["images"]}
    for a in d["annotations"]: gt[im[a["image_id"]]].add((a["category_id"], tuple(a["bbox"])))

def load(p):
    d = json.load(open(p)); im = {i["id"]: i for i in d["images"]}
    boxes = collections.defaultdict(set)
    for a in d["annotations"]: boxes[im[a["image_id"]]["file_name"]].add((a["category_id"], tuple(a["bbox"])))
    return d, im, boxes

# 2. merged
excl = set()
if os.path.exists(f"{O}/excluded_images.csv"):
    excl = {r[0] for r in list(csv.reader(open(f"{O}/excluded_images.csv")))[1:] if r}
    ok("excluded images: few (<=10) and real", 0 < len(excl) <= 10 and excl <= set(gt), f"{len(excl)}")
if os.path.exists(f"{O}/merged_all.json"):
    d, im, bx = load(f"{O}/merged_all.json")
    ok("merged: one entry per file_name", len(im) == len({i['file_name'] for i in im.values()}))
    ok("merged: images = 19524 - excluded", len(bx) == 19524 - len(excl), f"{len(bx)}")
    ok("merged: boxes identical to originals for every kept image",
       all(bx[f] == gt[f] for f in gt if f not in excl) and not (set(bx) & excl))
    ok("merged: class10/nature fields correct",
       all(a.get("class10") == a["category_id"] % 10 and a.get("nature") == a["category_id"] // 10 for a in d["annotations"]))
else: skip("merged_all.json")

# 3. groups
G = {}
if os.path.exists(f"{O}/groups.csv"):
    for r in csv.DictReader(open(f"{O}/groups.csv")): G[r["file_name"]] = r["group_id"]
    kept = set(gt) - excl
    ok("groups: every kept image has a group", kept <= set(G), f"missing {len(kept - set(G))}")
    if H:
        by = collections.defaultdict(set)
        for f in kept & set(G): by[H[f]["sha256"]].add(G[f])
        ok("groups: same sha256 -> same group", all(len(v) == 1 for v in by.values()))
    def stem(f):  # strip label-studio prefix; strip _N only after IMG_<date>_<time> (burst/crop index)
        s = re.sub(r"^[0-9a-f]{8}-", "", os.path.splitext(f)[0])
        m = re.match(r"^(IMG_\d{8}_\d{6})_\d+$", s)
        return m.group(1) if m else s
    by = collections.defaultdict(set)
    for f in kept & set(G): by[stem(f)].add(G[f])
    ok("groups: same source-photo stem -> same group", all(len(v) == 1 for v in by.values()))
    sizes = collections.Counter(G[f] for f in kept & set(G))
    ok("groups: largest group <= 2% of images", max(sizes.values()) <= 0.02 * len(kept), f"max {max(sizes.values())}")
else: skip("groups.csv")

# 4. split
S = {}
if all(os.path.exists(f"{O}/{s}.json") for s in ["train", "val", "test"]):
    for s in ["train", "val", "test"]: S[s] = load(f"{O}/{s}.json")[2]
    names = [set(S[s]) for s in S]
    ok("split: no file in two splits", not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2]))
    allf = names[0] | names[1] | names[2]
    ok("split: covers all kept images", allf == set(gt) - excl, f"{len(allf)}")
    ok("split: boxes identical to originals", all(S[s][f] == gt[f] for s in S for f in S[s]))
    n = len(allf)
    for s, t in [("train", .6), ("val", .1), ("test", .3)]:
        ok(f"split: {s} fraction ~{t}", abs(len(S[s]) / n - t) <= .02, f"{len(S[s]) / n:.3f}")
    for s in S:
        cells = {c for f in S[s] for c, _ in S[s][f]}
        ok(f"split: {s} has all 20 class x nature cells", cells == set(range(20)), f"{len(cells)}")
    if G:
        sg = {s: {G[f] for f in S[s]} for s in S}
        ok("split: no group in two splits", not (sg["train"] & sg["val"] or sg["train"] & sg["test"] or sg["val"] & sg["test"]))
    if H:
        sp = {f: s for s in S for f in S[s]}
        by = collections.defaultdict(set)
        for f, s in sp.items(): by[H[f]["sha256"]].add(s)
        ok("split: no sha256 shared across splits", all(len(v) == 1 for v in by.values()))
        # near-duplicate check with byte buckets (dist<=7 => share a byte position)
        hv = {f: int(H[f]["dhash"], 16) for f in sp}
        buck = collections.defaultdict(list)
        for f, h in hv.items():
            for k in range(8): buck[(k, (h >> (8 * k)) & 255)].append(f)
        cross = collections.Counter()
        seen = set()
        for L in buck.values():
            if len(L) > 400: continue  # skip degenerate buckets (blank-ish images), reported below
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    a, b = L[i], L[j]
                    if sp[a] == sp[b] or (a, b) in seen: continue
                    seen.add((a, b)); dist = bin(hv[a] ^ hv[b]).count("1")
                    if dist <= 10: cross[dist] += 1
        ok("split: no cross-split near-duplicates (dHash <= 6)", sum(v for k, v in cross.items() if k <= 6) == 0,
           "residual 7-10: " + str({k: cross[k] for k in range(7, 11)}))
    for p in ["test_cls.json", "test_state.json", "val_cls.json", "val_state.json", "orig_val_per_image.json"]:
        ok(f"exists {p}", os.path.exists(f"{O}/{p}"))
    if os.path.exists(f"{O}/orig_val_per_image.json"):
        _, im2, b2 = load(f"{O}/orig_val_per_image.json")
        ok("orig_val_per_image: 7570 images, one entry each", len(im2) == 7570 == len(b2), f"{len(im2)}")
else: skip("train/val/test.json")

for st, name, info in res: print(f"[{st}] {name}" + (f"  ({info})" if info else ""))
print("\nSUMMARY:", collections.Counter(r[0] for r in res))
