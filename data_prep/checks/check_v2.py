"""Supervisor check for v2 (session-disjoint). Usage from Project/: python3 data_prep/checks/check_v2.py"""
import json, re, os, csv, collections, datetime, subprocess
O = "data_prep/out"; V = f"{O}/v2"; res = []
def ok(n, c, i=""): res.append(("PASS" if c else "FAIL", n, i))
need = ["train.json", "val.json", "test.json", "sessions.csv", "split_stats.csv",
        "test_cls.json", "test_state.json", "val_cls.json", "val_state.json"]
miss = [f for f in need if not os.path.exists(f"{V}/{f}")]
if miss: print("NOT READY, missing:", miss); raise SystemExit(2)
out = subprocess.run(["sha256sum", "-c", "--quiet", "../../data_prep/checks/original_annotations.sha256"], cwd="RODD/annotations", capture_output=True, text=True)
ok("original RODD annotations unchanged", out.returncode == 0)
v1 = subprocess.run(["sha256sum", f"{O}/train.json", f"{O}/val.json", f"{O}/test.json"], capture_output=True, text=True).stdout.split()[::2]
ok("v1 files unchanged", v1 == ["bedc8271d9b4092f1cfd74b3d40c8756e0184f5d2e2314ce167197896584dbbd", "ad5ee1f858a7e200d562d79985bc00420935dec43c76e3f5df1b622dc2d0413a", "73f748639891a350413694c384ff68fe69d1d1f2eac30cc4c9efd2e45ff40c8b"])
gt = collections.defaultdict(set)
for f in ["train", "val", "test"]:
    d = json.load(open(f"{O}/{f}.json")); im = {i["id"]: i["file_name"] for i in d["images"]}
    for a in d["annotations"]: gt[im[a["image_id"]]].add((a["category_id"], tuple(a["bbox"])))
S = {}
for s in ["train", "val", "test"]:
    d = json.load(open(f"{V}/{s}.json")); im = {i["id"]: i["file_name"] for i in d["images"]}
    b = collections.defaultdict(set)
    for a in d["annotations"]: b[im[a["image_id"]]].add((a["category_id"], tuple(a["bbox"])))
    S[s] = b
sp = {f: s for s in S for f in S[s]}
ok("no image in two splits", sum(len(S[s]) for s in S) == len(sp))
ok("same image set as v1 (19,519)", set(sp) == set(gt), f"{len(sp)}")
ok("boxes identical to v1", all(S[s][f] == gt[f] for s in S for f in S[s]))
n = len(sp)
for s, t in [("train", .6), ("val", .1), ("test", .3)]:
    ok(f"{s} fraction ~{t} (±0.05)", abs(len(S[s]) / n - t) <= .05, f"{len(S[s]) / n:.3f}")
tot = collections.Counter(); per = {}
for s in S:
    c = collections.Counter(k for f in S[s] for k, _ in S[s][f]); per[s] = c; tot += c
    ok(f"{s} has all 20 cells", set(c) == set(range(20)), f"{len(c)}")
dev = max((abs(per[s][k] / tot[k] - t) * 100, s, k) for s, t in [("train", .6), ("val", .1), ("test", .3)] for k in tot)
res.append(("INFO", "max per-cell box deviation (pp)", f"{dev[0]:.2f} ({dev[1]}, cat {dev[2]})"))
T = {}
for f in sp:
    m = re.search(r"IMG_(\d{8})_(\d{6})", f)
    if m: T[f] = datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
seq = sorted((t, f) for f, t in T.items()); sess = {}; sid = 0
for i, (t, f) in enumerate(seq):
    if i and (t - seq[i - 1][0]).total_seconds() > 120: sid += 1
    sess[f] = sid
ss = collections.defaultdict(set)
for f, i in sess.items(): ss[i].add(sp[f])
ok("no 120s-session in two splits (recomputed independently)", all(len(v) == 1 for v in ss.values()), f"{sid + 1} sessions")
close = sum(1 for (t1, f1), (t2, f2) in zip(seq, seq[1:]) if sp[f1] != sp[f2] and (t2 - t1).total_seconds() <= 120)
ok("no cross-split phone photos within 120 s", close == 0, f"{close}")
H = {r["file_name"]: r for r in csv.DictReader(open("data_prep/cache/hashes.csv"))}
by = collections.defaultdict(set)
for f, s in sp.items(): by[H[f]["sha256"]].add(s)
ok("no sha256 shared across splits", all(len(v) == 1 for v in by.values()))
G = {r["file_name"]: r["group_id"] for r in csv.DictReader(open(f"{O}/groups.csv"))}
web = [f for f in sp if f not in T]
gs = collections.defaultdict(set)
for f in web: gs[G[f]].add(sp[f])
ok("non-phone v1 groups not split", all(len(v) == 1 for v in gs.values()))
for s in ["val", "test"]:
    mn = min(per[s][k] for k in range(20))
    ok(f"{s}: every class x nature cell has >= 20 boxes", mn >= 20, f"min {mn}")
hv = {f: int(H[f]["dhash"], 16) for f in sp}
buck = collections.defaultdict(list)
for f, h in hv.items():
    for k in range(8): buck[(k, (h >> (8 * k)) & 255)].append(f)
seen = set(); strong = 0
for L in buck.values():
    if len(L) > 400: continue
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            a, b = L[i], L[j]
            if sp[a] == sp[b] or (a, b) in seen: continue
            seen.add((a, b))
            if bin(hv[a] ^ hv[b]).count("1") <= 4 and a in T and b in T and T[a].date() == T[b].date(): strong += 1
ok("no same-day cross-split phone pairs with dHash <= 4 (revisited scenes)", strong == 0, f"{strong}")
ok("report has v2 section", "v2" in open("data_prep/REPORT.md").read())
for st, nm, i in res: print(f"[{st}] {nm}" + (f"  ({i})" if i else ""))
print("SUMMARY:", collections.Counter(r[0] for r in res))
