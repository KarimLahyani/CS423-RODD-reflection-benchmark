"""Generate two contact-sheet PNGs in data_prep/figures/:
1. 12 random cross-split image pairs with dHash distance 7-8
2. 12 random same-group pairs with distance 4-6
Each pair side by side, labeled with file names, distance, and splits. Seed 0.
"""
import os
import sys
import json
import csv
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_font(size, bold=False):
    font_paths = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()

def render_contact_sheet(pairs, title, subtitle, out_path, is_same_group=False, img_dir=""):
    """
    Renders 12 pairs in a 6-row x 2-column layout.
    Each pair is displayed side by side with clear labels.
    """
    title_font = get_font(22, bold=True)
    sub_font = get_font(14, bold=False)
    panel_title_font = get_font(14, bold=True)
    label_font = get_font(12, bold=False)
    bold_label_font = get_font(12, bold=True)

    # Dimensions
    panel_w = 780
    panel_h = 320
    thumb_w = 360
    thumb_h = 220
    pad_x = 25
    pad_y = 20
    top_header_h = 90

    cols = 2
    rows = 6
    sheet_w = pad_x * 3 + panel_w * cols
    sheet_h = top_header_h + pad_y * (rows + 1) + panel_h * rows

    # Create background canvas (light neutral)
    canvas = Image.new("RGB", (sheet_w, sheet_h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    # Top Header banner
    draw.rectangle([(0, 0), (sheet_w, top_header_h - 10)], fill=(33, 43, 54))
    draw.text((pad_x, 15), title, fill=(255, 255, 255), font=title_font)
    draw.text((pad_x, 48), subtitle, fill=(180, 195, 210), font=sub_font)

    for idx, pair_info in enumerate(pairs):
        r = idx // cols
        c = idx % cols
        px = pad_x + c * (panel_w + pad_x)
        py = top_header_h + r * (panel_h + pad_y)

        # Draw panel card background
        draw.rounded_rectangle(
            [(px, py), (px + panel_w, py + panel_h)],
            radius=8,
            fill=(255, 255, 255),
            outline=(210, 218, 226),
            width=1
        )

        # Parse pair details
        if is_same_group:
            fn1, fn2, dist, gid, s1 = pair_info
            s2 = s1
            header_text = f"Pair #{idx + 1:02d}  |  dHash Distance: {dist}  |  Group: {gid}  |  Split: {s1.upper()}"
            tag_color = (40, 167, 69) if s1 == "train" else ((0, 123, 255) if s1 == "test" else (255, 153, 0))
        else:
            fn1, fn2, dist, s1, s2 = pair_info
            header_text = f"Pair #{idx + 1:02d}  |  dHash Distance: {dist} (Residual Risk)  |  Cross-Split: {s1.upper()} vs {s2.upper()}"
            tag_color = (220, 53, 69)

        # Panel header banner
        draw.rounded_rectangle(
            [(px, py), (px + panel_w, py + 32)],
            radius=8,
            fill=(240, 244, 248),
            outline=(210, 218, 226),
            width=1
        )
        draw.text((px + 12, py + 7), header_text, fill=(33, 37, 41), font=panel_title_font)

        # Load and place Left Image
        p1 = os.path.join(img_dir, fn1)
        p2 = os.path.join(img_dir, fn2)

        def make_thumb(img_path):
            try:
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
                    # paste centered on background
                    card_thumb = Image.new("RGB", (thumb_w, thumb_h), (235, 238, 242))
                    ox = (thumb_w - im_rgb.width) // 2
                    oy = (thumb_h - im_rgb.height) // 2
                    card_thumb.paste(im_rgb, (ox, oy))
                    return card_thumb
            except Exception as e:
                err_thumb = Image.new("RGB", (thumb_w, thumb_h), (240, 220, 220))
                ed = ImageDraw.Draw(err_thumb)
                ed.text((10, 10), f"Error loading image:\n{e}", fill=(180, 0, 0), font=label_font)
                return err_thumb

        thumb1 = make_thumb(p1)
        thumb2 = make_thumb(p2)

        img1_x = px + 15
        img2_x = px + 15 + thumb_w + 15
        img_y = py + 42

        canvas.paste(thumb1, (img1_x, img_y))
        canvas.paste(thumb2, (img2_x, img_y))

        # Thin border around thumbnails
        draw.rectangle([(img1_x, img_y), (img1_x + thumb_w, img_y + thumb_h)], outline=(200, 205, 215), width=1)
        draw.rectangle([(img2_x, img_y), (img2_x + thumb_w, img_y + thumb_h)], outline=(200, 205, 215), width=1)

        # Sub-labels below each image
        label_y = img_y + thumb_h + 6
        draw.text((img1_x, label_y), f"[{s1.upper()}] ", fill=tag_color, font=bold_label_font)
        s1_w = int(draw.textlength(f"[{s1.upper()}] ", font=bold_label_font))
        draw.text((img1_x + s1_w, label_y), fn1, fill=(70, 80, 95), font=label_font)

        draw.text((img2_x, label_y), f"[{s2.upper()}] ", fill=tag_color, font=bold_label_font)
        s2_w = int(draw.textlength(f"[{s2.upper()}] ", font=bold_label_font))
        draw.text((img2_x + s2_w, label_y), fn2, fill=(70, 80, 95), font=label_font)

    # Save PNG
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)
    print(f"Saved contact sheet: {out_path} ({sheet_w}x{sheet_h} px)")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    rodd_dir = os.path.join(project_root, "RODD")
    rodd_img_dir = os.path.join(rodd_dir, "images")
    out_dir = os.path.join(script_dir, "..", "out")
    cache_dir = os.path.join(script_dir, "..", "cache")
    inspect_dir = os.path.join(out_dir, "inspect")
    os.makedirs(inspect_dir, exist_ok=True)

    hashes_csv = os.path.join(cache_dir, "hashes.csv")
    groups_csv = os.path.join(out_dir, "groups.csv")

    print(f"Loading hashes and groups...")
    hashes = {}
    with open(hashes_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            hashes[r[0]] = (r[1], int(r[2], 16))

    fn_to_gid = {}
    with open(groups_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            fn_to_gid[r[0]] = int(r[1])

    # Load splits
    split_fns = {}
    splits_by_fn = {}
    for s in ["train", "val", "test"]:
        sp_path = os.path.join(out_dir, f"{s}.json")
        with open(sp_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            fns = sorted([img["file_name"] for img in d["images"]])
            split_fns[s] = fns
            for fn in fns:
                splits_by_fn[fn] = s

    # -------------------------------------------------------------
    # 1. 12 random same-group pairs with distance 4-6 (seed 0)
    # -------------------------------------------------------------
    print("Collecting same-group pairs with dHash distance 4-6...")
    from collections import defaultdict
    group_to_fns = defaultdict(list)
    for fn, s in splits_by_fn.items():
        group_to_fns[fn_to_gid[fn]].append(fn)

    same_group_candidates = []
    for gid, fns in group_to_fns.items():
        if len(fns) > 1:
            fns_sorted = sorted(fns)
            for i in range(len(fns_sorted)):
                for j in range(i + 1, len(fns_sorted)):
                    fn1, fn2 = fns_sorted[i], fns_sorted[j]
                    h1 = hashes[fn1][1]
                    h2 = hashes[fn2][1]
                    dist = (h1 ^ h2).bit_count()
                    if 4 <= dist <= 6:
                        same_group_candidates.append((fn1, fn2, dist, gid, splits_by_fn[fn1]))

    same_group_candidates.sort()
    rng_same = random.Random(0)
    sampled_same = rng_same.sample(same_group_candidates, 12)
    print(f"Sampled 12 same-group pairs from {len(same_group_candidates)} candidates.")

    same_png_path = os.path.join(inspect_dir, "same_group_dist_4_6.png")
    render_contact_sheet(
        sampled_same,
        title="RODD Clean Split: 12 Random Same-Group Pairs (dHash Distance 4–6)",
        subtitle="Verification of near-duplicate grouping via Union-Find (Seed 0, all pairs kept in same split)",
        out_path=same_png_path,
        is_same_group=True,
        img_dir=rodd_img_dir
    )

    # -------------------------------------------------------------
    # 2. 12 random cross-split pairs with distance 7-8 (seed 0)
    # -------------------------------------------------------------
    print("Collecting cross-split pairs with dHash distance 7-8...")
    train_fns = split_fns["train"]
    val_fns = split_fns["val"]
    test_fns = split_fns["test"]

    train_h = np.array([hashes[fn][1] for fn in train_fns], dtype=np.uint64)
    val_h = np.array([hashes[fn][1] for fn in val_fns], dtype=np.uint64)
    test_h = np.array([hashes[fn][1] for fn in test_fns], dtype=np.uint64)

    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def get_cross_pairs(h1, fns1, s1, h2, fns2, s2):
        pairs = []
        chunk_size = 500
        n1 = len(h1)
        for start in range(0, n1, chunk_size):
            end = min(start + chunk_size, n1)
            sub1 = h1[start:end]
            xor_res = sub1[:, None] ^ h2[None, :]
            xor_bytes = xor_res.view(np.uint8)
            dist = lut[xor_bytes].reshape(xor_res.shape + (8,)).sum(axis=-1)

            matches = np.argwhere((dist >= 7) & (dist <= 8))
            for r, c in matches:
                fn_a = fns1[start + r]
                fn_b = fns2[c]
                if fn_a > fn_b:
                    pairs.append((fn_b, fn_a, int(dist[r, c]), s2, s1))
                else:
                    pairs.append((fn_a, fn_b, int(dist[r, c]), s1, s2))
        return pairs

    cross_candidates = (
        get_cross_pairs(val_h, val_fns, "val", train_h, train_fns, "train") +
        get_cross_pairs(val_h, val_fns, "val", test_h, test_fns, "test") +
        get_cross_pairs(test_h, test_fns, "test", train_h, train_fns, "train")
    )

    cross_candidates.sort()
    rng_cross = random.Random(0)
    sampled_cross = rng_cross.sample(cross_candidates, 12)
    print(f"Sampled 12 cross-split pairs from {len(cross_candidates)} candidates.")

    cross_png_path = os.path.join(inspect_dir, "cross_split_dist_7_8.png")
    render_contact_sheet(
        sampled_cross,
        title="RODD Clean Split: 12 Random Cross-Split Pairs (dHash Distance 7–8)",
        subtitle="Inspection of residual-risk pairs across Train, Val, and Test splits (Seed 0, zero pairs have distance <= 6)",
        out_path=cross_png_path,
        is_same_group=False,
        img_dir=rodd_img_dir
    )

if __name__ == "__main__":
    main()

