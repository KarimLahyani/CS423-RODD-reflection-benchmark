#!/usr/bin/env python3
"""
evaluation/predict.py

Runs a trained Ultralytics YOLO model on a folder of images and writes detections
in the format read by evaluation/score.py.

Usage:
  predict.py --weights W.pt --images DIR --out FILE.json
             [--imgsz 640] [--conf 0.001] [--max-det 100] [--device 0] [--batch 16]

Output: a JSON list of
  {"file_name": str, "category_id": int, "bbox": [x, y, w, h], "score": float}
- file_name:   image file name inside DIR (not recursive).
- category_id: the model's class index, 0-19 for M20 and 0-9 for M10.
- bbox:        top-left x, y, width, height in pixels of the original image,
               clipped to the image, rounded to 2 decimals.
- score:       confidence, rounded to 5 decimals.
At most --max-det (<= 100) detections per image, highest scores first.
Images with no detections have no entries.
The file is written to FILE.json.part and then renamed to FILE.json,
so a crashed run never leaves a complete-looking file.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_DET_LIMIT = 100  # score.py rejects more than 100 detections per image


def parse_args():
    p = argparse.ArgumentParser(description="Write YOLO detections for score.py")
    p.add_argument("--weights", required=True, help="trained weights (.pt)")
    p.add_argument("--images", required=True, help="folder of images (not recursive)")
    p.add_argument("--out", required=True, help="output detections JSON file")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--device", default=None, help="e.g. 0, cpu (default: Ultralytics picks)")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    if not 1 <= args.max_det <= MAX_DET_LIMIT:
        p.error(f"--max-det must be between 1 and {MAX_DET_LIMIT}")
    if not 0.0 <= args.conf < 1.0:
        p.error("--conf must be in [0, 1)")
    if args.batch < 1:
        p.error("--batch must be >= 1")
    return args


def list_images(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"--images is not a folder: {folder}")
    files = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


def to_detections(result, file_name, n_classes, max_det):
    """Convert one Ultralytics Results object to score.py detections."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    img_h, img_w = result.orig_shape
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    dets = []
    for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
        cat = int(cls)
        if not 0 <= cat < n_classes:
            raise ValueError(f"Class {cat} out of range for a {n_classes}-class model in {file_name}")
        x1 = min(max(float(x1), 0.0), img_w)
        y1 = min(max(float(y1), 0.0), img_h)
        x2 = min(max(float(x2), 0.0), img_w)
        y2 = min(max(float(y2), 0.0), img_h)
        x, y = round(x1, 2), round(y1, 2)
        w, h = round(x2 - x1, 2), round(y2 - y1, 2)
        if w <= 0 or h <= 0:  # score.py rejects empty boxes
            continue
        dets.append({"file_name": file_name, "category_id": cat,
                     "bbox": [x, y, w, h], "score": round(float(score), 5)})

    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets[:max_det]


def main():
    args = parse_args()
    from ultralytics import YOLO

    images = list_images(Path(args.images))
    model = YOLO(args.weights)
    n_classes = len(model.names)
    if n_classes not in (10, 20):
        raise ValueError(f"Expected a 10- or 20-class model, got {n_classes} classes: {args.weights}")
    print(f"[predict] {args.weights}: {n_classes} classes, {len(images)} images in {args.images}")

    predict_kwargs = dict(imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                          stream=True, verbose=False)
    if args.device is not None:
        predict_kwargs["device"] = args.device

    all_dets = []
    n_with_dets = 0
    t0 = time.time()
    for start in range(0, len(images), args.batch):
        chunk = images[start:start + args.batch]
        results = model.predict([str(f) for f in chunk], **predict_kwargs)
        for f, result in zip(chunk, results):
            dets = to_detections(result, f.name, n_classes, args.max_det)
            if dets:
                n_with_dets += 1
            all_dets.extend(dets)
        done = start + len(chunk)
        if done % (args.batch * 50) < args.batch or done == len(images):
            print(f"[predict] {done}/{len(images)} images, {time.time() - t0:.0f} s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with open(part, "w", encoding="utf-8") as fh:
        json.dump(all_dets, fh)
    os.replace(part, out)
    print(f"[predict] wrote {len(all_dets)} detections for {n_with_dets}/{len(images)} images to {out}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"[predict] error: {e}", file=sys.stderr)
        sys.exit(1)
