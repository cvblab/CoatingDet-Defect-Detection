"""Defect<->particle confusion analysis on Test 3, the only subset containing
both classes.

Predictions are matched greedily to ground truth by IoU (>=0.5), ignoring class,
so that a matched pair with different labels counts as a class confusion rather
than as an independent false positive plus false negative.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import RTDETR, YOLO

DS = Path(os.environ.get("COATINGDET_ROOT", "datasets"))
ROOT = Path(os.environ.get("COATINGDET_WORK", "."))
NAMES = {0: "defect", 1: "particle"}

MODELS = {
    "YOLOv11n (released, 20 ep)": ("yolo", os.environ.get("YOLO_W", "runs/yolo11n_20ep/weights/best.pt")),
    "RT-DETR (released, diverged)": ("rtdetr", os.environ.get("RTDETR_W", "runs/rtdetr_20ep/weights/best.pt")),
    "RT-DETR (re-trained, 20 ep)": ("rtdetr", os.environ.get("RTDETR2_W", "runs/rtdetr_20ep/weights/best.pt")),
}


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def run(kind, weights, tiles: Path, conf):
    model = (YOLO if kind == "yolo" else RTDETR)(weights)
    cnt = Counter()
    for img in sorted((tiles / "images").glob("*.jpg")):
        gt_b, gt_c = [], []
        lp = tiles / "labels" / f"{img.stem}.txt"
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                c, x, y, w, h = int(p[0]), *map(float, p[1:])
                gt_b.append([(x - w / 2) * 512, (y - h / 2) * 512, (x + w / 2) * 512, (y + h / 2) * 512])
                gt_c.append(c)
        r = model.predict(str(img), imgsz=512, conf=conf, iou=0.7, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        pc = [int(v) for v in r.boxes.cls.cpu().numpy()] if r.boxes is not None else []

        M = iou_mat(pb, gt_b)
        used_g, used_p = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0] if M.size else []
        for pi, gi in order:
            if M[pi, gi] < 0.5 or pi in used_p or gi in used_g:
                continue
            used_p.add(int(pi)); used_g.add(int(gi))
            cnt[(NAMES[gt_c[gi]], NAMES[pc[pi]])] += 1
        for gi in range(len(gt_b)):
            if gi not in used_g:
                cnt[(NAMES[gt_c[gi]], "missed")] += 1
        for pi in range(len(pb)):
            if pi not in used_p:
                cnt[("background", NAMES[pc[pi]])] += 1
    return cnt


if __name__ == "__main__":
    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    out = {}
    for label, (kind, w) in MODELS.items():
        if not Path(w).exists():
            continue
        c = run(kind, w, DS / "test_3" / "tiles", conf)
        out[label] = {f"{k[0]}->{k[1]}": v for k, v in sorted(c.items())}
        print(f"\n=== {label}  (Test 3, conf={conf}, NMS IoU=0.7, match IoU>=0.5) ===")
        gt = {"defect": 152, "particle": 784}
        for t in ("defect", "particle"):
            row = {p: c[(t, p)] for p in ("defect", "particle", "missed")}
            n = sum(row.values())
            print(f"  true {t:9s} (n={n:4d}): " + "  ".join(
                f"{p}={v} ({100 * v / max(n, 1):.1f}%)" for p, v in row.items()))
        print(f"  background -> defect  : {c[('background', 'defect')]}")
        print(f"  background -> particle: {c[('background', 'particle')]}")
    (ROOT / "results" / f"confusion_test3_conf{conf}.json").write_text(json.dumps(out, indent=2))
