"""Version-independent AP analysis of the Test-1 failure.

Ultralytics 8.3.x and 8.4.x disagree by an order of magnitude on RT-DETR /
Test 1 even though their raw predictions are byte-identical, so neither number
can be trusted on its own. This script computes AP@0.5 from the predictions
directly, in two modes:

  class-aware      a prediction matches a ground-truth box only if the class
                   also agrees  (what mAP normally means)
  class-agnostic   a prediction matches on geometry alone

A high class-agnostic AP with a near-zero class-aware AP means the objects were
found and localised but assigned the wrong class.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from ultralytics import RTDETR, YOLO

DS = Path(os.environ.get("COATINGDET_ROOT", "datasets"))
ROOT = Path(os.environ.get("COATINGDET_WORK", "."))

SUBSETS = {
    "test_1": DS / "test_1" / "tiles",
    "test_1_wb": ROOT / "runs" / "domain_shift" / "variants" / "test_1_wb",
    "test_2": DS / "test_2" / "tiles",
    "test_3": DS / "test_3" / "tiles",
}
MODELS = {
    "YOLOv11n (released)": (YOLO, os.environ.get("YOLO_W", "runs/yolo11n_20ep/weights/best.pt")),
    "RT-DETR (released, diverged)": (RTDETR, os.environ.get("RTDETR_W", "runs/rtdetr_20ep/weights/best.pt")),
    "RT-DETR (re-trained 20 ep)": (RTDETR, os.environ.get("RTDETR2_W", "runs/rtdetr_20ep/weights/best.pt")),
    "RT-DETR (re-trained 60 ep)": (RTDETR, os.environ.get("RTDETR3_W", "runs/rtdetr_60ep/weights/best.pt")),
}


def iou_mat(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def ap_from(tp, conf, n_gt):
    """101-point interpolated AP, COCO style."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    o = np.argsort(-np.asarray(conf))
    tp = np.asarray(tp, float)[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [rec[-1]]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return float(np.trapz(np.interp(np.linspace(0, 1, 101), mrec, mpre), np.linspace(0, 1, 101)))


def collect(model, tiles: Path, conf=0.001):
    recs, n_gt = [], {0: 0, 1: 0}
    for p in sorted(glob.glob(str(tiles / "images" / "*.jpg"))):
        stem = Path(p).stem
        gb, gc = [], []
        lp = tiles / "labels" / f"{stem}.txt"
        if lp.exists():
            for line in lp.read_text().splitlines():
                q = line.split()
                if len(q) != 5:
                    continue
                c, x, y, w, h = int(q[0]), *map(float, q[1:])
                gb.append([(x - w / 2) * 512, (y - h / 2) * 512, (x + w / 2) * 512, (y + h / 2) * 512])
                gc.append(c); n_gt[c] += 1
        r = model.predict(p, imgsz=512, conf=conf, iou=0.7, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        pc = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
        ps = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        recs.append((pb, pc, ps, np.asarray(gb), np.asarray(gc)))
    return recs, n_gt


def score(recs, n_gt, class_aware, cls=None, thr=0.5):
    tp, cf = [], []
    for pb, pc, ps, gb, gc in recs:
        keep = np.ones(len(pb), bool) if cls is None else (pc == cls)
        pb2, ps2, pc2 = pb[keep], ps[keep], pc[keep]
        gkeep = np.ones(len(gb), bool) if cls is None else (gc == cls)
        gb2, gc2 = (gb[gkeep], gc[gkeep]) if len(gb) else (gb, gc)
        M = iou_mat(pb2, gb2)
        order = np.argsort(-ps2)
        used = set()
        for pi in order:
            best, bi = 0.0, -1
            for gi in range(len(gb2)):
                if gi in used or M[pi, gi] < thr:
                    continue
                if class_aware and pc2[pi] != gc2[gi]:
                    continue
                if M[pi, gi] > best:
                    best, bi = M[pi, gi], gi
            tp.append(1.0 if bi >= 0 else 0.0)
            cf.append(ps2[pi])
            if bi >= 0:
                used.add(bi)
    total = sum(n_gt.values()) if cls is None else n_gt[cls]
    return ap_from(tp, cf, total)


if __name__ == "__main__":
    out = {}
    for label, (cls_, w) in MODELS.items():
        model = cls_(w)
        for sname, tiles in SUBSETS.items():
            recs, n_gt = collect(model, tiles)
            r = {"n_gt": n_gt,
                 "AP50_class_aware": score(recs, n_gt, True),
                 "AP50_class_agnostic": score(recs, n_gt, False)}
            for c in (0, 1):
                if n_gt[c]:
                    r[f"AP50_class{c}_aware"] = score(recs, n_gt, True, cls=c)
            out.setdefault(label, {})[sname] = r
            print(f"{label:30s} {sname:10s} gt={n_gt}  "
                  f"class-aware AP50={r['AP50_class_aware']:.3f}  "
                  f"class-agnostic AP50={r['AP50_class_agnostic']:.3f}", flush=True)
    (ROOT / "results" / "ap_analysis.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", ROOT / "results" / "ap_analysis.json")
