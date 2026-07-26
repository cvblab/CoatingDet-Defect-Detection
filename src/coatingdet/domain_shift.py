"""Illumination-domain-shift control experiment on Test 1.

Test 1 is the only CoatingDet subset acquired under warm illumination and the
only one that contains exclusively particle instances. To separate "the model
cannot see particles" from "the model cannot see particles under this
illumination", the script builds a grey-world white-balanced copy of the Test-1
tiles - pixels only, labels untouched - and evaluates the same checkpoints on
the original and on the corrected copy.

A large jump on the corrected copy isolates the colour cast as the cause. If the
score stays low, the failure is not illumination related and the checkpoint
itself should be inspected (start with runs/<name>/results.csv: the RT-DETR run
behind the first submission diverged to NaN mid-training).

Example
-------
    python -m coatingdet.domain_shift --root datasets \
        --model "YOLOv11n=runs/yolo11n_20ep/weights/best.pt" \
        --model "RT-DETR=runs/rtdetr-l_20ep/weights/best.pt"
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from .evaluate import load
from .whitebalance import grey_world


def build_variant(src: Path, dst: Path) -> int:
    for d in (dst / "images", dst / "labels"):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    n = 0
    for p in sorted((src / "images").glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(dst / "images" / p.name), grey_world(img))
        lbl = src / "labels" / f"{p.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, dst / "labels" / f"{p.stem}.txt")
        n += 1
    return n


def write_yaml(path: Path, root: Path, name: str):
    path.write_text(f"path: {root.resolve()}\ntrain: {name}/images\nval: {name}/images\n"
                    "nc: 2\nnames: ['defect', 'particle']\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("datasets"))
    ap.add_argument("--subset", default="test_1")
    ap.add_argument("--tiles", default="tiles")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=WEIGHTS")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--work", type=Path, default=Path("runs/domain_shift"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    src = a.root / a.subset / a.tiles
    variants = a.work / "variants"
    n = build_variant(src, variants / f"{a.subset}_wb")
    print(f"white-balanced {n} tiles -> {variants / (a.subset + '_wb')}")

    cfg = a.work / "cfg"
    cfg.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg / "orig.yaml", src.parent, a.tiles)
    write_yaml(cfg / "wb.yaml", variants, f"{a.subset}_wb")

    res = {}
    for spec in a.model:
        name, w = spec.split("=", 1)
        for tag, y in (("original", cfg / "orig.yaml"), ("white-balanced", cfg / "wb.yaml")):
            m = load(w).val(data=str(y), imgsz=a.imgsz, conf=a.conf, iou=a.iou, split="val",
                            project=str(a.work), name=f"{name}_{tag}".replace(" ", "_"),
                            exist_ok=True, plots=False, verbose=False)
            res.setdefault(name, {})[tag] = {
                "P": float(m.box.mp), "R": float(m.box.mr),
                "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}

    (a.out / f"domain_shift_{a.subset}.json").write_text(json.dumps(res, indent=2))
    print(f"\n{a.subset}, conf={a.conf}, NMS IoU={a.iou}")
    print(f"{'model':16s}{'variant':16s}{'P':>8s}{'R':>8s}{'mAP50':>8s}{'mAP50-95':>10s}")
    for name, d in res.items():
        for tag, v in d.items():
            print(f"{name:16s}{tag:16s}{v['P']:8.3f}{v['R']:8.3f}{v['mAP50']:8.3f}{v['mAP50_95']:10.3f}")
        o, wbv = d["original"]["mAP50"], d["white-balanced"]["mAP50"]
        print(f"{'':16s}{'-> mAP@0.5 change':16s}{'':8s}{'':8s}{wbv - o:+8.3f}")
    print("\nwrote", a.out / f"domain_shift_{a.subset}.json")


if __name__ == "__main__":
    main()
