"""Formal consistency audit of the released CoatingDet annotations.

This is the automatic half of the annotation-quality evidence: it checks every
property that can be verified without a second human annotator.

  * malformed YOLO records (wrong field count, non-numeric class)
  * coordinates outside [0, 1] or boxes extending beyond the image
  * degenerate boxes (zero width/height)
  * duplicated boxes inside one image (same class, IoU > 0.9)
  * boxes touching the image border (possible truncation)
  * disagreement between the image-level ``label`` of metadata.csv and the
    box-level content of the matching .txt file
  * per-session box geometry, as a coarse indicator of annotator drift

The human half - a second-reviewer check on a random sample - is handled by
``coatingdet.consistency_sample``.

Example
-------
    python -m coatingdet.annotation_audit --root datasets --out results
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def iou_yolo(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("datasets"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    meta = list(csv.DictReader(open(a.root / "metadata.csv", encoding="utf-8-sig"), delimiter=";"))
    rep = {"n_images": len(meta), "n_boxes": 0, "n_border_touching": 0,
           "malformed": [], "out_of_range": [], "degenerate": [], "duplicates": [],
           "missing_label_file": [], "label_box_mismatch": []}
    geom = defaultdict(list)

    for r in meta:
        stem = Path(r["filename"]).stem
        lp = a.root / r["path"].lstrip("./").replace("images", "labels") / f"{stem}.txt"
        if not lp.exists():
            rep["missing_label_file"].append(r["filename"])
            continue
        boxes = []
        for ln, line in enumerate(lp.read_text().splitlines(), 1):
            if not line.strip():
                continue
            p = line.split()
            if len(p) != 5 or not p[0].lstrip("-").isdigit():
                rep["malformed"].append(f"{lp.name}:{ln}")
                continue
            try:
                c, x, y, w, h = int(p[0]), *map(float, p[1:])
            except ValueError:
                rep["malformed"].append(f"{lp.name}:{ln}")
                continue
            rep["n_boxes"] += 1
            if (not all(0 <= v <= 1 for v in (x, y, w, h)) or x - w / 2 < -1e-6
                    or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6):
                rep["out_of_range"].append(f"{lp.name}:{ln}")
            if w <= 1e-4 or h <= 1e-4:
                rep["degenerate"].append(f"{lp.name}:{ln}")
            if min(x - w / 2, y - h / 2) < 0.002 or max(x + w / 2, y + h / 2) > 0.998:
                rep["n_border_touching"] += 1
            boxes.append((c, x, y, w, h))
            geom[r["code"]].append((w, h))

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][0] == boxes[j][0] and iou_yolo(boxes[i][1:], boxes[j][1:]) > 0.9:
                    rep["duplicates"].append(f"{lp.name}:{i}~{j}")

        n0 = sum(1 for b in boxes if b[0] == 0)
        if (n0 > 0) != (r["label"] != "no_defect"):
            rep["label_box_mismatch"].append(
                {"file": r["filename"], "metadata_label": r["label"],
                 "n_defect_boxes": n0, "n_particle_boxes": len(boxes) - n0})

    rep["session_geometry"] = {
        k: {"n_boxes": len(v), "mean_w": round(sum(x for x, _ in v) / len(v), 5),
            "mean_h": round(sum(y for _, y in v) / len(v), 5)}
        for k, v in sorted(geom.items())}
    for k in ("malformed", "out_of_range", "degenerate", "duplicates", "missing_label_file"):
        rep[f"n_{k}"] = len(rep[k])
        rep[k] = rep[k][:50]
    rep["n_label_box_mismatch"] = len(rep["label_box_mismatch"])

    (a.out / "annotation_audit.json").write_text(json.dumps(rep, indent=2))
    print(f"images                     {rep['n_images']:,}")
    print(f"boxes                      {rep['n_boxes']:,}")
    print(f"malformed records          {rep['n_malformed']}")
    print(f"coordinates out of range   {rep['n_out_of_range']}")
    print(f"degenerate boxes           {rep['n_degenerate']}")
    print(f"duplicate boxes (IoU>0.9)  {rep['n_duplicates']}")
    print(f"missing label files        {rep['n_missing_label_file']}")
    print(f"boxes touching the border  {rep['n_border_touching']}")
    print(f"metadata/box mismatches    {rep['n_label_box_mismatch']}"
          f"  ({100 * rep['n_label_box_mismatch'] / rep['n_images']:.2f} % of images)")
    for m in rep["label_box_mismatch"]:
        print("   ", m)
    print("\nwrote", a.out / "annotation_audit.json")


if __name__ == "__main__":
    main()
