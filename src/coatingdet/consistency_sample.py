"""Second-reviewer labelling-consistency check.

Two sub-commands:

``sample``
    Draws a reproducible random sample of annotated images, stratified by
    acquisition subset, and writes a review package: the cropped context around
    every box, an overlay of the original annotation, and a CSV the second
    reviewer fills in. The reviewer is asked, per box, only two things:

        confirmed   1 = the box marks a real instance of the stated class
                    0 = it does not (wrong class, or nothing there)
        redraw      optional corrected box, "x1,y1,x2,y2" in pixels of the
                    exported crop, if the extent is judged wrong

``score``
    Reads the completed CSV plus any redrawn boxes and reports:

        * percentage of boxes confirmed, overall and per class
        * mean and median IoU between original and redrawn boxes
        * percentage of boxes with IoU >= 0.5 / >= 0.7 with the redraw
        * a Cohen's kappa for the binary defect/particle class decision on the
          subset of boxes both reviewers marked as real

The agreement reported for the released annotations was measured with this
script; see the Annotation quality section of the README. Re-run it with a
different seed or sample size to extend the measurement.

The reviewer should be independent of the original annotator, and should decide
the class from the unannotated crop before consulting the annotated one --
otherwise the kappa measures agreement with a prior rather than an independent
judgement.

Examples
--------
    python -m coatingdet.consistency_sample sample --root datasets \
        --n-images 150 --seed 20260101 --out review_package
    # ... second reviewer fills review_package/review_sheet.csv ...
    python -m coatingdet.consistency_sample score --package review_package
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

CLASS_NAMES = {0: "defect", 1: "particle"}
PAD = 128  # px of context around each box in the exported crop


def subset_of(row) -> str:
    for s in ("test_1", "test_2", "test_3"):
        if s in row["path"]:
            return s
    return "train" if row["set"] == "train" else "val"


def iou_xyxy(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cmd_sample(a):
    meta = list(csv.DictReader(open(a.root / "metadata.csv", encoding="utf-8-sig"), delimiter=";"))
    by_subset = defaultdict(list)
    for r in meta:
        lp = a.root / r["path"].lstrip("./").replace("images", "labels") / (
            Path(r["filename"]).stem + ".txt")
        if lp.exists() and lp.read_text().strip():
            by_subset[subset_of(r)].append((r, lp))

    rng = random.Random(a.seed)
    total = sum(len(v) for v in by_subset.values())
    chosen = []
    for s, items in sorted(by_subset.items()):
        k = max(1, round(a.n_images * len(items) / total))
        chosen += [(s, *x) for x in rng.sample(items, min(k, len(items)))]
    rng.shuffle(chosen)
    chosen = chosen[:a.n_images]

    crops = a.out / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    sheet, n_box = [], 0
    for s, r, lp in chosen:
        img = cv2.imread(str(a.root / r["path"].lstrip("./") / r["filename"]))
        if img is None:
            continue
        H, W = img.shape[:2]
        for bi, line in enumerate(lp.read_text().splitlines()):
            p = line.split()
            if len(p) != 5:
                continue
            c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
            x1, y1 = int((xc - bw / 2) * W), int((yc - bh / 2) * H)
            x2, y2 = int((xc + bw / 2) * W), int((yc + bh / 2) * H)
            cx1, cy1 = max(0, x1 - PAD), max(0, y1 - PAD)
            cx2, cy2 = min(W, x2 + PAD), min(H, y2 + PAD)
            crop = img[cy1:cy2, cx1:cx2].copy()
            name = f"{Path(r['filename']).stem}__b{bi}"
            cv2.imwrite(str(crops / f"{name}_clean.jpg"), crop)
            marked = crop.copy()
            cv2.rectangle(marked, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 215, 255), 3)
            cv2.imwrite(str(crops / f"{name}_annotated.jpg"), marked)
            sheet.append({"crop_id": name, "subset": s, "image": r["filename"],
                          "box_index": bi, "original_class": CLASS_NAMES.get(c, c),
                          "orig_x1": x1 - cx1, "orig_y1": y1 - cy1,
                          "orig_x2": x2 - cx1, "orig_y2": y2 - cy1,
                          "confirmed": "", "reviewer_class": "", "redraw_x1y1x2y2": "",
                          "comment": ""})
            n_box += 1

    with open(a.out / "review_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    (a.out / "sampling.json").write_text(json.dumps(
        {"seed": a.seed, "n_images_requested": a.n_images,
         "n_images_drawn": len({s['image'] for s in sheet}), "n_boxes": n_box,
         "root": str(a.root), "stratified_by": "acquisition subset"}, indent=2))
    print(f"review package: {a.out}")
    print(f"  images  {len({s['image'] for s in sheet})}")
    print(f"  boxes   {n_box}")
    print(f"  crops   {crops} (*_clean.jpg for blind review, *_annotated.jpg for reference)")
    print("\nAsk the second reviewer to fill `confirmed` (1/0), `reviewer_class`")
    print("(defect/particle) and optionally `redraw_x1y1x2y2`, then run `score`.")


def kappa(a_lbl, b_lbl):
    n = len(a_lbl)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a_lbl, b_lbl)) / n
    ca, cb = Counter(a_lbl), Counter(b_lbl)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_score(a):
    rows = [r for r in csv.DictReader(open(a.package / "review_sheet.csv"))
            if r["confirmed"].strip() != ""]
    if not rows:
        raise SystemExit("review_sheet.csv has no completed `confirmed` entries")
    conf = [r for r in rows if r["confirmed"].strip() == "1"]
    per_class = defaultdict(lambda: [0, 0])
    for r in rows:
        per_class[r["original_class"]][1] += 1
        if r["confirmed"].strip() == "1":
            per_class[r["original_class"]][0] += 1

    ious = []
    for r in conf:
        s = r["redraw_x1y1x2y2"].strip()
        if not s:
            continue
        try:
            b = [float(v) for v in s.replace(";", ",").split(",")]
        except ValueError:
            continue
        if len(b) == 4:
            ious.append(iou_xyxy([float(r["orig_x1"]), float(r["orig_y1"]),
                                  float(r["orig_x2"]), float(r["orig_y2"])], b))

    both = [r for r in conf if r["reviewer_class"].strip()]
    k = kappa([r["original_class"] for r in both], [r["reviewer_class"].strip() for r in both])

    print(f"boxes reviewed                {len(rows)}")
    print(f"boxes confirmed               {len(conf)}  ({100 * len(conf) / len(rows):.1f} %)")
    for c, (ok, tot) in sorted(per_class.items()):
        print(f"  {c:10s}                  {ok}/{tot}  ({100 * ok / tot:.1f} %)")
    if ious:
        ious.sort()
        mean = sum(ious) / len(ious)
        med = ious[len(ious) // 2]
        print(f"boxes redrawn                 {len(ious)}")
        print(f"  mean IoU original/redraw    {mean:.3f}")
        print(f"  median IoU                  {med:.3f}")
        print(f"  IoU >= 0.5                  {100 * sum(i >= 0.5 for i in ious) / len(ious):.1f} %")
        print(f"  IoU >= 0.7                  {100 * sum(i >= 0.7 for i in ious) / len(ious):.1f} %")
    else:
        print("boxes redrawn                 0 (no IoU agreement computable)")
    if both:
        print(f"class agreement (n={len(both)})       "
              f"{100 * sum(r['original_class'] == r['reviewer_class'].strip() for r in both) / len(both):.1f} %"
              f"   Cohen's kappa = {k:.3f}")
    json.dump({"n_reviewed": len(rows), "n_confirmed": len(conf),
               "pct_confirmed": 100 * len(conf) / len(rows),
               "per_class": {c: {"confirmed": v[0], "total": v[1]} for c, v in per_class.items()},
               "n_redrawn": len(ious),
               "mean_iou": (sum(ious) / len(ious)) if ious else None,
               "cohens_kappa_class": k if both else None},
              open(a.package / "consistency_result.json", "w"), indent=2)
    print("\nwrote", a.package / "consistency_result.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--root", type=Path, default=Path("datasets"))
    s.add_argument("--n-images", type=int, default=150)
    s.add_argument("--seed", type=int, default=20260101)
    s.add_argument("--out", type=Path, default=Path("review_package"))
    s.set_defaults(func=cmd_sample)
    t = sub.add_parser("score")
    t.add_argument("--package", type=Path, default=Path("review_package"))
    t.set_defaults(func=cmd_score)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
