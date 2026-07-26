"""Regenerate the CoatingDet dataset-composition table (Table 2) from the data.

Every number in the table is recomputed from ``metadata.csv`` and the YOLO label
files, so the table can never drift from the released data. Column definitions:

  Images            rows of metadata.csv belonging to the subset
  Defective Images  images with >=1 *critical defect* box (class 0). Images that
                    contain only particle boxes (class 1) are NOT counted here.
  Particle Images   images with >=1 particle box and no defect box
  Tiles             tiles produced by tile_creator.py with the published defaults
  Instances         boxes surviving the tiling filter, split per class

Also prints the per-subtype breakdown of class-0 boxes, obtained by attributing
each image's defect boxes to the image-level ``label`` field of metadata.csv.
That breakdown is the evidence behind the two-class design decision.

Example
-------
    python -m coatingdet.dataset_stats --root datasets --tiles tiles
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

SUBSET_ORDER = ["train", "val", "test_1", "test_2", "test_3"]
SUBTYPES = ["inclusion", "pinhole", "scratch", "contamination"]


def subset_of(row) -> str:
    for s in ("test_1", "test_2", "test_3"):
        if s in row["path"]:
            return s
    return "train" if row["set"] == "train" else "val"


def count_boxes(label_file: Path) -> Counter:
    c = Counter()
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            p = line.split()
            if len(p) == 5 and p[0].isdigit():
                c[int(p[0])] += 1
    return c


def tile_dir(root: Path, subset: str, tiles: str) -> Path:
    if subset in ("train", "val"):
        return root / "train" / tiles / subset / "labels"
    return root / subset / tiles / "labels"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("datasets"))
    ap.add_argument("--tiles", default="tiles", help="tile directory name inside each subset")
    ap.add_argument("--latex", type=Path, default=None, help="write the LaTeX table here")
    a = ap.parse_args(argv)

    meta = list(csv.DictReader(open(a.root / "metadata.csv", encoding="utf-8-sig"), delimiter=";"))
    img = defaultdict(Counter)          # subset -> counters
    subtype = defaultdict(Counter)      # subset -> subtype -> class-0 boxes
    for r in meta:
        s = subset_of(r)
        img[s]["images"] += 1
        lbl = a.root / r["path"].lstrip("./").replace("images", "labels") / (
            Path(r["filename"]).stem + ".txt")
        b = count_boxes(lbl)
        if b[0]:
            img[s]["defective"] += 1
            subtype[s][r["label"]] += b[0]
        elif b[1]:
            img[s]["particle_only"] += 1

    tiles = defaultdict(Counter)
    for s in SUBSET_ORDER:
        d = tile_dir(a.root, s, a.tiles)
        if not d.is_dir():
            continue
        for f in d.glob("*.txt"):
            tiles[s]["tiles"] += 1
            for k, v in count_boxes(f).items():
                tiles[s][f"cls{k}"] += v

    hdr = (f"{'Subset':22s}{'Images':>8s}{'Defective':>11s}{'Particle-only':>15s}"
           f"{'Tiles':>8s}{'Instances':>11s}{'Defect':>9s}{'Particle':>10s}")
    print(hdr); print("-" * len(hdr))
    tot = Counter()
    rows = []
    for s in SUBSET_ORDER:
        if not img[s]:
            continue
        n_inst = tiles[s]["cls0"] + tiles[s]["cls1"]
        rows.append((s, img[s]["images"], img[s]["defective"], img[s]["particle_only"],
                     tiles[s]["tiles"], n_inst, tiles[s]["cls0"], tiles[s]["cls1"]))
        for k, v in zip(("images", "defective", "particle_only", "tiles", "inst", "c0", "c1"),
                        rows[-1][1:]):
            tot[k] += v
        print(f"{s:22s}{rows[-1][1]:8d}{rows[-1][2]:11d}{rows[-1][3]:15d}"
              f"{rows[-1][4]:8d}{rows[-1][5]:11d}{rows[-1][6]:9d}{rows[-1][7]:10d}")
    print("-" * len(hdr))
    print(f"{'TOTAL':22s}{tot['images']:8d}{tot['defective']:11d}{tot['particle_only']:15d}"
          f"{tot['tiles']:8d}{tot['inst']:11d}{tot['c0']:9d}{tot['c1']:10d}")

    print("\nclass-0 (defect) boxes by image-level subtype label:")
    print(f"{'subtype':16s}" + "".join(f"{s:>10s}" for s in SUBSET_ORDER) + f"{'TOTAL':>9s}")
    for k in SUBTYPES:
        vals = [subtype[s][k] for s in SUBSET_ORDER]
        print(f"{k:16s}" + "".join(f"{v:10d}" for v in vals) + f"{sum(vals):9d}")

    if a.latex:
        out = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
               r"Dataset & Images & Defective Images & Tiles & Instances & "
               r"Defect Instances & Particle Instances \\", r"\midrule"]
        for s, i, d, _, t, n, c0, c1 in rows:
            out.append(f"{s} & {i:,} & {d:,} & {t:,} & {n:,} & {c0:,} & {c1:,} \\\\")
        out += [r"\midrule",
                rf"\textbf{{Total}} & \textbf{{{tot['images']:,}}} & \textbf{{{tot['defective']:,}}} "
                rf"& \textbf{{{tot['tiles']:,}}} & \textbf{{{tot['inst']:,}}} "
                rf"& \textbf{{{tot['c0']:,}}} & \textbf{{{tot['c1']:,}}} \\",
                r"\bottomrule", r"\end{tabular}"]
        a.latex.write_text("\n".join(out))
        print("\nwrote", a.latex)


if __name__ == "__main__":
    main()
