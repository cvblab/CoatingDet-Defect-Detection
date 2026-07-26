"""Tile full-resolution CoatingDet images into YOLO-ready patches.

This is the ``tile_creator.py`` referred to in the manuscript. It replaces the
separate train/test tiling scripts used during development with a single
parameterised entry point, so that the tiled datasets behind Tables 2 and 3 can
be reproduced exactly, and so that users can generate their own variants.

Defaults reproduce the published experiments:

    tile size            512 x 512 px
    overlap              0.0 (non-overlapping)
    min. bbox fraction   0.35  (>=35 % of the original annotated area must fall
                                inside the tile for the box to be kept)
    min. tile fraction   0.01  (the retained box must cover >=1 % of the tile)
    empty tiles          discarded

Only tiles that retain at least one bounding box are written, which is why the
tile counts in Table 2 are much smaller than the number of images.

Examples
--------
    # reproduce the published training / validation tiles
    python -m coatingdet.tile_creator --root datasets --subset train \
        --val-prefix "Image__2024-05-27"

    # reproduce a test subset
    python -m coatingdet.tile_creator --root datasets --subset test_1

    # a custom overlapping variant
    python -m coatingdet.tile_creator --root datasets --subset test_3 \
        --tile-size 640 --overlap 0.25 --out my_tiles
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def yolo_to_xyxy(box, w, h):
    xc, yc, bw, bh = (float(v) for v in box)
    return (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h


def read_labels(path: Path, w: int, h: int):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        boxes.append((cls, *yolo_to_xyxy(parts[1:], w, h)))
    return boxes


def tile_one(img, boxes, stem, img_dir: Path, lbl_dir: Path, *, tile: int,
             overlap: float, min_bbox_frac: float, min_tile_frac: float,
             keep_empty: bool) -> int:
    h, w = img.shape[:2]
    step = max(1, int(round(tile * (1.0 - overlap))))
    written = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            xe, ye = min(x + tile, w), min(y + tile, h)
            if xe <= x or ye <= y:
                continue
            lines = []
            for cls, bx, by, bw, bh in boxes:
                ix1, iy1 = max(bx, x), max(by, y)
                ix2, iy2 = min(bx + bw, xe), min(by + bh, ye)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 0 or ih <= 0:
                    continue
                inter = iw * ih
                if inter / (bw * bh) < min_bbox_frac:
                    continue
                if inter / ((xe - x) * (ye - y)) < min_tile_frac:
                    continue
                lines.append(
                    f"{cls} {(ix1 - x + iw / 2) / (xe - x):.6f} "
                    f"{(iy1 - y + ih / 2) / (ye - y):.6f} "
                    f"{iw / (xe - x):.6f} {ih / (ye - y):.6f}"
                )
            if not lines and not keep_empty:
                continue
            name = f"{stem}_{y}_{x}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img[y:ye, x:xe])
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            written += 1
    return written


def fresh(*dirs: Path):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("datasets"),
                    help="dataset root containing train/ test_1/ test_2/ test_3/")
    ap.add_argument("--subset", required=True,
                    choices=["train", "test_1", "test_2", "test_3"])
    ap.add_argument("--out", default="tiles", help="output directory name inside the subset")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.0, help="0.0-0.9 fractional overlap")
    ap.add_argument("--min-bbox-fraction", type=float, default=0.35)
    ap.add_argument("--min-tile-fraction", type=float, default=0.01)
    ap.add_argument("--keep-empty", action="store_true",
                    help="also write tiles that contain no annotation")
    ap.add_argument("--val-prefix", default=None,
                    help="filename prefix routed to a val/ split instead of train/ "
                         "(the published split uses 'Image__2024-05-27', the Low-Cold "
                         "acquisition session of 27 May 2024)")
    a = ap.parse_args(argv)

    src = a.root / a.subset / "raw_data"
    img_src, lbl_src = src / "images", src / "labels"
    if not img_src.is_dir():
        raise SystemExit(f"{img_src} not found - check --root")

    base = a.root / a.subset / a.out
    if a.val_prefix:
        tr_i, tr_l = base / "train" / "images", base / "train" / "labels"
        va_i, va_l = base / "val" / "images", base / "val" / "labels"
        fresh(tr_i, tr_l, va_i, va_l)
    else:
        tr_i, tr_l = base / "images", base / "labels"
        va_i = va_l = None
        fresh(tr_i, tr_l)

    n_img = n_tile = n_val = 0
    for p in sorted(img_src.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"  ! unreadable, skipped: {p.name}")
            continue
        h, w = img.shape[:2]
        boxes = read_labels(lbl_src / f"{p.stem}.txt", w, h)
        if not boxes and not a.keep_empty:
            continue
        to_val = bool(a.val_prefix) and p.name.startswith(a.val_prefix)
        out_i, out_l = (va_i, va_l) if to_val else (tr_i, tr_l)
        k = tile_one(img, boxes, p.stem, out_i, out_l, tile=a.tile_size,
                     overlap=a.overlap, min_bbox_frac=a.min_bbox_fraction,
                     min_tile_frac=a.min_tile_fraction, keep_empty=a.keep_empty)
        if k:
            n_img += 1
            n_tile += k
            n_val += k if to_val else 0

    print(f"\nTiling of '{a.subset}' complete "
          f"(tile={a.tile_size}, overlap={a.overlap}, "
          f"min_bbox={a.min_bbox_fraction}, min_tile={a.min_tile_fraction})")
    print(f"  source images contributing tiles : {n_img}")
    print(f"  tiles written                    : {n_tile}"
          + (f"  (train {n_tile - n_val} / val {n_val})" if a.val_prefix else ""))
    print(f"  output                           : {base}")


if __name__ == "__main__":
    main()
