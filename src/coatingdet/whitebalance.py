"""Grey-world white balancing for CoatingDet images.

Test 1 (Low-Warm) is the only subset acquired under warm illumination; models
trained on the cold-balanced majority of the data lose a large amount of recall
on it (see docs/REPRODUCE.md). Applying a grey-world correction at inference
time recovers most of that loss and costs one pass over the pixels, so it is
recommended as the default pre-processing step for users who mix acquisition
sessions.

    from coatingdet.whitebalance import grey_world
    corrected = grey_world(cv2.imread(path))

The CLI writes a corrected copy of an image tree, preserving filenames so that
the original label files remain valid.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def grey_world(img: np.ndarray) -> np.ndarray:
    """Scale each colour channel so that all channel means coincide."""
    means = img.reshape(-1, img.shape[-1]).mean(0)
    gain = means.mean() / np.maximum(means, 1e-6)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="directory with images/ and labels/")
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args(argv)

    (a.dst / "images").mkdir(parents=True, exist_ok=True)
    (a.dst / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((a.src / "images").iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(a.dst / "images" / f"{p.stem}.jpg"), grey_world(img))
        lbl = a.src / "labels" / f"{p.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, a.dst / "labels" / f"{p.stem}.txt")
        n += 1
    print(f"white-balanced {n} images -> {a.dst}")


if __name__ == "__main__":
    main()
