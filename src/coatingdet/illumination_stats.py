"""Measure the illumination descriptors that define the CoatingDet subsets.

The ``balance`` (cold/warm) and ``contrast`` (high/low) fields of metadata.csv
are session-level attributes, assigned from the illumination configuration used
during each acquisition campaign. This script verifies them against photometric
measurements of the images themselves, and reports how well each label can be
recovered from a single-image statistic.

Per image, on a fast 1/8-scale JPEG decode:

  RB            ratio of the red and blue channel means (colour-temperature proxy)
  lum_mean      mean of the Rec.601 luminance
  rms_contrast  std(L) / mean(L)
  p95_p5        95th minus 5th luminance percentile
  spread_ratio  p95_p5 / lum_mean  (normalised dynamic range)

Example
-------
    python -m coatingdet.illumination_stats --root datasets --out results
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def image_stats(path: Path) -> dict:
    im = Image.open(path)
    im.draft("RGB", (max(1, im.width // 8), max(1, im.height // 8)))
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    p5, p95 = np.percentile(lum, [5, 95])
    mean = float(lum.mean())
    return {"R": float(r), "G": float(g), "B": float(b), "RB": float(r / max(b, 1e-6)),
            "lum_mean": mean, "lum_std": float(lum.std()),
            "rms_contrast": float(lum.std() / max(mean, 1e-6)),
            "p95_p5": float(p95 - p5), "spread_ratio": float((p95 - p5) / max(mean, 1e-6))}


def best_threshold(x: np.ndarray, y: np.ndarray):
    """Best single-feature decision threshold and its agreement with the labels."""
    best = (0.0, None, 1)
    for t in np.unique(np.round(x, 4)):
        for sign in (1, -1):
            acc = float((((x * sign) >= (t * sign)) == y).mean())
            if acc > best[0]:
                best = (acc, float(t), sign)
    xp, xn = x[y], x[~y]
    d = abs(xp.mean() - xn.mean()) / np.sqrt((xp.var() + xn.var()) / 2 + 1e-12)
    return {"accuracy": best[0], "threshold": best[1],
            "direction": ">=" if best[2] > 0 else "<=", "cohens_d": float(d),
            "positive_mean": float(xp.mean()), "positive_sd": float(xp.std()),
            "negative_mean": float(xn.mean()), "negative_sd": float(xn.std())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("datasets"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    meta = list(csv.DictReader(open(a.root / "metadata.csv", encoding="utf-8-sig"), delimiter=";"))
    rows = []
    for i, r in enumerate(meta):
        p = a.root / r["path"].lstrip("./") / r["filename"]
        if not p.exists():
            continue
        s = image_stats(p)
        s.update({k: r[k] for k in ("filename", "code", "focal", "label", "balance",
                                    "contrast", "set")})
        rows.append(s)
        if i % 500 == 0:
            print(f"  {i}/{len(meta)}", flush=True)

    with open(a.out / "illumination_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} images measured\n")
    for field, positive, feats in [
        ("balance", "Warm", ["RB", "lum_mean", "rms_contrast", "spread_ratio"]),
        ("contrast", "High", ["lum_mean", "spread_ratio", "rms_contrast", "RB"]),
    ]:
        y = np.array([r[field] == positive for r in rows])
        print(f"--- recovering `{field} == {positive}` from a single image statistic ---")
        for feat in feats:
            x = np.array([r[feat] for r in rows])
            b = best_threshold(x, y)
            print(f"  {feat:14s} {b['direction']}{b['threshold']:9.4f}  "
                  f"agreement={b['accuracy'] * 100:6.2f}%  d={b['cohens_d']:5.2f}  "
                  f"{positive}={b['positive_mean']:.3f}+-{b['positive_sd']:.3f}  "
                  f"other={b['negative_mean']:.3f}+-{b['negative_sd']:.3f}")
        print()
    print("wrote", a.out / "illumination_stats.csv")


if __name__ == "__main__":
    main()
