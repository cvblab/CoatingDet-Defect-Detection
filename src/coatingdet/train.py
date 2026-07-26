"""Reproducible training of the CoatingDet technical-validation models.

The published experiments train YOLOv11n and RT-DETR-L from their COCO-pretrained
checkpoints on the tiled training split, using the Ultralytics defaults except
for the values listed below. No hyper-parameter search, no early stopping and no
learning-rate tuning were performed - the validation split is used only to
monitor convergence and to select the best-epoch checkpoint.

    epochs      20      (see --epochs for the longer-schedule control runs)
    batch        8
    imgsz      512
    optimizer  auto  -> AdamW, lr0 = 0.00167, momentum 0.9 (chosen by Ultralytics)
    seed         0
    patience  1000     (effectively disabled, so all epochs always run)

IMPORTANT (see the revision note in docs/REPRODUCE.md): the RT-DETR run used for
the first submission diverged to NaN and must not be reused. Always check
runs/<name>/results.csv for NaN losses before reporting metrics from a run;
``--check-nan`` does this automatically at the end of training.

Examples
--------
    python -m coatingdet.train --model yolo11n --epochs 20
    python -m coatingdet.train --model rtdetr-l --epochs 20
    python -m coatingdet.train --model rtdetr-l --epochs 60 --name rtdetr_60ep
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from ultralytics import RTDETR, YOLO

BUILDERS = {"yolo11n": (YOLO, "yolo11n.pt"), "rtdetr-l": (RTDETR, "rtdetr-l.pt")}


def check_nan(results_csv: Path) -> bool:
    """Return True if any epoch recorded a NaN loss (i.e. the run diverged)."""
    if not results_csv.exists():
        return False
    bad = []
    for row in csv.DictReader(open(results_csv)):
        for k, v in row.items():
            if "loss" in k:
                try:
                    if math.isnan(float(v)):
                        bad.append(row["epoch"])
                        break
                except (TypeError, ValueError):
                    pass
    if bad:
        print("\n" + "!" * 72)
        print(f"! TRAINING DIVERGED: NaN losses at epoch(s) {', '.join(bad)}.")
        print("! Do not report metrics from this run - re-train (e.g. with a")
        print("! different --seed, or --amp False) before evaluating.")
        print("!" * 72)
    return bool(bad)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n", choices=list(BUILDERS))
    ap.add_argument("--data", type=Path, default=Path("configs/data_val.yaml"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", default="true", choices=["true", "false"],
                    help="set false if a run diverges to NaN under mixed precision")
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)

    cls, weights = BUILDERS[a.model]
    name = a.name or f"{a.model}_{a.epochs}ep"
    cls(weights).train(
        data=str(a.data), epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, seed=a.seed,
        device=a.device, amp=(a.amp == "true"), patience=1000, plots=True, val=True,
        project=str(a.project), name=name, exist_ok=True,
    )
    check_nan(a.project / name / "results.csv")
    print(f"\nweights: {a.project / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
