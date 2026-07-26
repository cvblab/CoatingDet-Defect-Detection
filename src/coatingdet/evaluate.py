"""Evaluate CoatingDet checkpoints and regenerate the manuscript tables.

Produces, for every model x subset combination:

  * overall precision, recall, mAP@0.5, mAP@0.5:0.95
  * the same metrics per class (defect / particle)
  * the defect <-> particle confusion counts
  * LaTeX for Table 3 and for the per-class supplementary table

All numbers come from a single ``model.val()`` call per pair, so the whole table
is internally consistent. Evaluation settings are stated explicitly rather than
inherited silently:

    imgsz  512
    conf   0.001  (Ultralytics `val` default; the operating point at which
                   average-precision metrics are meant to be computed)
    iou    0.7    (NMS IoU threshold, Ultralytics default)

Use ``--conf 0.25`` to obtain the single-operating-point precision/recall that a
deployed system would see; note that mAP is not meaningful at that threshold.

Example
-------
    python -m coatingdet.evaluate \
        --model "YOLOv11n=runs/yolo11n_20ep/weights/best.pt" \
        --model "RT-DETR=runs/rtdetr-l_20ep/weights/best.pt" \
        --subsets configs/data_val.yaml configs/data_test1.yaml \
                  configs/data_test2.yaml configs/data_test3.yaml \
        --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import RTDETR, YOLO

CLASS_NAMES = {0: "defect", 1: "particle"}


def load(weights: str):
    """RT-DETR checkpoints must be opened with the RTDETR wrapper, YOLO with YOLO."""
    stem = Path(weights).as_posix().lower()
    return RTDETR(weights) if "rtdetr" in stem or "detr" in stem else YOLO(weights)


def evaluate(weights: str, data: Path, *, imgsz: int, conf: float, iou: float,
             project: Path, tag: str):
    m = load(weights).val(data=str(data), imgsz=imgsz, conf=conf, iou=iou, split="val",
                          project=str(project), name=tag, exist_ok=True,
                          plots=True, verbose=False)
    rec = {
        "weights": weights, "data": str(data),
        "settings": {"imgsz": imgsz, "conf": conf, "nms_iou": iou},
        "overall": {"P": float(m.box.mp), "R": float(m.box.mr),
                    "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)},
        "per_class": {},
    }
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rec["per_class"][CLASS_NAMES[int(c)]] = {
            "P": float(p), "R": float(r), "mAP50": float(ap50), "mAP50_95": float(ap)}

    cm = np.asarray(m.confusion_matrix.matrix)  # [pred, true], last index = background
    if cm.shape == (3, 3):
        rec["confusion"] = {
            "defect_as_defect": float(cm[0, 0]),
            "defect_as_particle": float(cm[1, 0]),
            "defect_missed": float(cm[2, 0]),
            "particle_as_particle": float(cm[1, 1]),
            "particle_as_defect": float(cm[0, 1]),
            "particle_missed": float(cm[2, 1]),
            "background_as_defect": float(cm[0, 2]),
            "background_as_particle": float(cm[1, 2]),
        }
    return rec


def latex_main(res, models, subsets):
    cols = "l" + "cccc" * len(models)
    head = " & ".join(rf"\multicolumn{{4}}{{c}}{{{m}}}" for m in models)
    sub = " & ".join(["P & R & mAP@0.5 & mAP@0.5:0.95"] * len(models))
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Comparative performance of the object detection models across "
             r"the CoatingDet validation and test subsets.}",
             r"\label{tab:model-comparison}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             rf"\multirow{{2}}{{*}}{{Dataset}} & {head} \\",
             " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(models))),
             rf" & {sub} \\", r"\midrule"]
    for s in subsets:
        cells = []
        for m in models:
            o = res[m][s]["overall"]
            cells += [f"{o['P']:.3f}", f"{o['R']:.3f}", f"{o['mAP50']:.3f}", f"{o['mAP50_95']:.3f}"]
        lines.append(f"{s} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_per_class(res, models, subsets):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Per-class detection performance on the CoatingDet subsets. "
             r"Subsets that contain instances of a single class are reported for that class only.}",
             r"\label{tab:per-class}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Subset & Class & P & R & mAP@0.5 \\", r"\midrule"]
    for m in models:
        for s in subsets:
            for c, v in res[m][s]["per_class"].items():
                lines.append(f"{m} & {s} & {c} & {v['P']:.3f} & {v['R']:.3f} & {v['mAP50']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=WEIGHTS",
                    help="repeatable, e.g. --model 'YOLOv11n=runs/a/weights/best.pt'")
    ap.add_argument("--subsets", nargs="+", type=Path, required=True,
                    help="dataset yaml files; the `val:` key of each is evaluated")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--project", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    models = [s.split("=", 1)[0] for s in a.model]
    subsets = [p.stem.replace("data_", "") for p in a.subsets]
    res = {}
    for spec in a.model:
        name, w = spec.split("=", 1)
        for p, s in zip(a.subsets, subsets):
            res.setdefault(name, {})[s] = evaluate(
                w, p, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                project=a.project, tag=f"{name}_{s}_c{a.conf}".replace(" ", "_"))

    (a.out / f"metrics_conf{a.conf}.json").write_text(json.dumps(res, indent=2))
    (a.out / "table3_model_comparison.tex").write_text(latex_main(res, models, subsets))
    (a.out / "table_per_class.tex").write_text(latex_per_class(res, models, subsets))

    w = max(len(m) for m in models) + 2
    print(f"\n{'model':{w}s}{'subset':10s}{'class':10s}{'P':>7s}{'R':>7s}{'mAP50':>8s}{'mAP50-95':>10s}")
    for m in models:
        for s in subsets:
            o = res[m][s]["overall"]
            print(f"{m:{w}s}{s:10s}{'ALL':10s}{o['P']:7.3f}{o['R']:7.3f}{o['mAP50']:8.3f}{o['mAP50_95']:10.3f}")
            for c, v in res[m][s]["per_class"].items():
                print(f"{'':{w}s}{'':10s}{c:10s}{v['P']:7.3f}{v['R']:7.3f}{v['mAP50']:8.3f}{v['mAP50_95']:10.3f}")
            cf = res[m][s].get("confusion")
            if cf and (cf["particle_as_defect"] or cf["defect_as_particle"]):
                print(f"{'':{w}s}{'':10s}confusions: particle->defect="
                      f"{cf['particle_as_defect']:.0f}, defect->particle={cf['defect_as_particle']:.0f}")
    print(f"\nwrote {a.out}/metrics_conf{a.conf}.json, table3_model_comparison.tex, table_per_class.tex")


if __name__ == "__main__":
    main()
