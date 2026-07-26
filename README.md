# High-Resolution Images for Coating Defect Detection in Wind Turbine Structures

Processing, training, evaluation and quality-control code for the **CoatingDet**
dataset: 5,416 high-resolution RGB images of wind-tower coating surfaces acquired
in production, annotated with
bounding boxes in YOLO format.

* **Dataset (Zenodo):** https://doi.org/10.5281/zenodo.21600391

* **Paper:** *High-Resolution Images for Coating Defect Detection in Wind Turbine
  Structures*, Scientific Data (under review)

---

## Annotation scheme

The released annotations use **two classes**:

| id | name       | contents |
|----|------------|----------|
| 0  | `defect`   | the four critical coating defect types — inclusions, pinholes, scratches, contaminants |
| 1  | `particle` | minor surface irregularities (dust, small particles) |

The finer-grained subtype is retained **at image level** in the `label` field of
`metadata.csv`, not per box. The reason for merging the four critical types is
instance scarcity: of the 768 released `defect` boxes, 621 are inclusions, 84
scratches, 44 contaminants and 19 pinholes — and in the training split alone the
counts are 439 / 27 / 1 / 19. Three of the four subtypes cannot support
fine-grained training or evaluation, and all four trigger the same repair action,
so the operationally meaningful decision is `defect` vs `particle`.

## Dataset layout

The Zenodo archive distributes full-resolution images only; tiles are generated
locally by `tile_creator.py`.

```
datasets/
├── metadata.csv                 # one row per image, 5,416 rows
├── train/
│   └── raw_data/{images,labels}/    1,154 images (1,128 train + 26 val)
├── test_1/
│   └── raw_data/{images,labels}/      138 images   Low-Warm
├── test_2/
│   └── raw_data/{images,labels}/      349 images   Low-Cold
└── test_3/
    └── raw_data/{images,labels}/    3,775 images   High-Cold
```

Note that `metadata.csv` is **semicolon**-separated. Subset names combine the two
session-level illumination fields: **Low-Warm** = `contrast=Low` + `balance=Warm`,
and so on. `docs/DATA_RECORD.md` documents every field, and
`data/illumination_stats_v2.csv` gives the per-image photometry behind the
`balance` / `contrast` labels (regenerate with
`python -m coatingdet.illumination_stats`).

## Annotation quality

Every label file in the release was audited with `coatingdet.annotation_audit`:
no malformed records, no out-of-range coordinates, no degenerate or duplicate
boxes, no missing files, and the image-level `label` agrees with the box-level
annotations for all 5,416 images.

Inter-annotator agreement was measured on a seeded, subset-stratified sample of
150 images (250 boxes) reviewed independently by a second domain expert. All 250
boxes were confirmed to mark a real instance, and the two experts agreed on the
`defect`/`particle` assignment for 220 of 250 (88.0 %, Cohen's κ = 0.710).
Disagreement concentrates in the warm-balance sessions (22.2 % versus 7.9 % under
cold balance) — the same illumination axis that degrades detector performance on
Test 1, consistent with colour being a principal cue separating the two classes.

The completed review sheet, the computed statistics and the sampling seed are
released as `data/consistency_*` so the figure is checkable, and
`coatingdet.consistency_sample` reproduces or extends the study.

## Installation

```bash
git clone https://github.com/cvblab/CoatingDet-Defect-Detection.git
cd CoatingDet-Defect-Detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Then download the dataset from Zenodo and unpack it into `./datasets/`.

> **Version pin.** `requirements.txt` pins `ultralytics==8.3.100`. RT-DETR
> post-processing changed in the 8.4 series and the published metrics are not
> reproducible across that boundary — do not upgrade it.

## Quick start

```bash
bash scripts/reproduce_all.sh          # everything, end to end (~1 h on one GPU)
```

That script runs tiling, the composition table, the audit, training and
evaluation in order — read it to run the steps selectively. Every module also has
`--help`.

## Modules

| module | purpose |
|---|---|
| `coatingdet.tile_creator` | full-resolution images → 512 px YOLO tiles; all filter parameters exposed |
| `coatingdet.train` | reproducible YOLOv11n / RT-DETR training; aborts loudly on NaN divergence |
| `coatingdet.evaluate` | overall + per-class metrics, confusion counts, LaTeX tables |
| `coatingdet.dataset_stats` | recomputes the dataset-composition table from the data |
| `coatingdet.illumination_stats` | photometric verification of the `balance` / `contrast` fields |
| `coatingdet.annotation_audit` | formal consistency audit of every label file |
| `coatingdet.consistency_sample` | second-reviewer sampling package + IoU/κ agreement scoring |
| `coatingdet.domain_shift` | grey-world white-balance control experiment on Test 1 |
| `coatingdet.whitebalance` | grey-world correction, usable as a library function |
| `coatingdet.confusion_analysis` | defect ↔ particle confusion counts on Test 3 |
| `coatingdet.ap_analysis` | class-aware vs class-agnostic AP, version-independent |

## Reproducibility notes

Training uses Ultralytics defaults except: `epochs=20`, `batch=8`, `imgsz=512`,
`seed=0`, `patience` effectively disabled. `optimizer=auto` resolves to AdamW
with `lr0=0.00167` and `momentum=0.9`. **No hyper-parameter search and no early
stopping were performed** — the validation split is used only to monitor
convergence and to select the best-epoch checkpoint.

Evaluation uses `imgsz=512`, `conf=0.001` (the Ultralytics `val` default, the
operating point at which average precision is defined) and NMS `iou=0.7`. Report
`--conf 0.25` separately if a single deployed operating point is wanted; mAP is
not meaningful at that threshold.

`docs/REPRODUCE.md` documents two defects in the originally submitted results
table — swapped model columns and a checkpoint from a diverged training run —
and the controls used to diagnose the Test-1 anomaly.
`docs/DATA_RECORD.md` documents every metadata field and the per-subtype
distribution of the defect boxes.

## Mixing acquisition sessions

Test 1 is the only warm-balanced subset, and models trained on the cold-balanced
majority lose a large amount of recall on it. Apply a grey-world correction, or
widen the hue augmentation range, when combining sessions:

```python
from coatingdet.whitebalance import grey_world
img = grey_world(cv2.imread(path))
```

## Citation

```bibtex
@misc{coatingdet_data,
  author    = {P{\'e}rez Garc{\'i}a de la Puente, Natalia Lourdes and
               Mateos Luengo, Javier and Lario Femenia, Joan and
               L{\'o}pez L{\'o}pez, Eric and Aksu, Salih and
               Colomer, Adri{\'a}n and Naranjo, Valery},
  title     = {CoatingDet: Coating Defect Detection Dataset},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2},
  doi       = {10.5281/zenodo.21600391},
  url       = {https://doi.org/10.5281/zenodo.21600391}
}
```

## Licence

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, distributed via Zenodo.

## Funding

Horizon Europe grant 101057404 (ZDZW); PID2022-140189OB-C21 funded by
MICIU/AEI/10.13039/501100011033, ERDF/EU and FSE+; CIPROM/2022/20 (PROMETEO).

## Funding

Horizon Europe grant 101057404 (ZDZW); PID2022-140189OB-C21 funded by
MICIU/AEI/10.13039/501100011033, ERDF/EU and FSE+; CIPROM/2022/20 (PROMETEO,
Generalitat Valenciana).

