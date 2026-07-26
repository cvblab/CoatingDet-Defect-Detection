# Reproducing the technical validation

Everything below was produced with the code in this repository against the
released dataset. Hardware: one NVIDIA RTX 3080 (12 GB). Reference numbers were
also cross-checked on the NVIDIA DGX A100 system used for the original
experiments.

---

## 1. Tiling

`coatingdet.tile_creator` with the published defaults reproduces the tiled
subsets exactly:

| subset | full-resolution images | tiles produced |
|---|---:|---:|
| train | 1,128 | 1,126 |
| val (Low-Cold, 27 May 2024) | 26 | 30 |
| test_1 (Low-Warm) | 138 | 210 |
| test_2 (Low-Cold) | 349 | 107 |
| test_3 (High-Cold) | 3,775 | 912 |

Only tiles that retain at least one bounding box after the 35 % / 1 % area
filters are written, which is why the tile counts are far smaller than the image
counts. The validation split is carved out by filename prefix
(`Image__2024-05-27`), not at random — see §5.

## 2. Training configuration

Ultralytics defaults except for the values below. `optimizer=auto` resolves to
**AdamW, lr0 = 0.001667, momentum = 0.9**; the `lr0: 0.01` recorded in
`args.yaml` is the requested value that `auto` overrides.

| parameter | value |
|---|---|
| epochs | 20 |
| batch | 8 |
| imgsz | 512 |
| seed | 0 |
| optimizer | auto → AdamW (lr0 0.001667, momentum 0.9) |
| lrf | 0.01 (cosine-free linear decay) |
| weight_decay | 0.0005 |
| warmup_epochs | 3 |
| patience | disabled (all epochs always run) |
| amp | true |
| augmentation | Ultralytics defaults: `hsv_h 0.015`, `hsv_s 0.7`, `hsv_v 0.4`, `translate 0.1`, `scale 0.5`, `fliplr 0.5`, `mosaic 1.0`, `erasing 0.4` |

**No hyper-parameter search, no learning-rate tuning and no early stopping were
performed.** The validation split is used only to monitor convergence and to
pick the best-epoch checkpoint.

## 3. Convergence

YOLOv11n, 20 epochs — training losses decrease monotonically; validation
mAP@0.5 settles from epoch ~16:

| epoch | 5 | 10 | 15 | 16 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|
| val mAP@0.5 | 0.761 | 0.668 | 0.804 | 0.848 | 0.830 | 0.853 | 0.836 |

RT-DETR-L, 20 epochs — plateaus from epoch ~14:

| epoch | 5 | 10 | 14 | 15 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|
| val mAP@0.5 | 0.490 | 0.653 | 0.831 | 0.846 | 0.846 | 0.831 | 0.826 |

**Longer-schedule control.** RT-DETR-L re-trained for 60 epochs under identical
settings peaks at val mAP@0.5 = 0.849 (epoch 27) versus 0.846 (epoch 15) within
the 20-epoch budget — a gain of 0.003 for triple the compute. YOLOv11n peaks at
0.871 over 60 epochs versus 0.853 within 20. The 20-epoch schedule is therefore
not the limiting factor. Note that the validation split holds 30 tiles, so
run-to-run variance is large relative to these differences.

## 4. Evaluation settings

`model.val()` with `imgsz=512`, **`conf=0.001`** and **NMS `iou=0.7`** — the
Ultralytics `val` defaults, which is the operating point at which average
precision is defined. Precision and recall in the tables are consequently the
values at the F1-optimal point of the PR curve, not at a fixed deployment
threshold. Pass `--conf 0.25` to `coatingdet.evaluate` for a single deployed
operating point; mAP should not be read at that threshold.

## 5. Split design

The five subsets are **acquisition campaigns, not random partitions**:

| subset | sessions | camera | focal | balance / contrast |
|---|---|---|---|---|
| train | 2023-06-06/07/09/28, 2023-10-04, 2024-07-29, 2025-03-11 | mixed | 25 mm | mixed |
| val | 2024-05-27 | 5320×3032 | 25 mm | Cold / Low |
| test_1 | 2023-05-02 | 5472×3648 | **12 mm** | Warm / Low |
| test_2 | 2024-05-28, 2024-08-01 | 5320×3032 | 25 mm | Cold / Low |
| test_3 | 2025-05-07/09/12 | 3032×5320 (portrait) | 25 mm | Cold / High |

Keeping whole sessions intact is what makes the test subsets informative: a
random split would leak the same tower, coating batch, lighting setup and even
neighbouring tiles of the same image across train and test, and the reported
numbers would measure memorisation rather than generalisation.

Test 3 is larger than the training set because it is a single campaign — the
May 2025 inspection with the final robot-mounted configuration — that was
reserved in full for testing. It is the dominant operational condition of the
deployed system, so it is the subset on which generalisation matters most.

The validation subset is one session (2024-05-27, 26 images, 30 tiles). It is
small, and it is used only to monitor convergence and select the best epoch;
because no hyper-parameters were tuned on it, its size does not create a
selection-bias risk for the reported test numbers. Users who want to tune
hyper-parameters should build a larger validation split by re-running
`tile_creator` with a different `--val-prefix`, or by holding out one of the
test campaigns.


## 6. Two defects in the originally submitted Table 3

### 6.1 The model columns are swapped

Re-running the exact evaluation of the submitted table (`ultralytics 8.3.100`,
`model.val(imgsz=500, rect=False)`, released checkpoints) reproduces every value
to ±0.001 — but assigned to the *other* model for three of the four rows:

| subset | value printed under "YOLOv11n" | actually produced by | value printed under "RT-DETR" | actually produced by |
|---|---|---|---|---|
| Val | 0.930 / 0.818 / 0.856 / 0.435 | **RT-DETR** | 0.828 / 0.728 / 0.812 / 0.483 | **YOLOv11n** |
| Test 1 | 0.395 / 0.963 / 0.785 / 0.432 | YOLOv11n ✓ | 0.828 / 0.009 / 0.020 / 0.004 | RT-DETR ✓ |
| Test 2 | 0.867 / 0.860 / 0.822 / 0.286 | **RT-DETR** | 0.866 / 0.804 / 0.860 / 0.405 | **YOLOv11n** |
| Test 3 | 0.692 / 0.780 / 0.700 / 0.430 | **RT-DETR** | 0.753 / 0.772 / 0.833 / 0.543 | **YOLOv11n** |

The authors' own intermediate files agree: `metrics_table_yolo.tex` holds the
YOLOv11n values and `metrics_table_detr.tex` the RT-DETR values, and the columns
were transposed when the two were merged into the manuscript table. Only the
Test-1 row survived correctly, which is why the RT-DETR anomaly is real while
the surrounding comparison is not.

This inverts the conclusion drawn in the text: on Test 3 it is **YOLOv11n**
that reaches mAP@0.5 = 0.833, not RT-DETR (0.700). `coatingdet.evaluate` now
produces the whole table from a single call per model/subset pair so the
columns cannot be mismatched again.

### 6.2 The RT-DETR checkpoint comes from a diverged run

The released RT-DETR weights come from a run configured for 100 epochs that
**diverged to NaN at epoch 23** and terminated at epoch 32; `best.pt` is the
epoch-22 checkpoint, saved immediately before the collapse.

`coatingdet.train` now scans `results.csv` after every run and warns loudly if
any epoch recorded a NaN loss. If a run diverges, re-train with a different
`--seed` or with `--amp false`.

The divergence is a real reproducibility problem, but it is **not** the cause of
the Test-1 result: freshly trained RT-DETR checkpoints (20 and 60 epochs, no
NaN) show the same behaviour on Test 1 under `ultralytics 8.3.100`
(mAP@0.5 = 0.026 and 0.005). See §7.

## 7. What actually happens on Test 1

Test 1 is the only warm-illumination subset and its 218 boxes are **all**
`particle`. Four controls were run.

**Control A — is it a localisation failure or a classification failure?**
AP@0.5 recomputed from the raw predictions in two modes, class-aware (the class
must also match) and class-agnostic (geometry only). Same implementation for
every row, so the two columns are directly comparable:

| model | Test 1, `particle` AP@0.5 (class-aware) | Test 1, AP@0.5 (class-agnostic) |
|---|---:|---:|
| YOLOv11n (released) | 0.239 | **0.870** |
| RT-DETR (released, diverged) | 0.365 | **0.606** |
| RT-DETR (re-trained, 20 ep) | 0.879 | **0.956** |
| RT-DETR (re-trained, 60 ep) | 0.625 | **0.962** |

The particles are found and localised almost perfectly. What fails is the class
decision: at conf 0.25, **39 of 43** RT-DETR detections on Test 1 carry the label
`defect`, while every ground-truth box is `particle`. Test 1 is a *classification*
failure under domain shift, not a detection failure.

**Control B — is the colour cast responsible?** Grey-world white balancing of
the Test-1 tiles, pixels only and labels untouched, recovers the class decision:

| model | `particle` AP@0.5 as released | after grey-world WB | Δ |
|---|---:|---:|---:|
| YOLOv11n (released) | 0.239 | 0.927 | **+0.688** |
| RT-DETR (released, diverged) | 0.365 | 0.699 | **+0.334** |
| RT-DETR (re-trained, 20 ep) | 0.879 | 0.937 | +0.058 |
| RT-DETR (re-trained, 60 ep) | 0.625 | 0.855 | **+0.230** |

Under the submitted evaluation (`ultralytics 8.3.100`) the same correction moves
the released checkpoints from mAP@0.5 = 0.785 → 0.949 (YOLOv11n) and
0.020 → 0.396 (RT-DETR). Colour is the feature both detectors rely on to
separate benign particles from critical defects, and the warm cast pushes
particles into the `defect` region of feature space.

**Control C — is it a geometric/scale shift?** Test 1 is also the only subset
acquired with the 5472×3648 body and a 12 mm lens. Re-tiling it from images
rescaled by 25/12 to match the training pixel scale, without colour correction,
makes every model *worse* (mAP@0.5 ≤ 0.10). The shift is chromatic, not
geometric.

**Control D — is it under-training?** No. RT-DETR peaks at val mAP@0.5 = 0.849
(epoch 27 of 60) versus 0.846 (epoch 15 of 20), and its Test-1 behaviour is
unchanged. YOLOv11n peaks at 0.871 over 60 epochs versus 0.853 within 20 — a
modest gain that alters no conclusion.

**A caution about the 0.020 figure.** RT-DETR's Test-1 mAP@0.5 is unstable across
Ultralytics minor versions — 0.020 under 8.3.100, 0.365 under 8.4.105 — even
though the two versions return *byte-identical raw predictions* for these
weights. The difference is in how average precision is aggregated when the
validation set contains instances of only one class. The extreme value 0.020
should therefore not be quoted without the version pin; the version-independent
statement is the one in Control A.

**Conclusion.** The Test-1 result reflects a genuine, dataset-level
illumination domain shift that both architectures are subject to and that
degrades the *class* decision, not the localisation; RT-DETR is markedly more
sensitive to it than YOLOv11n. It is not evidence that the Test-1 annotations
are unusable, and it is not explained by under-training or by the diverged
checkpoint alone.

Practical consequence for users: apply `coatingdet.whitebalance.grey_world` when
mixing acquisition sessions, or widen the hue augmentation — the Ultralytics
default `hsv_h=0.015` spans only ±1.5 % of the hue circle, far narrower than the
warm/cold gap in this dataset (mean R/B 1.121 versus 0.991).

## 8. Per-class behaviour and defect ↔ particle confusion

Per-class metrics, `ultralytics 8.3.100`, `conf=0.001`, NMS IoU 0.7 (subsets
containing a single class are listed once):

| model | subset | class | P | R | mAP@0.5 |
|---|---|---|---:|---:|---:|
| YOLOv11n (released) | val | defect | 0.828 | 0.727 | 0.812 |
| YOLOv11n (released) | test_1 | particle | 0.395 | 0.963 | 0.785 |
| YOLOv11n (released) | test_2 | defect | 0.866 | 0.804 | 0.860 |
| YOLOv11n (released) | test_3 | defect | 0.575 | 0.842 | 0.755 |
| YOLOv11n (released) | test_3 | particle | 0.929 | 0.699 | 0.910 |
| RT-DETR (released) | val | defect | 0.930 | 0.818 | 0.856 |
| RT-DETR (released) | test_1 | particle | 0.575 | 0.009 | 0.020 |
| RT-DETR (released) | test_2 | defect | 0.867 | 0.860 | 0.822 |
| RT-DETR (released) | test_3 | defect | 0.492 | 0.914 | 0.683 |
| RT-DETR (released) | test_3 | particle | 0.892 | 0.643 | 0.717 |

Confusion on Test 3, the only subset containing both classes (conf 0.25, NMS IoU
0.7, prediction↔ground-truth matching at IoU ≥ 0.5):

| model | true defect → defect | → particle | missed | true particle → particle | **→ defect** | missed |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv11n (released) | 88.8 % | 9.2 % | 2.0 % | 77.8 % | **14.4 %** | 7.8 % |
| RT-DETR (released, diverged) | 94.1 % | 3.9 % | 2.0 % | 75.0 % | **24.0 %** | 1.0 % |
| RT-DETR (re-trained, 20 ep) | 88.8 % | 5.3 % | 5.9 % | 67.2 % | **31.2 %** | 1.5 % |

Absolute counts for YOLOv11n: of 784 particle boxes, 113 are called `defect`; of
152 defect boxes, 14 are called `particle` and 3 are missed entirely. There are
additionally 133 `defect` and 89 `particle` detections with no matching
ground-truth box.

The confusion is strongly asymmetric, and **not** in the direction the raw
instance counts suggest. Despite particles outnumbering defects 1,688 to 773 in
the tiles, both detectors escalate benign particles to `defect` far more often
than they downgrade real defects (14–31 % versus 4–9 %). Critical defects are
rarely missed outright (2–6 %), so the operational cost of this dataset's
imbalance is unnecessary repair calls, not undetected damage.

Mitigations available to users:

* class-weighted or focal loss, weighting `particle` up so that the
  particle→defect direction is penalised;
* rebalancing at the data level by re-running `tile_creator` with different
  `--min-bbox-fraction` / `--overlap`, which changes the per-class tile yield;
* raising the decision threshold for `defect` only, trading defect recall
  (currently 0.78–0.92) against the false-alarm rate;
* a two-stage cascade — detect any surface feature, then classify the crop —
  which decouples localisation from the defect/particle decision;
* grey-world white balancing, which as §7 shows fixes most of the class errors
  induced by illumination differences.
