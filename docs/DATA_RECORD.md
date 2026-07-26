# Data record: how the metadata fields were assigned

`metadata.csv` has one row per image and the columns
`path; filename; code; focal; label; balance; contrast; set`.

---

## `code`, `focal` — acquisition session and optics

`code` is the acquisition date (`YYYYMMDD`) and identifies the session. The
dataset is the union of 15 sessions spanning three years:

| code | images | camera | focal | balance | contrast | subset |
|---|---:|---|---|---|---|---|
| 20230502 | 137 | 5472×3648 | **12 mm** | Warm | Low | test_1 |
| 20230606 | 47 | 5472×3648 | 25 mm | Warm | Low | train |
| 20230607 | 126 | 5472×3648 | 25 mm | Warm | Low | train |
| 20230609 | 74 | 5472×3648 | 25 mm | Warm | Low | train |
| 20230628 | 50 | 5472×3648 | 25 mm | Warm | Low | train |
| 20230802 | 1 | 5472×3648 | 25 mm | Warm | Low | test_1 |
| 20231004 | 102 | 5472×3648 | 25 mm | Warm | Low | train |
| 20240527 | 26 | 5320×3032 | 25 mm | Cold | Low | **val** |
| 20240528 | 266 | 5320×3032 | 25 mm | Cold | Low | test_2 |
| 20240729 | 34 | 5320×3032 | 25 mm | Cold | Low | train |
| 20240801 | 83 | 5320×3032 | 25 mm | Cold | Low | test_2 |
| 20250311 | 695 | 5320×3032 | 25 mm | Cold | High | train |
| 20250507 | 570 | 3032×5320 | 25 mm | Cold | High | test_3 |
| 20250509 | 1,988 | 3032×5320 | 25 mm | Cold | High | test_3 |
| 20250512 | 1,217 | 3032×5320 | 25 mm | Cold | High | test_3 |

`balance` and `contrast` are therefore **not** per-image judgements: they are
properties of the optical setup used in each session. Every 2023 session used
the alternative camera body and its warm illumination and is labelled `Warm`;
every 2024/2025 session used the Basler ace 2 with the cold-white LED bar and is
labelled `Cold`. The `High` contrast label marks the 2025 sessions, acquired
after the tangential illumination bar reached its final configuration.

## `balance` and `contrast` — verification against the pixels

Because the assignment is session-level, it should still be recoverable from the
images. `coatingdet.illumination_stats` measures every image and reports how
well a single statistic separates the two label groups (5,416 images):

**`balance = Warm`** — recovered from the ratio of the red and blue channel
means:

| statistic | best threshold | agreement | Cohen's *d* | Warm | Cold |
|---|---|---:|---:|---|---|
| **R/B** | ≥ 1.037 | **99.5 %** | **4.70** | 1.121 ± 0.036 | 0.991 ± 0.015 |
| mean luminance | ≥ 170.6 | 94.7 % | 1.32 | 161.0 ± 27.3 | 132.9 ± 13.0 |
| RMS contrast | ≤ 0.186 | 94.7 % | 0.84 | 0.206 ± 0.103 | 0.274 ± 0.050 |

The R/B ratio separates the two groups essentially perfectly. The `Cold`
sessions cluster tightly at R/B = 0.991 ± 0.015 because the camera's white
balance was locked; the `Warm` sessions sit roughly nine Cold standard
deviations away.

**`contrast = High`** — no single statistic separates it as cleanly, which is
expected for a label that describes the illumination geometry rather than a
colour:

| statistic | best threshold | agreement | Cohen's *d* | High | Low |
|---|---|---:|---:|---|---|
| mean luminance | ≤ 139.9 | **95.5 %** | **1.81** | 130.4 ± 9.6 | 161.0 ± 21.9 |
| (P95−P5)/mean | ≥ 0.740 | 95.4 % | 1.28 | 0.905 ± 0.135 | 0.658 ± 0.236 |
| RMS contrast σ/µ | ≥ 0.232 | 95.0 % | 1.10 | 0.280 ± 0.047 | 0.208 ± 0.079 |

The `High` sessions are darker overall and have a wider normalised luminance
range — the signature of tangential lighting that leaves the background dark and
picks out surface relief. The residual ~5 % disagreement is images whose
photometry falls on the other side of the threshold from their session label,
which is the expected cost of a session-level assignment.

Users who need a per-image criterion rather than the session label can apply the
thresholds above directly; the script prints them for any dataset version.

## `label` — the image-level predominant category

`label` is one of `inclusion`, `pinhole`, `scratch`, `contamination`,
`no_defect`. It describes the **image**, not any individual box, and the
assignment rule is visible in the data:

| `label` | images with only defect boxes | with both | with only particle boxes | with no boxes |
|---|---:|---:|---:|---:|
| inclusion | 573 | 17 | 0 | 0 |
| scratch | 49 | 5 | 0 | 0 |
| contamination | 33 | 3 | 0 | 0 |
| pinhole | 19 | 0 | 0 | 0 |
| no_defect | 0 | 0 | 884 | 3,833 |

Critical defects take precedence over particles, and where several critical
categories occur in one image the most severe determines the label. The rule
holds without exception in the released data: every image containing at least one
class-0 box carries a defect subtype, and every image containing no class-0 box
carries `no_defect`. **`no_defect` therefore means "no critical defect", not
"nothing annotated"** — 884 of the 4,717 `no_defect` images carry particle boxes,
and the remaining 3,833 have no annotations at all.

(In the working copy from which the deposit was built, seven images violated this
rule: four carried a defect subtype although annotated with particle boxes only,
and three carried `no_defect` despite containing a defect box. All seven were
reconciled with their box annotations before release, so the deposited archive is
internally consistent for all 5,416 images. Re-run
`python -m coatingdet.annotation_audit --root <dataset>` to verify.)

Distribution of the class-0 boxes over the four subtypes, obtained by
attributing each image's defect boxes to its image-level label. All 768 boxes are
accounted for:

| subtype | train | val | test_1 | test_2 | test_3 | total |
|---|---:|---:|---:|---:|---:|---:|
| inclusion | 439 | 30 | 0 | 100 | 52 | **621** |
| pinhole | 19 | 0 | 0 | 0 | 0 | **19** |
| scratch | 27 | 0 | 0 | 0 | 57 | **84** |
| contamination | 1 | 0 | 0 | 0 | 43 | **44** |

This table is the reason the released detection annotations are binary. A
four-way scheme would train `contamination` on a single instance and `pinhole`
on 19, and would have **no test instances at all** for `pinhole`.

## `set`

`train`, `val` or `test`. The three test subsets are distinguished by their
`path` (`test_1/`, `test_2/`, `test_3/`); `val` is the 26 images of the
2024-05-27 session, which live under `train/raw_data/` but are routed to the
validation split by `tile_creator --val-prefix "Image__2024-05-27"`.

## Not present, despite the manuscript's field list

* `section` — the Data Records section lists a `section` column; it does not
  exist in `metadata.csv`. The section identifier is encoded in the Test-3
  filenames instead (e.g. `20250509T091747Z_002670_265_bot_raw.jpg` →
  section 265, lower band). Either add the column or amend the field list.
