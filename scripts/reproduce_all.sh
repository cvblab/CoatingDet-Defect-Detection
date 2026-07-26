#!/usr/bin/env bash
# End-to-end reproduction of every number reported in the CoatingDet manuscript.
#
# Prerequisite: download the dataset from Zenodo and unpack it so that
# ./datasets/ contains metadata.csv plus train/, test_1/, test_2/, test_3/.
#
# Runtime on a single RTX 3080: roughly 25 min for tiling, 5 min for YOLOv11n
# training, 20 min for RT-DETR training, 5 min for all evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

DATA=${1:-datasets}

echo "== 1/6  tiling (512 px, non-overlapping, min_bbox 0.35, min_tile 0.01) =="
python -m coatingdet.tile_creator --root "$DATA" --subset train --val-prefix "Image__2024-05-27"
for s in test_1 test_2 test_3; do
  python -m coatingdet.tile_creator --root "$DATA" --subset "$s"
done

echo
echo "== 2/6  dataset composition (Table 2) =="
python -m coatingdet.dataset_stats --root "$DATA" --latex results/table2_dataset.tex

echo
echo "== 3/6  annotation audit and illumination measurements =="
python -m coatingdet.annotation_audit  --root "$DATA" --out results
python -m coatingdet.illumination_stats --root "$DATA" --out results

echo
echo "== 4/6  training =="
python -m coatingdet.train --model yolo11n  --epochs 20 --data configs/data_val.yaml --name yolo11n_20ep
python -m coatingdet.train --model rtdetr-l --epochs 20 --data configs/data_val.yaml --name rtdetr_20ep

echo
echo "== 5/6  evaluation (Table 3 + per-class table) =="
python -m coatingdet.evaluate \
  --model "YOLOv11n=runs/yolo11n_20ep/weights/best.pt" \
  --model "RT-DETR=runs/rtdetr_20ep/weights/best.pt" \
  --subsets configs/data_val.yaml configs/data_test1.yaml \
            configs/data_test2.yaml configs/data_test3.yaml \
  --conf 0.001 --iou 0.7 --out results

echo
echo "== 6/6  illumination domain-shift control on Test 1 =="
python -m coatingdet.domain_shift --root "$DATA" --subset test_1 \
  --model "YOLOv11n=runs/yolo11n_20ep/weights/best.pt" \
  --model "RT-DETR=runs/rtdetr_20ep/weights/best.pt"

echo
echo "Done. Tables and metrics are in ./results, training artefacts in ./runs."
