#!/bin/bash
# T37: YOLO detector arm 4Y, its detection evaluation (with 4F and 4Z for comparison) and the
# full-image/crop vote with detector crops. Run on Raptor from the repo root:
#   source scripts/hpc/relabel_env.sh && bash scripts/hpc/submit_t37.sh
# Every job waits for the jobs it reads from (afterok); test is touched only after selection.json exists.
set -euo pipefail
: "${EXPERIMENTS_DIR:?source scripts/hpc/relabel_env.sh first}"

P=/project/a260012_FORWCL
X=$EXPERIMENTS_DIR
RAW=$P/raw_data/wound_ml_7classes
W=$P/yolo_weights
Y=$X/auto_4Y
PATIENTS=$WOUND_DATA_DIR/splits/patient_cases.csv
S=scripts/slurm/t37.sbatch
CPU=(-p compute --gres=none --cpus-per-task=4 --mem=16G)
sub() { sbatch --parsable --export=ALL "$@"; }

mkdir -p logs

# 1. YOLO dataset (train + validation photos), CPU
prep=$(sub -J 4Y_prepare "${CPU[@]}" --time=3:00:00 \
  --export=ALL,PY_ARGS="src/detection/yolo.py prepare --image-root $RAW --output-dir $Y/yolo_data" $S)

# 2. Train each size, then detect its validation photos (pytest first, in the first job only)
trains=()
for size in s m l; do
  C=yolo11$size
  args="src/detection/yolo.py train --size $size --weights $W/$C.pt --data-dir $Y/yolo_data --output-dir $Y/detector/$C"
  args="$args | src/detection/yolo.py detect --run-dir $Y/detector/$C --splits validation --output $Y/detection/dev/$C.csv --image-root $RAW"
  [[ $size == s ]] && args="-m pytest -q tests/test_yolo_detection.py tests/test_detection_evaluation.py | $args"
  trains+=("$(sub -J 4Y_$C --time=24:00:00 --dependency=afterok:$prep --export=ALL,PY_ARGS="$args" $S)")
done
deps=$(IFS=:; echo "${trains[*]}")

# 3. Choose the size and threshold on validation; judge the gate (CPU)
sel=$(sub -J 4Y_select "${CPU[@]}" --time=1:00:00 --dependency=afterok:$deps --export=ALL,VENV=none,PY_ARGS="src/detection/select.py --arm 4Y --detections-dir $Y/detection/dev --configs yolo11s yolo11m yolo11l --output $Y/detection/selection.json" $S)

# 4. Test photos, once, with the chosen run
test=$(sub -J 4Y_test --time=3:00:00 --dependency=afterok:$sel --export=ALL,PY_ARGS="src/detection/yolo.py detect --detector-dir $Y/detector --selection $Y/detection/selection.json --splits test --output-dir $Y/detection/test --image-root $RAW" $S)

# 5. Detection metrics for 4Y, 4F and 4Z (patients resampled), then the comparison
metric_jobs=()
for A in 4Y 4F 4Z; do
  dep=(); [[ $A == 4Y ]] && dep=(--dependency=afterok:$test)
  metric_jobs+=("$(sub -J ${A}_t37_metrics "${CPU[@]}" --time=12:00:00 "${dep[@]}" --export=ALL,VENV=none,PY_ARGS="src/detection/detection_metrics.py evaluate --detections $X/auto_$A/detection/test/*.csv --selection $X/auto_$A/detection/selection.json --split test --patient-cases $PATIENTS --output-dir $X/detection_t37/$A" $S)")
done
sub -J t37_metrics_report "${CPU[@]}" --time=0:30:00 --dependency=afterok:$(IFS=:; echo "${metric_jobs[*]}") \
  --export=ALL,VENV=none,PY_ARGS="src/detection/detection_metrics.py report --arm 4Y=$X/detection_t37/4Y 4F=$X/detection_t37/4F 4Z=$X/detection_t37/4Z --output-dir $X/detection_t37/report" $S

# 6. 4Y crops of the test photos (8% padding, T19 threshold/cap/fallback rules), CPU
crops=$(sub -J 4Y_crops "${CPU[@]}" --time=2:00:00 --dependency=afterok:$test --export=ALL,VENV=none,PY_ARGS="src/data/create_auto_roi_dataset.py --selection $Y/detection/selection.json --detections $Y/detection/test/*.csv --splits test --output-dir $Y/roi --image-root $RAW --crop-root $DERIVED_DIR/roi_auto_4Y" $S)

# 7. The per-crop models (S25-S32) on the auto crops; 4F crops already exist (T19 on T23 data)
p4y=$(sub -J 4Y_crop_predict --time=3:00:00 --dependency=afterok:$crops --export=ALL,VENV=none,PY_ARGS="src/evaluation/predict_on_manifest.py --experiments-root $X/roi_crop --design crop --manifest $Y/roi/box/roi_manifest.csv --image-root ${DERIVED_DIR}/roi_auto_4Y_box --output-root $X/fusion_auto/source_4Y/roi_crop --check-manual" $S)
p4f=$(sub -J 4F_crop_predict --time=3:00:00 --export=ALL,VENV=none,PY_ARGS="src/evaluation/predict_on_manifest.py --experiments-root $X/roi_crop --design crop --manifest $WOUND_DATA_DIR/roi_auto/4F/box/roi_manifest.csv --image-root ${DERIVED_DIR}/roi_auto_4F_box --output-root $X/fusion_auto/source_4F/roi_crop" $S)

# 8. Vote, rescore per patient, compare (CPU)
fus=$(sbatch --parsable --export=ALL --dependency=afterok:$p4y:$p4f scripts/slurm/t37_fusion.sbatch)

echo "prepare $prep | train ${trains[*]} | select $sel | test $test | metrics ${metric_jobs[*]} | crops $crops | predict $p4y $p4f | fusion $fus"
