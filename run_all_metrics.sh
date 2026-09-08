#!/bin/bash
# Full metric + symmetry sweep for results sections 4.5-4.7 (block-reading kou-jing, original scripts).
# 3 models x 7 label files x 5 seeds; every cell gets all 7 metrics. Results append to results_metrics.jsonl.
cd "D:/UCL 上课/PROJECT/Century Tech/code" || exit 1
export PYTHONPATH=pykt-toolkit
PY="E:/anaconda2021python3.9/envs/century/python.exe"
SEEDS=0,1,2,3,4
LOG=results_metrics_run.log
rm -f results_metrics.jsonl results_metrics_synth.jsonl
echo "START $(date)" | tee "$LOG"

# priority order: headline K=10, then W-sweep gaps, then rest of K-sweep
labels=(W40K10 W20K50 W60K50 W40K50 W40K20 W40K30 W40K40)
lf() { echo "delta_qikt_nips_task34_question_train_valid_$1.npz"; }

run() {  # $1 label-tag  $2.. extra args
  local L="$1"; shift
  echo "=== $* | $L | $(date +%H:%M:%S) ===" | tee -a "$LOG"
  "$PY" "$@" --seeds $SEEDS --labels "$(lf "$L")" >> "$LOG" 2>&1 \
    && echo "  ok $(date +%H:%M:%S)" | tee -a "$LOG" \
    || echo "  FAILED $(date +%H:%M:%S)" | tee -a "$LOG"
}

# LEAP (plain_delta, 15 epochs) -- fast; fills LEAP block across K
for L in "${labels[@]}"; do run "$L" models/plain_delta.py --emb plain --epochs 15; done
# QIKT single-task (corr_w 0, 30 epochs)
for L in "${labels[@]}"; do run "$L" models/qikt_delta_mt.py --corr_w 0 --epochs 30; done
# QIKT multi-task (corr_w 1, 30 epochs)
for L in "${labels[@]}"; do run "$L" models/qikt_delta_mt.py --corr_w 1 --epochs 30; done

# Synthetic (LEAP only, metrics vs TRUE gain): K-sweep (W=40) + W-sweep (K=50)
synth=("40 10" "40 20" "40 30" "40 40" "40 50" "20 50" "60 50")
for WK in "${synth[@]}"; do
  echo "=== SYNTH-LEAP W K = $WK | $(date +%H:%M:%S) ===" | tee -a "$LOG"
  "$PY" experiments/synthetic_leap.py $WK >> "$LOG" 2>&1 \
    && echo "  ok $(date +%H:%M:%S)" | tee -a "$LOG" \
    || echo "  FAILED $(date +%H:%M:%S)" | tee -a "$LOG"
done

echo "DONE $(date)" | tee -a "$LOG"