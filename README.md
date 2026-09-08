# kt-irt-delta

Code for a UCL MSc thesis at Century Tech: **predicting learning gain $\Delta\theta$** instead of
$P(\text{correct})$, so that a recommender can score content offline. Item Response Theory (IRT) gives an
ability estimate $\theta$; the label is $\Delta\theta = \theta_{\text{after}} - \theta_{\text{before}}$
across a block of content, read against a single frozen IRT ruler. **LEAP** is the evaluator trained to
predict $\Delta\theta$; a **QIKT** model is used as a baseline and as a diagnostic.

## Layout

```
irt/
  irt_demo.py                     fit 3PL IRT on synthetic data; recover ability / difficulty
labels/
  build_delta_qikt.py             question-level Δθ built ON pyKT's quelevel sequences, aligned to QIKT
  build_delta_labels.py           skill-level Δθ from a raw KT csv (earlier variant)
models/
  plain_delta.py                  LEAP (main model): [q,c,r] embedding + two LSTMs + Δθ head, MSE
  leap_content.py                 LEAP, content-only variant: masks the block's responses, keeps its q/c
  qikt_delta_mt.py                QIKT baseline: --corr_w 0 single-task, --corr_w 1 multi-task
  _delta_metrics.py               the seven evaluation metrics + cross-seed aggregation (finalize)
  train_delta.py, train_qikt_delta.py   earlier DeltaKT skeletons (superseded by LEAP)
experiments/
  synthetic_leap.py               LEAP on a synthetic world with known true gain; W / K sweep
  qikt_delta_corr.py              diagnostic: a frozen QIKT's internal state already encodes the gain
  make_teacher.py                 privileged-information (LUPI) teacher; distillation study
  synthetic_delta_experiment.py   earlier ZPD synthetic world; effect of window size
  theta_estimator_experiment.py   MLE vs MAP θ estimator on synthetic data
run_all_metrics.sh                full sweep: LEAP / QIKT-single / QIKT-multi over K, W, seeds + synthetic
run_mt.sh                         UCL Myriad SGE job script
```

## Metrics

`_delta_metrics.py` reports seven measures of a prediction against the $\Delta\theta$ label: raw
correlation, block signal (partial correlation controlling $\theta_{\text{before}}$), pairwise ranking
accuracy, matched pairwise ranking accuracy (pairs of similar-ability students only), RMSE, MAE, and AUC at
the median-gain split. `finalize` aggregates the mean and standard deviation across seeds into a JSONL log.

## Setup

- conda env `century` (Python 3.9): `EduCDM`, `torch` (CUDA build for GPU), `pandas`, `numpy`, plus pyKT
  deps. The LEAP / QIKT runs use a GPU.
- **pyKT** is a separate dependency, NOT in this repo (gitignored). Clone it to `pykt-toolkit/` at the repo
  root and `pip install -e .` it, then run with `PYTHONPATH=pykt-toolkit`. The scripts locate its data at
  `pykt-toolkit/data/<dataset>/`. Note: pyKT's assist2009 preprocessor hardcodes `utf-8` while the raw csv
  is `latin-1`, so that read encoding must be changed.
- Datasets, `*.npz` label files, and model checkpoints are gitignored (large / regenerable).

## Pipeline (question-level, aligned with QIKT)

1. Preprocess in pyKT: `python data_preprocess.py -d <dataset>` (from `pykt-toolkit/examples/`).
2. Build the $\Delta\theta$ labels: `python labels/build_delta_qikt.py --W 40 --K 10 --split test`.
3. Train and evaluate LEAP: `python models/plain_delta.py --labels <W40K10 npz> --dataset <dataset>`.
4. Compare against the QIKT baseline: `python models/qikt_delta_mt.py --corr_w 0` (single-task) and
   `--corr_w 1` (multi-task).
5. Reproduce the full sweep and all seven metrics with `run_all_metrics.sh`.
