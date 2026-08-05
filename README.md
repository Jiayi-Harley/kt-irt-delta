# kt-irt-delta

Code for a UCL FYP at Century Tech: predicting **learning gain Δθ** instead of P(correct),
so that recommendations can be scored offline. IRT gives an ability estimate θ; the label is
Δθ = θ_after − θ_before across a learning block. A KT model is then trained/probed to predict Δθ.

## Layout

```
irt/          IRT sanity checks
  irt_demo.py                     fit 3PL IRT on synthetic data, check it recovers ability/difficulty
labels/       build Δθ labels
  build_delta_labels.py           skill-level Δθ from a raw KT csv (assist2015 / assist2009)
  build_delta_qikt.py             question-level Δθ built ON pykt's quelevel sequences, aligned to QIKT
models/
  train_delta.py                  DeltaKT: LSTM over a block + θ_before -> predict Δθ (MSE)
experiments/  controlled / diagnostic runs on synthetic data
  synthetic_delta_experiment.py   ZPD synthetic world with known true Δθ; effect of window size
  theta_estimator_experiment.py   MLE vs MAP θ estimator on synthetic data
```

## Setup

- conda env `century` (Python 3.9): `EduCDM`, `torch` (CPU), `pandas`, `numpy`, plus pyKT deps.
- **pyKT** is a separate dependency, NOT in this repo (gitignored). Clone it to `pykt-toolkit/`
  at the repo root and `pip install -e .` it. The scripts locate its data at
  `pykt-toolkit/data/<dataset>/`. Note: pyKT's assist2009 preprocessor hardcodes `utf-8`;
  the raw csv is `latin-1`, so that read encoding must be changed.
- Datasets, `*.npz` label files, and model checkpoints are gitignored (large / regenerable).

## Pipeline (question-level, aligned with QIKT)

1. Preprocess assist2009 in pyKT: `python data_preprocess.py -d assist2009` (from `pykt-toolkit/examples/`).
2. Build Δθ on pyKT's question-level sequences: `python labels/build_delta_qikt.py --W 40 --K 10 --split test`.
3. Train / extract QIKT, then correlate its knowledge-acquisition output with Δθ.