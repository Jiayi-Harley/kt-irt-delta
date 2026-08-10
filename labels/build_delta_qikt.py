"""
Rebuild Δθ labels ON pykt's question-level (quelevel) sequences, so the labels sit on the
exact same student/position axis QIKT sees and can be aligned with QIKT's per-position outputs.
Generalized over dataset and IRT item granularity.

    python labels/build_delta_qikt.py --dataset nips_task34 --item question --W 40 --K 10 --split test
    python labels/build_delta_qikt.py --dataset assist2009  --item concept  --W 40 --K 10 --split test

- window / sequence granularity: always QUESTION level (we read pykt's quelevel sequences, so
  ordering, truncation, folds all match QIKT exactly). Only the theta ruler granularity changes.
- --item = which unit gets an IRT difficulty when estimating theta:
    concept  : skill-level theta. Dense enough everywhere. assist2009 MUST use this (17737
               questions, ~16 answers each, question-level IRT too sparse).
    question : question-level theta. Only sensible on dense data like nips_task34 (948 questions,
               ~1459 answers each). This is the whole point of switching to a denser dataset:
               it lets theta itself go to question level, closing the last homogeneous-assumption gap.
"""
import argparse, json, os, numpy as np, pandas as pd
from collections import defaultdict
from EduCDM import EMIRT
import logging, warnings
logging.getLogger().setLevel(logging.ERROR); warnings.filterwarnings("ignore")
np.random.seed(0)   # EMIRT init uses global numpy RNG; seed so the fit (and saturation) is reproducible

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (code/)
CODE = _ROOT
FILES = {"test": "test_sequences_quelevel.csv",
         "train_valid": "train_valid_sequences_quelevel.csv"}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="assist2009")
ap.add_argument("--item", default="concept", choices=["concept", "question"])
ap.add_argument("--W", type=int, default=40)
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--split", default="test", choices=list(FILES))
ap.add_argument("--cap", type=int, default=100,
                help="cap per-student observed items during IRT calibration; avoids EM likelihood "
                     "underflow on many-item (question-level) fits. No-op for concept level. 0 = off.")
a = ap.parse_args()
W, K = a.W, a.K; NEED = W + K + W
DATA = os.path.join(_ROOT, "pykt-toolkit", "data", a.dataset)

keyid = json.load(open(os.path.join(DATA, "keyid2idx.json")))
COL = "concepts" if a.item == "concept" else "questions"    # which column is the IRT item
n_items = len(keyid[COL])
NUM_Q, NUM_C = len(keyid["questions"]), len(keyid["concepts"])
MAX_C = int(keyid.get("max_concepts", 1))                   # concepts per position (for QIKT's QueEmb)

def to_ints(s):
    # a position can hold multiple concepts like "5_10"; take the primary one. padding is -1.
    return [int(str(x).split("_")[0]) for x in str(s).split(",")]

def to_multiconcept(s):
    # each position -> up to MAX_C concept ids, padded with -1 (the format QIKT's QueEmb expects)
    out = []
    for x in str(s).split(","):
        cs = [int(v) for v in str(x).split("_")]
        out.append(cs[:MAX_C] + [-1] * (MAX_C - len(cs)))
    return out

def valid_seq(row):
    """(item_seq, responses, question_seq, multiconcept_seq) over real positions (r in {0,1})."""
    items, rs = to_ints(row[COL]), to_ints(row["responses"])
    qs, cms = to_ints(row["questions"]), to_multiconcept(row["concepts"])
    I, R, Q, CM = [], [], [], []
    for it, r, q, cm in zip(items, rs, qs, cms):
        if r in (0, 1):
            I.append(it); R.append(r); Q.append(q); CM.append(cm)
    return I, R, Q, CM

# ---------- 1. IRT calibration on train_valid, aggregated per uid ----------
tv = pd.read_csv(os.path.join(DATA, FILES["train_valid"]))
acc = defaultdict(lambda: defaultdict(list))        # uid -> item -> [responses]
for _, row in tv.iterrows():
    I, R, _, _ = valid_seq(row)
    uid = int(row["uid"])
    for it, r in zip(I, R):
        acc[uid][it].append(r)
uids = sorted(acc)
Rmat = -1.0 * np.ones((len(uids), n_items))
rng = np.random.default_rng(0)
for i, uid in enumerate(uids):
    items = list(acc[uid].items())
    if a.cap and len(items) > a.cap:            # subsample so the EM likelihood does not underflow
        items = [items[k] for k in rng.choice(len(items), a.cap, replace=False)]
    for it, lst in items:
        Rmat[i, it] = 1.0 if np.mean(lst) >= 0.5 else 0.0
cdm = EMIRT(Rmat, len(uids), n_items, dim=1, skip_value=-1)
cdm.train(lr=1e-4, epoch=5)
print(f"IRT {a.item}-level fit on {a.dataset}/train_valid: {len(uids)} uids, {n_items} items")

def theta_of_window(I, R):
    vec = -1.0 * np.ones(n_items)
    agg = defaultdict(list)
    for it, r in zip(I, R):
        agg[it].append(r)
    for it, lst in agg.items():
        vec[it] = 1.0 if np.mean(lst) >= 0.5 else 0.0
    return cdm.transform(vec).item()

# ---------- 2. slide windows within each row of --split ----------
df = pd.read_csv(os.path.join(DATA, FILES[a.split]))
recs = []                       # row_idx, block_start, theta_before, delta
block_c, block_r = [], []       # block: IRT-item ids + responses (for the plain DeltaKT encoder)
block_q, block_cm = [], []      # block: question ids + multi-concepts (for QIKT's QueEmb encoder)
pre_q, pre_cm, pre_r = [], [], []  # pre-window history: the W nuggets before the block (for the honest Δθ predictor, exp_log §18)
n_elig = 0
for ridx, (_, row) in enumerate(df.iterrows()):
    I, R, Q, CM = valid_seq(row)
    if len(I) >= NEED:
        n_elig += 1
    start = 0
    while start + NEED <= len(I):
        bs = start + W                                   # block occupies [bs, bs+K)
        tb = theta_of_window(I[start:bs], R[start:bs])
        ta = theta_of_window(I[bs + K:bs + K + W], R[bs + K:bs + K + W])
        recs.append((ridx, bs, tb, ta - tb))
        block_c.append(I[bs:bs + K]); block_r.append(R[bs:bs + K])
        block_q.append(Q[bs:bs + K]); block_cm.append(CM[bs:bs + K])
        pre_q.append(Q[start:bs]); pre_cm.append(CM[start:bs]); pre_r.append(R[start:bs])
        start += NEED
recs = np.array(recs, dtype=float)
block_c = np.array(block_c, dtype=np.int64); block_r = np.array(block_r, dtype=np.int64)
block_q = np.array(block_q, dtype=np.int64); block_cm = np.array(block_cm, dtype=np.int64)
pre_q = np.array(pre_q, dtype=np.int64); pre_cm = np.array(pre_cm, dtype=np.int64); pre_r = np.array(pre_r, dtype=np.int64)

print(f"dataset={a.dataset} split={a.split} item={a.item}  rows={len(df)}  "
      f"eligible(>= {NEED})={n_elig}  windows={len(recs)}")
if len(recs):
    d = recs[:, 3]
    print(f"Δθ  mean {d.mean():+.3f}  std {d.std():.3f}  frac>0 {(d>0).mean():.2f}  "
          f"saturated_before {(np.abs(recs[:,2])>3.9).mean():.2f}")
out = os.path.join(CODE, f"delta_qikt_{a.dataset}_{a.item}_{a.split}_W{W}K{K}.npz")
np.savez(out, dataset=a.dataset, item=a.item, split=a.split, W=W, K=K,
         n_items=n_items, num_q=NUM_Q, num_c=NUM_C, max_c=MAX_C,
         records=recs, block_c=block_c, block_r=block_r,
         block_q=block_q, block_cm=block_cm,               # for QIKT's QueEmb encoder
         pre_q=pre_q, pre_cm=pre_cm, pre_r=pre_r,           # pre-window history for the honest Δθ predictor (exp_log §18)
         theta_before=recs[:, 2], delta=recs[:, 3],        # separate arrays for train_delta.py
         col_names=np.array(["row_idx", "block_start", "theta_before", "delta"]))
print("saved", out)
