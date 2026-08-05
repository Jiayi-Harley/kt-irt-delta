"""
Rebuild Δθ labels ON pykt's question-level (quelevel) sequences, so the labels sit on
the exact same student/position axis QIKT sees, and can be aligned with QIKT's per-position
outputs later.

Design (agreed with Jiayi 2026-08-05):
- window / sequence granularity: QUESTION level (we read pykt's quelevel sequence files,
  so ordering, truncation, folds all match QIKT exactly).
- theta-fitting granularity: CONCEPT (skill) level, because assist2009 has 17737 questions
  and question-level IRT is too sparse to estimate; 123 concepts are dense enough.
  This is a resolution limit forced by the data, not an inconsistency: same positions,
  we just estimate ability with concept-level difficulties.

Per row (each row = one student's contiguous <=200 chunk, exactly one QIKT sequence):
    [pre-window W] [block K] [post-window W]  slid non-overlapping
    theta_before fit on pre concepts/responses, theta_after on post, Δθ = after - before.

    python build_delta_qikt.py --W 40 --K 10 --split test
"""
import argparse, json, os, numpy as np, pandas as pd
from collections import defaultdict
from EduCDM import EMIRT
import logging, warnings
logging.getLogger().setLevel(logging.ERROR); warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (code/)
DATA = os.path.join(_ROOT, "pykt-toolkit", "data", "assist2009")
CODE = _ROOT
FILES = {"test": "test_sequences_quelevel.csv",
         "train_valid": "train_valid_sequences_quelevel.csv"}

ap = argparse.ArgumentParser()
ap.add_argument("--W", type=int, default=40)
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--split", default="test", choices=list(FILES))
a = ap.parse_args()
W, K = a.W, a.K; NEED = W + K + W

keyid = json.load(open(os.path.join(DATA, "keyid2idx.json")))
n_c = len(keyid["concepts"])   # 123

def to_ints(s):
    # concepts can in principle be multi like "5_10"; take the first. padding is -1.
    return [int(str(x).split("_")[0]) for x in str(s).split(",")]

def valid_seq(row):
    """return (concepts, responses, questions) for real positions only (r in {0,1})."""
    cs, rs, qs = to_ints(row["concepts"]), to_ints(row["responses"]), to_ints(row["questions"])
    C, Rr, Q = [], [], []
    for c, r, q in zip(cs, rs, qs):
        if r in (0, 1):
            C.append(c); Rr.append(r); Q.append(q)
    return C, Rr, Q

# ---------- 1. concept-level IRT calibration on train_valid, aggregated per uid ----------
tv = pd.read_csv(os.path.join(DATA, FILES["train_valid"]))
acc = defaultdict(lambda: defaultdict(list))        # uid -> concept -> [responses]
for _, row in tv.iterrows():
    C, Rr, _ = valid_seq(row)
    uid = int(row["uid"])
    for c, r in zip(C, Rr):
        acc[uid][c].append(r)
uids = sorted(acc)
R = -1.0 * np.ones((len(uids), n_c))
for i, uid in enumerate(uids):
    for c, lst in acc[uid].items():
        R[i, c] = 1.0 if np.mean(lst) >= 0.5 else 0.0
cdm = EMIRT(R, len(uids), n_c, dim=1, skip_value=-1)
cdm.train(lr=1e-4, epoch=5)
print(f"IRT concept-level fit on train_valid: {len(uids)} uids, {n_c} concepts")

def theta_of_window(C, Rr):
    vec = -1.0 * np.ones(n_c)
    agg = defaultdict(list)
    for c, r in zip(C, Rr):
        agg[c].append(r)
    for c, lst in agg.items():
        vec[c] = 1.0 if np.mean(lst) >= 0.5 else 0.0
    return cdm.transform(vec).item()

# ---------- 2. slide windows within each row of --split ----------
df = pd.read_csv(os.path.join(DATA, FILES[a.split]))
recs = []            # row_idx, block_start, theta_before, delta
n_elig = 0
for ridx, (_, row) in enumerate(df.iterrows()):
    C, Rr, Q = valid_seq(row)
    if len(C) >= NEED:
        n_elig += 1
    start = 0
    while start + NEED <= len(C):
        tb = theta_of_window(C[start:start + W], Rr[start:start + W])
        ta = theta_of_window(C[start + W + K:start + W + K + W], Rr[start + W + K:start + W + K + W])
        recs.append((ridx, start + W, tb, ta - tb))     # block occupies [start+W, start+W+K)
        start += NEED
recs = np.array(recs, dtype=float)

print(f"split={a.split}  rows={len(df)}  eligible(>= {NEED})={n_elig}  windows={len(recs)}")
if len(recs):
    d = recs[:, 3]
    print(f"Δθ  mean {d.mean():+.3f}  std {d.std():.3f}  frac>0 {(d>0).mean():.2f}  "
          f"saturated_before {(np.abs(recs[:,2])>3.9).mean():.2f}")
out = os.path.join(CODE, f"delta_qikt_{a.split}_W{W}K{K}.npz")
np.savez(out, split=a.split, W=W, K=K, records=recs,
         col_names=np.array(["row_idx", "block_start", "theta_before", "delta"]))
print("saved", out)