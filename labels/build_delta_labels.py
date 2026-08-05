"""
Build Δθ labels from a KT dataset.  Configurable via command line so we can compare
datasets, item granularities, and window sizes.

    python build_delta_labels.py --dataset assist2009 --item problem --W 50 --K 10

Structure per student (see project03_notes §28):
    [pre-window W] [block K] [post-window W]
       θ_before                  θ_after       Δθ = θ_after - θ_before
Item parameters are fitted ONCE globally and frozen (shared ruler), so the two θ's compare.
"""
import argparse, os, numpy as np, pandas as pd
from EduCDM import EMIRT
import logging, warnings
logging.getLogger().setLevel(logging.ERROR); warnings.filterwarnings("ignore")

# data lives in the pykt-toolkit clone at <repo root>/pykt-toolkit/data (gitignored, cloned separately)
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pykt-toolkit", "data")

CONFIG = {
    "assist2015": dict(path=os.path.join(_DATA, "assist2015", "2015_100_skill_builders_main_problems.csv"),
                       enc="utf-8", user="user_id", time="log_id",
                       item={"concept": "sequence_id"}),
    "assist2009": dict(path=os.path.join(_DATA, "assist2009", "skill_builder_data_corrected_collapsed.csv"),
                       enc="latin-1", user="user_id", time="order_id",
                       item={"problem": "problem_id", "skill": "skill_id"}),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="assist2009")
ap.add_argument("--item", default="problem")      # granularity: which column is the IRT item
ap.add_argument("--W", type=int, default=50)
ap.add_argument("--K", type=int, default=10)
a = ap.parse_args()
cfg = CONFIG[a.dataset]; W, K = a.W, a.K; NEED = W + K + W
item_col = cfg["item"][a.item]

# --- load, keep valid rows ---
df = pd.read_csv(cfg["path"], encoding=cfg["enc"], low_memory=False)
df = df.dropna(subset=[cfg["user"], cfg["time"], item_col, "correct"])
df = df[df["correct"].isin([0, 1])]
df["correct"] = df["correct"].astype(int)

# keep only students long enough to yield one sample -> also bounds the IRT matrix size
seqlen = df.groupby(cfg["user"]).size()
keep = seqlen[seqlen >= NEED].index
df = df[df[cfg["user"]].isin(keep)]

df["iidx"] = pd.factorize(df[item_col])[0]
df["uidx"] = pd.factorize(df[cfg["user"]])[0]
n_items = int(df["iidx"].nunique()); n_students = int(df["uidx"].nunique())
print(f"{a.dataset} / item={a.item} / W={W} K={K}: "
      f"{n_students} eligible students, {n_items} items, {len(df)} responses")

# --- 1. global IRT fit -> freeze item parameters ---
mean_corr = df.groupby(["uidx", "iidx"])["correct"].mean()
R = -1.0 * np.ones((n_students, n_items))
u = mean_corr.index.get_level_values(0).to_numpy()
c = mean_corr.index.get_level_values(1).to_numpy()
R[u, c] = (mean_corr.to_numpy() >= 0.5).astype(float)
cdm = EMIRT(R, n_students, n_items, dim=1, skip_value=-1)
cdm.train(lr=1e-4, epoch=5)
print("global IRT fit done")

def theta_of_window(pairs):
    vec = -1.0 * np.ones(n_items)
    for it, r in pairs:
        vec[it] = r
    return cdm.transform(vec).item()

# --- 2. slide along each student, cut MULTIPLE non-overlapping samples ---
df = df.sort_values(["uidx", cfg["time"]])
block_c, block_r, theta_before, delta = [], [], [], []
for _, g in df.groupby("uidx", sort=False):
    seq = list(zip(g["iidx"].values, g["correct"].values))
    start = 0
    while start + NEED <= len(seq):                 # keep cutting until the tail is too short
        s = seq[start:start+NEED]
        tb = theta_of_window(s[0:W])
        ta = theta_of_window(s[W+K:W+K+W])
        blk = s[W:W+K]
        block_c.append([it for it, r in blk])
        block_r.append([r for it, r in blk])
        theta_before.append(tb)
        delta.append(ta - tb)
        start += NEED                               # non-overlapping

block_c = np.array(block_c); block_r = np.array(block_r)
theta_before = np.array(theta_before); delta = np.array(delta)
out = f"delta_labels_{a.dataset}_{a.item}_W{W}.npz"
np.savez(out, block_c=block_c, block_r=block_r,
         theta_before=theta_before, delta=delta, n_items=n_items)
print(f"saved {len(delta)} samples to {out}")
print(f"Δθ  mean {delta.mean():+.3f}  std {delta.std():.3f}  frac>0 {(delta>0).mean():.2f}  "
      f"saturated {(np.abs(theta_before)>3.9).mean():.2f}")