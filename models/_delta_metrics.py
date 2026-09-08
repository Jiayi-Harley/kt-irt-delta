"""Shared metrics for the Delta-theta evaluators (plain_delta / qikt_delta_mt).

Given best-epoch validation arrays P (prediction), T (true Delta-theta label), Z (theta_pre),
compute a spread of interpretable metrics covering four aspects:

  correlation : raw Pearson corr(P,T), Spearman rank corr, block signal (partial corr controlling Z)
  ranking     : pairwise ranking accuracy (of two blocks, how often the higher-gain one is ranked higher)
  error size  : RMSE and MAE on Delta-theta (in theta units, "how big is a mistake")
  classify    : AUC of P predicting whether the block helps (T>0)

Training is untouched; this only reads the predictions the scripts already produce, so the existing
raw_corr / block_signal reproduce exactly and the rest come along for free.
"""
import os, json, numpy as np


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ranks(x):                       # ordinal ranks (ties broken by order; fine for continuous scores)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float); r[order] = np.arange(1, len(x) + 1)
    return r


def all_metrics(P, T, Z, n_pairs=300000, seed=0):
    P = np.asarray(P, float); T = np.asarray(T, float); Z = np.asarray(Z, float)
    m = {}
    m["raw_corr"] = _pearson(P, T)
    rpy, rpz, ryz = _pearson(P, T), _pearson(P, Z), _pearson(T, Z)
    d = np.sqrt(max(0.0, (1 - rpz ** 2) * (1 - ryz ** 2)))
    m["block_signal"] = float((rpy - rpz * ryz) / d) if d > 1e-9 else float("nan")
    m["rmse"] = float(np.sqrt(np.mean((P - T) ** 2)))
    m["mae"] = float(np.mean(np.abs(P - T)))
    # pairwise ranking accuracy over random pairs (skip ties in the true gain)
    rng = np.random.default_rng(seed); n = len(P)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    ok = T[i] != T[j]
    m["pair_acc"] = float((np.sign(P[i][ok] - P[j][ok]) == np.sign(T[i][ok] - T[j][ok])).mean()) \
        if ok.any() else float("nan")
    # matched: only pairs with similar theta_pre, so prior ability cannot order them and the block must
    # (recommendation-faithful, the same spirit as the block signal)
    tau = 0.2 * (Z.std() if Z.std() > 1e-9 else 1.0)
    mm = ok & (np.abs(Z[i] - Z[j]) <= tau)
    m["pair_acc_matched"] = float((np.sign(P[i][mm] - P[j][mm]) == np.sign(T[i][mm] - T[j][mm])).mean()) \
        if int(mm.sum()) >= 1000 else float("nan")
    # AUC: can P tell above-median-gain blocks from below-median ones (Mann-Whitney form). The threshold
    # is the median gain, so the two classes are balanced; a raw T>0 split is degenerate because almost
    # every block has positive gain.
    thr = float(np.median(T))
    y = (T > thr).astype(int); npos = int(y.sum()); nneg = len(y) - npos
    if npos and nneg:
        r = _ranks(P)
        m["auc_med"] = float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    else:
        m["auc_med"] = float("nan")
    return m


ORDER = ["raw_corr", "block_signal", "pair_acc", "pair_acc_matched", "rmse", "mae", "auc_med"]


def finalize(out_path, tag, meta, ptz_list):
    """Compute per-seed metrics, aggregate mean/std, print a table, append one JSON line to out_path."""
    per = [all_metrics(P, T, Z) for (P, T, Z) in ptz_list]
    agg = {k: {"mean": float(np.nanmean([d[k] for d in per])),
               "std":  float(np.nanstd([d[k] for d in per]))} for k in ORDER}
    print(f"\n=== {tag} | {meta} | {len(per)} seeds ===")
    print("metric         " + "".join(f"{k:>18}" for k in ORDER))
    for si, d in enumerate(per):
        print(f"seed {si:<10}" + "".join(f"{d[k]:>18.4f}" for k in ORDER))
    print("mean       " + "".join(f"{agg[k]['mean']:>18.4f}" for k in ORDER))
    print("std        " + "".join(f"{agg[k]['std']:>18.4f}" for k in ORDER))
    rec = {"tag": tag, **meta, "n_seeds": len(per), "agg": agg, "per_seed": per}
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[appended -> {out_path}]", flush=True)