"""
Does a MAP theta estimator (MLE + a normal prior) beat plain MLE for our Δθ labels?

Diagnosis so far: labels are noisy because θ from a small window saturates at the ±4
grid edge. MLE pushes θ to the extreme when the window is nearly all-correct or all-wrong.
A MAP estimator adds a prior "θ should be a normal value near 0", which pulls θ back when
the window has little information. That should reduce saturation and improve the labels.

We reuse the SAME synthetic data and the SAME fitted item params, and only swap the θ
estimator, then compare corr(label Δθ, TRUE Δθ). Synthetic data has ground truth, so this
comparison is clean.
"""
import numpy as np
from EduCDM import EMIRT
import logging, warnings
logging.getLogger().setLevel(logging.ERROR); warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)
n_students, n_items = 3000, 400
W, K = 25, 10
g_max, width = 0.30, 1.0
b_true = rng.normal(0, 1, n_items)
D = 1.702

def gain(t, bj): return g_max * np.exp(-((t - bj) ** 2) / (2 * width ** 2))
def bern(t, bj): return float(rng.random() < 1 / (1 + np.exp(-(t - bj))))

# --- simulate (same generative model as synthetic_delta_experiment) ---
pre_q, post_q, true_delta, th0s = [], [], [], []
Rrows = []
for _ in range(n_students):
    items = rng.permutation(n_items)[:W + K + W]
    p, k, q = items[:W], items[W:W+K], items[W+K:]
    th0 = rng.normal(0, 1); th = th0
    for j in k: th += gain(th, b_true[j])
    th_after = th
    pre_r  = [bern(th0, b_true[j]) for j in p]
    blk_r  = [bern(th0, b_true[j]) for j in k]     # responses within block at ~th0
    post_r = [bern(th_after, b_true[j]) for j in q]
    pre_q.append((p, pre_r)); post_q.append((q, post_r))
    true_delta.append(th_after - th0); th0s.append(th0)
    Rrows.append((p, pre_r, k, blk_r, q, post_r))
true_delta = np.array(true_delta)

# --- global IRT fit -> item params a, b (frozen) ---
R = -1.0 * np.ones((n_students, n_items))
for i, (p, pr, k, kr, q, qr) in enumerate(Rrows):
    for j, r in zip(p, pr): R[i, j] = r
    for j, r in zip(k, kr): R[i, j] = r
    for j, r in zip(q, qr): R[i, j] = r
cdm = EMIRT(R, n_students, n_items, dim=1, skip_value=-1)
cdm.train(lr=1e-4, epoch=5)
a = cdm.a.reshape(-1); b = cdm.b.reshape(-1); c = cdm.c.reshape(-1)

# --- two theta estimators over the SAME fitted item params ---
def mle_theta(items, resps, iters=60, lr=0.3):
    """plain MLE: maximize log-likelihood of the window responses."""
    t = 0.0
    for _ in range(iters):
        z = a[items] * (t - b[items])
        p = c[items] + (1 - c[items]) / (1 + np.exp(-D * z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        grad = np.sum(D * a[items] * (np.array(resps) - p))   # d loglik / d theta
        t += lr * grad / max(len(items), 1)
    return np.clip(t, -6, 6)

def map_theta(items, resps, prior_sigma=1.0, iters=60, lr=0.3):
    """MAP: same likelihood plus a Normal(0, prior_sigma) prior pulling theta toward 0."""
    t = 0.0
    for _ in range(iters):
        z = a[items] * (t - b[items])
        p = c[items] + (1 - c[items]) / (1 + np.exp(-D * z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        grad = np.sum(D * a[items] * (np.array(resps) - p)) - t / prior_sigma**2   # + prior term
        t += lr * grad / max(len(items), 1)
    return t

def deltas(est):
    tb = np.array([est(p, pr) for p, pr in pre_q])
    ta = np.array([est(q, qr) for q, qr in post_q])
    return ta - tb, tb

for name, est in [("MLE (no prior)", mle_theta), ("MAP (normal prior)", map_theta)]:
    d, tb = deltas(est)
    corr = np.corrcoef(d, true_delta)[0, 1]
    sat = (np.abs(tb) > 3.9).mean()
    print(f"{name:22s}  corr(label,true) {corr:+.3f}   label std {d.std():.2f}   saturated {sat:.2f}")