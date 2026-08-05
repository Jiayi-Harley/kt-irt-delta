"""
IRT hands-on (EduCDM EM 3PL). Synthetic data generated with the SAME 3PL formula
EduCDM fits (a, b, c all present), so the generative model matches the fitting model.
KNOWN truth, so we can check IRT recovers student ability (theta) and item difficulty (b).

Three experiments:
  1. more epochs        -> does the fit get better?
  2. more items         -> does each student's theta get more accurate?
  3. difficulty recovery -> does IRT also recover how hard each item is?

Note: IRT is only identified up to a shift, so we compare via CORRELATION
(who is stronger/harder), not absolute values.
"""
import numpy as np
from EduCDM import EMIRT
import logging
logging.getLogger().setLevel(logging.ERROR)   # silence the per-step INFO logs


D = 1.702   # scaling constant used in EduCDM's 3PL, kept identical here

def run(n_students=200, n_items=50, epoch=5, seed=0):
    rng = np.random.default_rng(seed)
    true_theta = rng.normal(0, 1, n_students)          # true ability
    true_a     = rng.uniform(0.5, 2.0, n_items)        # true discrimination
    true_b     = rng.normal(0, 1, n_items)             # true difficulty
    true_c     = rng.uniform(0.0, 0.25, n_items)       # true guessing rate
    # 3PL, same formula EduCDM fits with:  c + (1-c) * sigmoid(D * a * (theta - b))
    z = D * true_a[None, :] * (true_theta[:, None] - true_b[None, :])
    p = true_c[None, :] + (1 - true_c[None, :]) / (1 + np.exp(-z))
    R = (rng.random((n_students, n_items)) < p).astype(float)

    cdm = EMIRT(R, n_students, n_items, dim=1, skip_value=-1)
    cdm.train(lr=1e-3, epoch=epoch)

    fitted_theta = np.array([cdm.transform(R[s]).item() for s in range(n_students)])
    fitted_b     = cdm.b.reshape(-1)

    theta_corr = np.corrcoef(fitted_theta, true_theta)[0, 1]
    b_corr     = np.corrcoef(fitted_b, true_b)[0, 1]
    return theta_corr, b_corr


print(f"{'setting':<28}{'theta corr':>12}{'difficulty corr':>18}")
print("-" * 58)

# baseline
tc, bc = run(n_items=50, epoch=5)
print(f"{'50 items, 5 epochs':<28}{tc:>12.3f}{bc:>18.3f}")

# experiment 1: more epochs
tc, bc = run(n_items=50, epoch=20)
print(f"{'50 items, 20 epochs':<28}{tc:>12.3f}{bc:>18.3f}")

# experiment 2: more items (keep the longer training)
tc, bc = run(n_items=200, epoch=20)
print(f"{'200 items, 20 epochs':<28}{tc:>12.3f}{bc:>18.3f}")
