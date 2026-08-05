"""
Controlled synthetic experiment: is the Δθ method sound, or is assist2015 just bad data?

We simulate students with a KNOWN learning rule, so we know each block's TRUE Δθ.
Learning rule (ZPD / desirable difficulty): answering a question near your current
ability gives the most learning; too easy or too hard gives little.
    gain(θ, b) = g_max * exp( -(θ - b)^2 / (2*width^2) )
During the block, θ grows by the sum of these gains. That sum is the true Δθ.
Pre-window and post-window are pure assessment: they measure θ but cause no learning.

Then we run the SAME pipeline as on real data (global IRT fit, windowed θ, Δθ labels,
DeltaKT model) and check two correlations against the TRUE Δθ:
  1. does the windowed Δθ LABEL recover true Δθ?   (is the labeling method sound?)
  2. does the MODEL prediction recover true Δθ?      (is the model sound?)
"""
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from EduCDM import EMIRT
import logging, warnings
logging.getLogger().setLevel(logging.ERROR); warnings.filterwarnings("ignore")

import sys
rng = np.random.default_rng(0); torch.manual_seed(0)

# --- settings ---
n_students, n_items = 3000, 400          # more items so bigger windows have distinct questions
W = int(sys.argv[1]) if len(sys.argv) > 1 else 20
K = 10
g_max, width = 0.30, 1.0
b = rng.normal(0, 1, n_items)                      # each question's difficulty

def gain(theta, bj):
    return g_max * np.exp(-((theta - bj) ** 2) / (2 * width ** 2))

def bern(theta, bj):
    return (rng.random() < 1 / (1 + np.exp(-(theta - bj)))).astype(float)

# --- 1. simulate students with known learning ---
pre_q, post_q, blk_q, blk_r, theta0_list, true_delta = [], [], [], [], [], []
for _ in range(n_students):
    items = rng.permutation(n_items)[:W + K + W]    # distinct questions for this student
    p, k, q = items[:W], items[W:W+K], items[W+K:]
    th0 = rng.normal(0, 1)
    # block: answer + learn
    th = th0; br = []
    for j in k:
        br.append(bern(th, b[j]))
        th += gain(th, b[j])
    th_after = th
    pre_q.append(p); post_q.append(q); blk_q.append(k); blk_r.append(br)
    theta0_list.append(th0); true_delta.append(th_after - th0)

blk_q = np.array(blk_q); blk_r = np.array(blk_r)
theta0_arr = np.array(theta0_list); true_delta = np.array(true_delta)

# responses for pre/post windows (assessment, no learning) and to build the global matrix
def answers(qs, thetas):
    return np.array([[bern(t, b[j]) for j in row] for row, t in zip(qs, thetas)])
pre_r  = answers(pre_q,  theta0_arr)
post_r = answers(post_q, theta0_arr + true_delta)     # post measured at the new ability

# --- 2. global IRT fit -> freeze item params ---
R = -1.0 * np.ones((n_students, n_items))
for i in range(n_students):
    for j, r in zip(pre_q[i],  pre_r[i]):  R[i, j] = r
    for j, r in zip(blk_q[i],  blk_r[i]):  R[i, j] = r
    for j, r in zip(post_q[i], post_r[i]): R[i, j] = r
cdm = EMIRT(R, n_students, n_items, dim=1, skip_value=-1)
cdm.train(lr=1e-4, epoch=5)

def theta_win(qs, rs):
    v = -1.0 * np.ones(n_items)
    for j, r in zip(qs, rs): v[j] = r
    return cdm.transform(v).item()

# --- 3. windowed Δθ labels, check vs true ---
tb = np.array([theta_win(pre_q[i],  pre_r[i])  for i in range(n_students)])
ta = np.array([theta_win(post_q[i], post_r[i]) for i in range(n_students)])
delta_label = ta - tb

c1 = np.corrcoef(delta_label, true_delta)[0, 1]
print(f"true Δθ   mean {true_delta.mean():+.3f} std {true_delta.std():.3f}")
print(f"label Δθ  mean {delta_label.mean():+.3f} std {delta_label.std():.3f}")
print(f"[check 1] corr(label Δθ, true Δθ) = {c1:+.3f}   (is the labeling sound?)")

# --- 4. train DeltaKT, check prediction vs true ---
inter = torch.tensor(blk_q * 2 + blk_r.astype(int), dtype=torch.long)
theta_b = torch.tensor(tb, dtype=torch.float32)
y_label = torch.tensor(delta_label, dtype=torch.float32)
y_true  = torch.tensor(true_delta, dtype=torch.float32)

idx = torch.randperm(n_students); nv = n_students // 5
vi, ti = idx[:nv], idx[nv:]

class DeltaKT(nn.Module):
    def __init__(s, n_inter, emb=64, hid=64):
        super().__init__()
        s.emb = nn.Embedding(n_inter, emb); s.lstm = nn.LSTM(emb, hid, batch_first=True)
        s.head = nn.Linear(hid + 1, 1)
    def forward(s, inter, th):
        _, (h, _) = s.lstm(s.emb(inter)); h = h.squeeze(0)
        return s.head(torch.cat([h, th.unsqueeze(1)], 1)).squeeze(1)

model = DeltaKT(2 * n_items)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
mse = nn.MSELoss()
tr = DataLoader(TensorDataset(inter[ti], theta_b[ti], y_label[ti]), batch_size=128, shuffle=True)

best = (1e9, 0, 0)
for ep in range(40):
    model.train()
    for xb, thb, yb in tr:
        opt.zero_grad(); mse(model(xb, thb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pv = model(inter[vi], theta_b[vi])
    m = mse(pv, y_label[vi]).item()
    if m < best[0]:
        ct = np.corrcoef(pv.numpy(), y_true[vi].numpy())[0, 1]     # vs TRUE
        cl = np.corrcoef(pv.numpy(), y_label[vi].numpy())[0, 1]    # vs label
        best = (m, ct, cl)

base = ((y_label[vi] - y_label[ti].mean()) ** 2).mean().item()
print(f"\nbaseline val MSE {base:.3f}   model val MSE {best[0]:.3f}   beats: {best[0] < base}")
print(f"[check 2] corr(model pred, true Δθ)  = {best[1]:+.3f}   (is the model sound?)")
print(f"          corr(model pred, label Δθ) = {best[2]:+.3f}")