"""
Synthetic validation with the LEAP model. Same controlled ZPD world and known true Δθ as
synthetic_delta_experiment.py, but the model is now LEAP's architecture instead of the early
block-only DeltaKT: a plain (question, response) embedding, an LSTM over the pre-window + block
history, reading the causal block-end state, plus θ_before. So corr(pred, true) here reflects the
model we actually use. (Synthetic data has no concepts, so LEAP's two branches collapse to one LSTM
over (q, r); a fair simplification.)

    python experiments/synthetic_leap.py 20 10   # W (default 20), K (default 10)
"""
import sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from EduCDM import EMIRT
import logging, warnings
logging.getLogger().setLevel(logging.ERROR); warnings.filterwarnings("ignore")

rng = np.random.default_rng(0); torch.manual_seed(0)

n_students, n_items = 3000, 400
W = int(sys.argv[1]) if len(sys.argv) > 1 else 20
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
g_max, width = 0.30, 1.0
b = rng.normal(0, 1, n_items)

def gain(theta, bj): return g_max * np.exp(-((theta - bj) ** 2) / (2 * width ** 2))
def bern(theta, bj): return (rng.random() < 1 / (1 + np.exp(-(theta - bj)))).astype(float)

# --- 1. simulate students with a known learning rule ---
pre_q, post_q, blk_q, blk_r, theta0_list, true_delta = [], [], [], [], [], []
for _ in range(n_students):
    items = rng.permutation(n_items)[:W + K + W]
    p, k, q = items[:W], items[W:W + K], items[W + K:]
    th0 = rng.normal(0, 1)
    th = th0; br = []
    for j in k:
        br.append(bern(th, b[j])); th += gain(th, b[j])
    pre_q.append(p); post_q.append(q); blk_q.append(k); blk_r.append(br)
    theta0_list.append(th0); true_delta.append(th - th0)

pre_q = np.array(pre_q); blk_q = np.array(blk_q); blk_r = np.array(blk_r)
theta0_arr = np.array(theta0_list); true_delta = np.array(true_delta)

def answers(qs, thetas): return np.array([[bern(t, b[j]) for j in row] for row, t in zip(qs, thetas)])
pre_r  = answers(pre_q,  theta0_arr)
post_r = answers(np.array(post_q), theta0_arr + true_delta)

# --- 2. global IRT fit, freeze item params ---
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
print(f"W={W}  true Δθ std {true_delta.std():.3f}  label Δθ std {delta_label.std():.3f}")
print(f"[check 1] corr(label Δθ, true Δθ) = {c1:+.3f}   (labeling sound?)")

# --- 4. LEAP model: read pre-window + block, take the block-end state ---
seq_q = torch.tensor(np.concatenate([pre_q, blk_q], axis=1), dtype=torch.long)   # [N, W+K]
seq_r = torch.tensor(np.concatenate([pre_r, blk_r], axis=1).astype(int), dtype=torch.long)
theta_b = torch.tensor(tb, dtype=torch.float32)
y_label = torch.tensor(delta_label, dtype=torch.float32)
y_true  = torch.tensor(true_delta, dtype=torch.float32)
bend = W + K - 1                                    # block-end index in the pre+block sequence

idx = torch.randperm(n_students); nv = n_students // 5
vi, ti = idx[:nv], idx[nv:]

class LEAP(nn.Module):
    def __init__(s, n_items, emb=64, hid=64):
        super().__init__()
        s.q = nn.Embedding(n_items, emb)
        s.r = nn.Embedding(2, emb)
        s.lstm = nn.LSTM(2 * emb, hid, batch_first=True)
        s.head = nn.Linear(hid + 1, 1)
    def forward(s, q, r, th):
        x = torch.cat([s.q(q), s.r(r)], -1)         # plain concat embedding (LEAP-style)
        out, _ = s.lstm(x)
        h = out[:, bend, :]                          # causal block-end state (sees pre-window + block)
        return s.head(torch.cat([h, th.unsqueeze(1)], 1)).squeeze(1)

model = LEAP(n_items)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
mse = nn.MSELoss()
tr = DataLoader(TensorDataset(seq_q[ti], seq_r[ti], theta_b[ti], y_label[ti]), batch_size=128, shuffle=True)

best = (1e9, 0, 0)
for ep in range(40):
    model.train()
    for qb, rb, thb, yb in tr:
        opt.zero_grad(); mse(model(qb, rb, thb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pv = model(seq_q[vi], seq_r[vi], theta_b[vi])
    m = mse(pv, y_label[vi]).item()
    if m < best[0]:
        ct = np.corrcoef(pv.numpy(), y_true[vi].numpy())[0, 1]
        cl = np.corrcoef(pv.numpy(), y_label[vi].numpy())[0, 1]
        best = (m, ct, cl)

base = ((y_label[vi] - y_label[ti].mean()) ** 2).mean().item()
print(f"baseline val MSE {base:.3f}   LEAP val MSE {best[0]:.3f}   beats: {best[0] < base}")
print(f"[check 2] corr(LEAP pred, true Δθ)  = {best[1]:+.3f}   (model sound?)")
print(f"          corr(LEAP pred, label Δθ) = {best[2]:+.3f}")