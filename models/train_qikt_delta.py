"""
Predict Δθ from a block. Compare, on the SAME nips labels and split, three configs that differ only
in encoder and/or prediction head, so each difference is cleanly isolated:
  1. plain encoder      + simple head   (a plain DeltaKT baseline)
  2. QIKT encoder       + simple head   (QIKT's two knowledge-state modules KA+KS)
  3. QIKT encoder       + weighted head (QIKT-style: each module -> its own MLP -> scalar,
                                         learnable-weighted additive sum; no sigmoid, Δθ is regression)
Repeats over several seeds to separate a real difference from seed noise.
QIKT's own QueEmb / LSTM structure is reused; qikt.py is untouched.

    python models/train_qikt_delta.py delta_qikt_nips_task34_question_train_valid_W40K10.npz 0,1,2,3,4
"""
import sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pykt.models.que_base_model import QueEmb

npz = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
d = np.load(npz)
block_q  = torch.tensor(d["block_q"],  dtype=torch.long)
block_cm = torch.tensor(d["block_cm"], dtype=torch.long)
block_r  = torch.tensor(d["block_r"],  dtype=torch.long)
# 做法2 (exp_log §18): an honest Δθ predictor sees the pre-window history + the block (everything
# available BEFORE the outcome), never the post-window (that would leak θ_after). So feed pre+block.
pre_q  = torch.tensor(d["pre_q"],  dtype=torch.long)
pre_cm = torch.tensor(d["pre_cm"], dtype=torch.long)
pre_r  = torch.tensor(d["pre_r"],  dtype=torch.long)
seq_q  = torch.cat([pre_q,  block_q],  dim=1)   # [N, W+K]  question ids
seq_cm = torch.cat([pre_cm, block_cm], dim=1)   # [N, W+K, max_c]  concepts
seq_r  = torch.cat([pre_r,  block_r],  dim=1)   # [N, W+K]  responses
theta_b  = torch.tensor(d["theta_before"], dtype=torch.float32)
delta    = torch.tensor(d["delta"],    dtype=torch.float32)
# ablation (exp_log §18): pass "notheta" as 3rd arg to zero out θ_before, leaving pre-window + block
# only. Tests whether the raw history alone recovers the score (θ_before redundant) or not.
if len(sys.argv) > 3 and sys.argv[3] == "notheta":
    theta_b = torch.zeros_like(theta_b)
    print("ABLATION: theta_before zeroed (input = pre-window history + block only)")
num_q, num_c = int(d["num_q"]), int(d["num_c"])
N, K = block_q.shape
# emb 64, not 256: this is a small regression (~7700 train windows). Capacity must match the data
# fit to it; emb 256 overfits here (val corr 0.28) while QIKT's correctness task (~1M targets) needs 256.
EMB, HID = 64, 64
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---- encoders: return a LIST of per-module block-end states ----
class PlainEncoder(nn.Module):    # DeltaKT-style: one interaction embedding per (question, response)
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_q * 2, EMB); self.lstm = nn.LSTM(EMB, HID, batch_first=True)
        self.states = 1
    def forward(self, q, cm, r):
        _, (h, _) = self.lstm(self.emb(q * 2 + r)); return [h.squeeze(0)]

class QIKTEncoder(nn.Module):     # QIKT's two knowledge-state modules: KA (question) + KS (concept)
    def __init__(self):
        super().__init__()
        self.que_emb = QueEmb(num_q=num_q, num_c=num_c, emb_size=EMB, emb_type="iekt",
                              model_name="qikt", device=device, emb_path="", pretrain_dim=768)
        self.que_lstm = nn.LSTM(EMB * 4, HID, batch_first=True)      # KA
        self.concept_lstm = nn.LSTM(EMB * 2, HID, batch_first=True)  # KS
        self.states = 2
    def forward(self, q, cm, r):
        _, emb_qca, _, _, emb_c = self.que_emb(q, cm, r.float())
        _, (ha, _) = self.que_lstm(emb_qca)
        rr = r.float().unsqueeze(-1)
        emb_ca = torch.cat([emb_c * (1 - rr), emb_c * rr], dim=-1)
        _, (hg, _) = self.concept_lstm(emb_ca)
        return [ha.squeeze(0), hg.squeeze(0)]

# ---- heads ----
class SimpleHead(nn.Module):      # concat all module states + theta_before -> one linear layer
    def __init__(self, n_states):
        super().__init__(); self.lin = nn.Linear(n_states * HID + 1, 1)
    def forward(self, states, tb):
        return self.lin(torch.cat(states + [tb.unsqueeze(1)], dim=1)).squeeze(1)

class WeightedHead(nn.Module):    # QIKT-style: each module -> its own MLP -> scalar; learnable weighted sum
    def __init__(self, n_states):
        super().__init__()
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1))
                                   for _ in range(n_states)])
        self.w = nn.Parameter(torch.ones(n_states))
        self.w_theta = nn.Parameter(torch.zeros(1)); self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, states, tb):
        scores = torch.stack([m(s).squeeze(1) for m, s in zip(self.mlps, states)], dim=1)  # [B, n_states]
        return (scores * self.w).sum(1) + self.w_theta * tb + self.bias

class DeltaNet(nn.Module):
    def __init__(self, encoder, head):
        super().__init__(); self.enc = encoder; self.head = head
    def forward(self, q, cm, r, tb):
        return self.head(self.enc(q, cm, r), tb)

mse = nn.MSELoss()

def pcorr(p, y, z):   # partial correlation of pred p and label y, controlling for θ_before z
    rpy, rpz, ryz = np.corrcoef(p, y)[0, 1], np.corrcoef(p, z)[0, 1], np.corrcoef(y, z)[0, 1]
    d = np.sqrt((1 - rpz ** 2) * (1 - ryz ** 2))
    return (rpy - rpz * ryz) / d if d > 1e-9 else 0.0

def train_eval(encoder_cls, head_cls, tr_i, val_i, seed):
    torch.manual_seed(seed)
    enc = encoder_cls(); model = DeltaNet(enc, head_cls(enc.states)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    def dl(ii, sh):
        return DataLoader(TensorDataset(seq_q[ii], seq_cm[ii], seq_r[ii], theta_b[ii], delta[ii]),
                          batch_size=128, shuffle=sh)
    tr_dl, va_dl = dl(tr_i, True), dl(val_i, False)
    tb_val = theta_b[val_i].numpy()          # val θ_before, same order as va_dl (shuffle=False)
    best_mse, best_corr, best_pcorr = 1e9, 0.0, 0.0
    for _ in range(40):
        model.train()
        for q, cm, r, tb, y in tr_dl:
            q, cm, r, tb, y = q.to(device), cm.to(device), r.to(device), tb.to(device), y.to(device)
            opt.zero_grad(); mse(model(q, cm, r, tb), y).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.cat([model(q.to(device), cm.to(device), r.to(device), tb.to(device)).cpu()
                           for q, cm, r, tb, _ in va_dl])
            yt = torch.cat([y for *_, y in va_dl])
        vm = mse(p, yt).item()
        if vm < best_mse:
            pn, yn = p.numpy(), yt.numpy()
            best_mse, best_corr, best_pcorr = vm, np.corrcoef(pn, yn)[0, 1], pcorr(pn, yn, tb_val)
    return best_corr, best_pcorr

CONFIGS = [("plain+simple", PlainEncoder, SimpleHead),
           ("QIKT+simple", QIKTEncoder, SimpleHead),
           ("QIKT+weighted", QIKTEncoder, WeightedHead)]

print(f"N={N} windows, K={K}, num_q={num_q}  |  seeds={seeds}")
print("seed  " + "  ".join(f"{n:>14}" for n, _, _ in CONFIGS))
res = {n: [] for n, _, _ in CONFIGS}
resp = {n: [] for n, _, _ in CONFIGS}
for s in seeds:
    torch.manual_seed(s); np.random.seed(s)
    g = torch.Generator().manual_seed(s)
    idx = torch.randperm(N, generator=g); nv = N // 5
    val_i, tr_i = idx[:nv], idx[nv:]
    row = []
    for n, ec, hc in CONFIGS:
        c, pc = train_eval(ec, hc, tr_i, val_i, s); res[n].append(c); resp[n].append(pc); row.append((c, pc))
    print(f"{s:>4}  " + "  ".join(f"{c:>+7.3f}/{pc:>+6.3f}" for c, pc in row))
print("corr  mean " + "  ".join(f"{np.mean(res[n]):>+11.3f}" for n, _, _ in CONFIGS))
print("pcorr mean " + "  ".join(f"{np.mean(resp[n]):>+11.3f}" for n, _, _ in CONFIGS) + "   <- block signal with θ_before controlled")