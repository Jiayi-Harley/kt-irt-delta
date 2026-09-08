"""
方向① 多任务共训 (multi-task, exp_log §18 / project03_notes §30). 方案 A + X:

  A (long sequences): the REAL QIKT correctness task is trained on the full student sequences (pyKT's
     normal regime, dense ~1M targets). A Δθ head reads the causal block-end state and regresses Δθ,
     supervised only at the block-end positions of the Δθ windows.
  X (placement): we do NOT modify pykt-toolkit. We import the real QIKTNet as the backbone, hook its
     block-end hidden states (KA a_t, KS g_t) WITHOUT detach so gradients flow, add our own Δθ head,
     and run our own training loop here in code/. The toolkit stays pristine.

Total loss = QIKT's own correctness + auxiliary loss (train_one_step) + λ · Δθ MSE.

Leakage discipline (verified: QIKT's LSTMs are unidirectional, no attention): the Δθ head reads the
hidden state at the block-END output index (bs+K-1), which by causality depends only on the pre-window
+ block, never the post-window. The correctness CE spans the whole sequence but only as TARGETS that
shape the shared weights, never as an input to a window's own Δθ prediction.

Metrics: raw corr(pred, Δθ) is the headline; partial corr controlling θ_before is the block-signal.

    python models/qikt_delta_mt.py --emb_size 256 --seeds 0,1,2,3,4,5,6,7,8,9
"""
import os, json, argparse, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/
PYKT = os.path.join(_ROOT, "pykt-toolkit")
from pykt.datasets.que_data_loader import KTQueDataset
from pykt.models.init_model import init_model

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nips_task34")
ap.add_argument("--labels", default=os.path.join(_ROOT, "delta_qikt_nips_task34_question_train_valid_W40K10.npz"))
ap.add_argument("--emb_size", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.4)
ap.add_argument("--mlp_layer_num", type=int, default=2)
ap.add_argument("--lam", type=float, default=1.0, help="weight on the Δθ MSE term")
ap.add_argument("--corr_w", type=float, default=1.0, help="weight on QIKT correctness loss; 0 = Δθ-only control (no multi-task)")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--seeds", default="0")
ap.add_argument("--beta", type=float, default=0.0, help="③ distillation weight; 0 = no distillation (= ①)")
ap.add_argument("--teacher", default="", help="③ teacher npz (KA/KS-change per window); needed if --beta>0")
a = ap.parse_args()
seeds = [int(s) for s in a.seeds.split(",")]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| emb", a.emb_size, "| lam", a.lam, "| beta", a.beta, "| seeds", seeds)

# ---- labels: group Δθ windows by row_idx (+ ③ teacher KA/KS-change per window) ----
lab = np.load(a.labels)
recs = lab["records"]; K = int(lab["K"])
teacher_arr = np.load(a.teacher)["teacher"] if a.teacher else None   # [N,2] = (KA-change, KS-change), NaN where none
windows = defaultdict(list)                 # row_idx -> [(block_start, θ_before, δ, KA_change, KS_change), ...]
for i, r in enumerate(recs):
    tka, tks = (float(teacher_arr[i, 0]), float(teacher_arr[i, 1])) if teacher_arr is not None else (np.nan, np.nan)
    windows[int(r[0])].append((int(r[1]), float(r[2]), float(r[3]), tka, tks))
rows_with_win = sorted(windows.keys())
print(f"{len(recs)} windows over {len(rows_with_win)} student rows")

# ---- full student sequences (aligned: dataset[i] corresponds to row_idx i) ----
DATA = os.path.join(PYKT, "data", a.dataset)
cfg = json.load(open(os.path.join(PYKT, "configs", "data_config.json")))[a.dataset]
ds = KTQueDataset(os.path.join(DATA, cfg["train_valid_file_quelevel"]), input_type=cfg["input_type"],
                  folds=[0, 1, 2, 3, 4], concept_num=cfg["num_c"], max_concepts=cfg["max_concepts"])
# alignment assertion: the block slice of dataset[row_idx] must equal the npz block_q
_it = ds[int(recs[0][0])]
_Q = torch.cat([_it["qseqs"][0:1], _it["shft_qseqs"]]).numpy()
_bs = int(recs[0][1])
assert np.array_equal(_Q[_bs:_bs + K], lab["block_q"][0]), "row_idx alignment broken!"
print("alignment check passed")

OTHER = {"output_mode": "an_irt", "loss_q_all_lambda": 1, "loss_c_all_lambda": 1, "loss_c_next_lambda": 1,
         "loss_q_next_lambda": 0, "output_q_all_lambda": 1, "output_c_all_lambda": 1,
         "output_q_next_lambda": 0, "output_c_next_lambda": 1}

class DeltaHead(nn.Module):                 # [a_t, g_t, θ_before] -> Δθ
    def __init__(self, hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * hid + 1, hid), nn.ReLU(), nn.Linear(hid, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def pcorr(p, y, z):                         # partial corr of p,y controlling z
    rpy, rpz, ryz = np.corrcoef(p, y)[0, 1], np.corrcoef(p, z)[0, 1], np.corrcoef(y, z)[0, 1]
    d = np.sqrt((1 - rpz ** 2) * (1 - ryz ** 2))
    return (rpy - rpz * ryz) / d if d > 1e-9 else 0.0

def gather(cap, row_idxs):                  # block-end a, g, θ_before, δ, teacher KA/KS-change per window
    a_, g_ = cap["a"], cap["g"]             # [B, seqlen-1, HID]
    L = a_.shape[1]
    A, G, tbs, tgts, tka, tks = [], [], [], [], [], []
    for b, ridx in enumerate(row_idxs):
        for bs, tb, d, ka_c, ks_c in windows[ridx]:
            pos = bs + K - 1                # block-end output index (causal: pre-window + block only)
            if pos >= L:
                continue
            A.append(a_[b, pos]); G.append(g_[b, pos])
            tbs.append(tb); tgts.append(d); tka.append(ka_c); tks.append(ks_c)
    A, G = torch.stack(A), torch.stack(G)
    return A, G, A.new_tensor(tbs), A.new_tensor(tgts), np.array(tka), np.array(tks)

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(rows_with_win)
    nv = len(perm) // 5
    val_idx = sorted(int(x) for x in perm[:nv])
    tr_idx = [int(x) for x in perm[nv:]]        # Subset(shuffle=False) yields in this exact order

    model = init_model("qikt", dict(emb_size=a.emb_size, dropout=a.dropout,
                                     mlp_layer_num=a.mlp_layer_num, other_config=OTHER), cfg, "iekt")
    HID = a.emb_size
    dhead = DeltaHead(HID).to(device)
    aux_ka = nn.Linear(HID, 1).to(device)   # ③: block-end KA state a_t -> predict teacher KA-change
    aux_ks = nn.Linear(HID, 1).to(device)   # ③: block-end KS state g_t -> predict teacher KS-change
    cap = {}
    model.model.que_lstm_layer.register_forward_hook(     lambda m, i, o: cap.__setitem__("a", o[0]))
    model.model.concept_lstm_layer.register_forward_hook( lambda m, i, o: cap.__setitem__("g", o[0]))
    opt = torch.optim.Adam(list(model.model.parameters()) + list(dhead.parameters())
                           + list(aux_ka.parameters()) + list(aux_ks.parameters()),
                           lr=1e-3, weight_decay=1e-5)
    mse = nn.MSELoss()
    tr_loader  = DataLoader(Subset(ds, tr_idx),  batch_size=a.batch_size, shuffle=False)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=a.batch_size, shuffle=False)

    best_mse, bestPTZ = 1e9, None
    for ep in range(a.epochs):
        model.model.train(); dhead.train(); ptr = 0
        for data in tr_loader:
            B = data["qseqs"].shape[0]; ridxs = tr_idx[ptr:ptr + B]; ptr += B
            opt.zero_grad()
            _, qikt_loss = model.train_one_step(data)          # correctness+aux loss; hooks fill cap
            A, G, tb, y, tka, tks = gather(cap, ridxs)
            loss = a.corr_w * qikt_loss + a.lam * mse(dhead(torch.cat([A, G, tb.unsqueeze(1)], 1)), y)
            if a.beta > 0:                                     # ③ distillation, only where teacher exists
                m = ~np.isnan(tka)
                if m.any():
                    mm = torch.tensor(m, device=device)
                    tk = torch.tensor(tka[m], dtype=torch.float32, device=device)
                    ts = torch.tensor(tks[m], dtype=torch.float32, device=device)
                    loss = loss + a.beta * (mse(aux_ka(A[mm]).squeeze(-1), tk)
                                            + mse(aux_ks(G[mm]).squeeze(-1), ts))
            loss.backward(); opt.step()
        model.model.eval(); dhead.eval(); ptr = 0
        P, T, Z = [], [], []
        with torch.no_grad():
            for data in val_loader:
                B = data["qseqs"].shape[0]; ridxs = val_idx[ptr:ptr + B]; ptr += B
                model.predict_one_step(data, return_details=True)   # fills cap
                A, G, tb, y, _, _ = gather(cap, ridxs)
                dpred = dhead(torch.cat([A, G, tb.unsqueeze(1)], 1))
                P.append(dpred.cpu().numpy()); T.append(y.cpu().numpy()); Z.append(tb.cpu().numpy())
        P, T, Z = np.concatenate(P), np.concatenate(T), np.concatenate(Z)
        vm = float(np.mean((P - T) ** 2))
        c_ep, pc_ep = np.corrcoef(P, T)[0, 1], pcorr(P, T, Z)
        print(f"  seed {seed} ep {ep:>2} | val corr {c_ep:+.3f}  pcorr(block) {pc_ep:+.3f}", flush=True)
        if vm < best_mse:
            best_mse = vm
            bestPTZ = (P.copy(), T.copy(), Z.copy())
    return bestPTZ

import re as _re
from _delta_metrics import finalize
_m = _re.search(r"W(\d+)K(\d+)", a.labels)
_W = int(_m.group(1)) if _m else -1
_task = "multi" if a.corr_w > 0 else "single"
_tag = f"QIKT-{_task}" + (f"-beta{a.beta}" if a.beta > 0 else "")
_meta = {"model": _tag, "W": _W, "K": int(K), "emb": a.emb_size, "corr_w": a.corr_w,
         "lam": a.lam, "beta": a.beta, "dataset": a.dataset}
_out = os.path.join(_ROOT, "results_metrics.jsonl")
_ptz = [run(s) for s in seeds]
finalize(_out, _tag, _meta, _ptz)