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
a = ap.parse_args()
seeds = [int(s) for s in a.seeds.split(",")]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| emb", a.emb_size, "| lam", a.lam, "| seeds", seeds)

# ---- labels: group Δθ windows by row_idx ----
lab = np.load(a.labels)
recs = lab["records"]; K = int(lab["K"])
windows = defaultdict(list)                 # row_idx -> [(block_start, theta_before, delta), ...]
for r in recs:
    windows[int(r[0])].append((int(r[1]), float(r[2]), float(r[3])))
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

def gather(cap, row_idxs):                  # pull block-end [a,g,θ_before] for every window in the batch
    a_, g_ = cap["a"], cap["g"]             # [B, seqlen-1, HID]
    L = a_.shape[1]
    feats, tgts, tbs = [], [], []
    for b, ridx in enumerate(row_idxs):
        for bs, tb, d in windows[ridx]:
            pos = bs + K - 1                # block-end output index (causal: pre-window + block only)
            if pos >= L:
                continue
            feats.append(torch.cat([a_[b, pos], g_[b, pos], a_.new_tensor([tb])]))
            tgts.append(d); tbs.append(tb)
    return torch.stack(feats), a_.new_tensor(tgts), np.array(tbs)

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
    cap = {}
    model.model.que_lstm_layer.register_forward_hook(     lambda m, i, o: cap.__setitem__("a", o[0]))
    model.model.concept_lstm_layer.register_forward_hook( lambda m, i, o: cap.__setitem__("g", o[0]))
    opt = torch.optim.Adam(list(model.model.parameters()) + list(dhead.parameters()),
                           lr=1e-3, weight_decay=1e-5)
    mse = nn.MSELoss()
    tr_loader  = DataLoader(Subset(ds, tr_idx),  batch_size=a.batch_size, shuffle=False)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=a.batch_size, shuffle=False)

    best_mse, best = 1e9, (0.0, 0.0)
    for ep in range(a.epochs):
        model.model.train(); dhead.train(); ptr = 0
        for data in tr_loader:
            B = data["qseqs"].shape[0]; ridxs = tr_idx[ptr:ptr + B]; ptr += B
            opt.zero_grad()
            _, qikt_loss = model.train_one_step(data)          # correctness+aux loss; hooks fill cap
            f, t, _ = gather(cap, ridxs)
            loss = a.corr_w * qikt_loss + a.lam * mse(dhead(f), t)
            loss.backward(); opt.step()
        model.model.eval(); dhead.eval(); ptr = 0
        P, T, Z = [], [], []
        with torch.no_grad():
            for data in val_loader:
                B = data["qseqs"].shape[0]; ridxs = val_idx[ptr:ptr + B]; ptr += B
                model.predict_one_step(data, return_details=True)   # fills cap
                f, t, tb = gather(cap, ridxs)
                P.append(dhead(f).cpu().numpy()); T.append(t.cpu().numpy()); Z.append(tb)
        P, T, Z = np.concatenate(P), np.concatenate(T), np.concatenate(Z)
        vm = float(np.mean((P - T) ** 2))
        c_ep, pc_ep = np.corrcoef(P, T)[0, 1], pcorr(P, T, Z)
        print(f"  seed {seed} ep {ep:>2} | val corr {c_ep:+.3f}  pcorr(block) {pc_ep:+.3f}", flush=True)
        if vm < best_mse:
            best_mse = vm
            best = (c_ep, pc_ep)
    return best

print(f"\nseed   raw_corr   block_signal(pcorr)")
res = []
for s in seeds:
    c, pc = run(s); res.append((c, pc))
    print(f"{s:>4}   {c:>+7.3f}   {pc:>+7.3f}")
cs = np.array([c for c, _ in res]); pcs = np.array([pc for _, pc in res])
print(f"mean   {cs.mean():>+7.3f}   {pcs.mean():>+7.3f}")
print(f"std    {cs.std():>7.3f}   {pcs.std():>7.3f}")