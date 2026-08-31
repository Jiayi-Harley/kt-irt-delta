"""
LEAP, content-only (deployable) variant.

The block's questions and concepts ARE read, but its RESPONSES are NOT: at
recommendation time the block has not been attempted, so its responses do not
exist. Here the block's K responses are masked to the "unknown" token (index 2,
the same token used for padding) while the block's q and c embeddings are kept.
The model therefore knows WHICH block it is scoring, but not how the student did
on it. Everything else is identical to models/plain_delta.py (--emb plain):
same labels, same sequence-level split, same DeltaHead ([h_A,h_B,theta_pre]->dtheta),
same optimiser / epochs / emb size / K, read at block-end (bs+K-1).

Backup/derivative of plain_delta.py; does NOT modify it.

    python models/leap_content.py --emb_size 256 --epochs 15 --seeds 0,1,2,3,4 \
        --labels delta_qikt_nips_task34_question_train_valid_W40K10.npz
"""
import os, json, argparse, numpy as np, torch, torch.nn as nn
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/
PYKT = os.path.join(_ROOT, "pykt-toolkit")
from pykt.datasets.que_data_loader import KTQueDataset

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nips_task34")
ap.add_argument("--labels", default=os.path.join(_ROOT, "delta_qikt_nips_task34_question_train_valid_W40K10.npz"))
ap.add_argument("--emb_size", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.4)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--seeds", default="0")
a = ap.parse_args()
seeds = [int(s) for s in a.seeds.split(",")]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| LEAP content-only (block RESPONSES masked, q/c kept) | emb", a.emb_size, "| seeds", seeds)

lab = np.load(a.labels)
recs = lab["records"]; K = int(lab["K"])
windows = defaultdict(list)                 # row_idx -> [(block_start, theta_pre, delta), ...]
for r in recs:
    windows[int(r[0])].append((int(r[1]), float(r[2]), float(r[3])))
rows_with_win = sorted(windows.keys())
print(f"{len(recs)} windows over {len(rows_with_win)} student rows, K={K}")

DATA = os.path.join(PYKT, "data", a.dataset)
cfg = json.load(open(os.path.join(PYKT, "configs", "data_config.json")))[a.dataset]
NUM_Q, NUM_C = cfg["num_q"], cfg["num_c"]
ds = KTQueDataset(os.path.join(DATA, cfg["train_valid_file_quelevel"]), input_type=cfg["input_type"],
                  folds=[0, 1, 2, 3, 4], concept_num=cfg["num_c"], max_concepts=cfg["max_concepts"])

def rowseq(ridx):
    it = ds[ridx]
    Q = torch.cat([it["qseqs"][0:1], it["shft_qseqs"]]).long()
    C = torch.cat([it["cseqs"][0:1], it["shft_cseqs"]]).long()
    R = torch.cat([it["rseqs"][0:1], it["shft_rseqs"]]).long()
    return Q, C, R
seqcache = {ridx: rowseq(ridx) for ridx in rows_with_win}
L = seqcache[rows_with_win[0]][0].shape[0]

# alignment assertion, identical spirit to plain_delta (dataset block slice == npz block_q)
_Q = seqcache[int(recs[0][0])][0].numpy()
_bs = int(recs[0][1])
assert np.array_equal(_Q[_bs:_bs + K], lab["block_q"][0]), "row_idx alignment broken!"
print("alignment check passed")

# one example per window (with a valid block-end position)
examples_all = []
for ridx in rows_with_win:
    for bs, tb, d in windows[ridx]:
        if 0 <= bs + K - 1 < L:
            examples_all.append((ridx, bs, tb, d))
print(f"{len(examples_all)} usable windows")


class DeltaHead(nn.Module):                 # IDENTICAL to plain_delta: [h_A, h_B, theta_pre] -> dtheta
    def __init__(self, hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * hid + 1, hid), nn.ReLU(), nn.Linear(hid, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class TwoLSTM(nn.Module):                    # IDENTICAL to plain_delta TwoLSTM
    def __init__(self, num_q, num_c, emb, dropout):
        super().__init__()
        self.q = nn.Embedding(num_q + 1, emb)
        self.c = nn.Embedding(num_c + 1, emb)
        self.r = nn.Embedding(3, emb)           # 0,1 responses; 2 = unknown/padding
        self.drop = nn.Dropout(dropout)
        self.lstm_a = nn.LSTM(3 * emb, emb, batch_first=True)   # question-side: [q,c,r]
        self.lstm_b = nn.LSTM(2 * emb, emb, batch_first=True)   # concept-side:  [c,r]
        self.numq, self.numc = num_q, num_c

    def forward(self, Q, C, R):
        if C.dim() == 3:
            C = C[..., 0]
        Qi = torch.where(Q < 0, torch.full_like(Q, self.numq), Q)
        Ci = torch.where(C < 0, torch.full_like(C, self.numc), C)
        Ri = torch.where(R < 0, torch.full_like(R, 2), R)
        qe, ce, re = self.drop(self.q(Qi)), self.drop(self.c(Ci)), self.drop(self.r(Ri))
        ha, _ = self.lstm_a(torch.cat([qe, ce, re], -1))
        hb, _ = self.lstm_b(torch.cat([ce, re], -1))
        return ha, hb


def pcorr(p, y, z):
    rpy, rpz, ryz = np.corrcoef(p, y)[0, 1], np.corrcoef(p, z)[0, 1], np.corrcoef(y, z)[0, 1]
    dd = np.sqrt((1 - rpz ** 2) * (1 - ryz ** 2))
    return (rpy - rpz * ryz) / dd if dd > 1e-9 else 0.0


def build_batch(exs):
    Qs, Cs, Rs, poss, tbs, tgts = [], [], [], [], [], []
    for ridx, bs, tb, d in exs:
        Q, C, R = seqcache[ridx]
        Rm = R.clone()
        Rm[bs:bs + K] = 2                    # <-- THE ONLY change vs plain_delta: mask block RESPONSES
        Qs.append(Q); Cs.append(C); Rs.append(Rm)
        poss.append(bs + K - 1); tbs.append(tb); tgts.append(d)
    Q = torch.stack(Qs).to(device); C = torch.stack(Cs).to(device); R = torch.stack(Rs).to(device)
    pos = torch.tensor(poss, device=device)
    tb = torch.tensor(tbs, dtype=torch.float, device=device)
    y = torch.tensor(tgts, dtype=torch.float, device=device)
    return Q, C, R, pos, tb, y


def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(rows_with_win)
    nv = len(perm) // 5
    val_rows = set(int(x) for x in perm[:nv])     # sequence-level split, like plain_delta
    tr = [e for e in examples_all if e[0] not in val_rows]
    va = [e for e in examples_all if e[0] in val_rows]
    HID = a.emb_size
    enc = TwoLSTM(NUM_Q, NUM_C, HID, a.dropout).to(device)
    dh = DeltaHead(HID).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dh.parameters()), lr=1e-3, weight_decay=1e-5)
    mse = nn.MSELoss()

    best_mse, best = 1e9, (0.0, 0.0)
    for ep in range(a.epochs):
        enc.train(); dh.train(); rng.shuffle(tr)
        for i in range(0, len(tr), a.batch_size):
            exs = tr[i:i + a.batch_size]
            Q, C, R, pos, tb, y = build_batch(exs)
            opt.zero_grad()
            HA, HB = enc(Q, C, R)
            idx = torch.arange(len(exs), device=device)
            loss = mse(dh(torch.cat([HA[idx, pos], HB[idx, pos], tb.unsqueeze(1)], 1)), y)
            loss.backward(); opt.step()
        enc.eval(); dh.eval()
        P, T, Z = [], [], []
        with torch.no_grad():
            for i in range(0, len(va), a.batch_size):
                exs = va[i:i + a.batch_size]
                Q, C, R, pos, tb, y = build_batch(exs)
                HA, HB = enc(Q, C, R)
                idx = torch.arange(len(exs), device=device)
                dp = dh(torch.cat([HA[idx, pos], HB[idx, pos], tb.unsqueeze(1)], 1))
                P.append(dp.cpu().numpy()); T.append(y.cpu().numpy()); Z.append(tb.cpu().numpy())
        P, T, Z = np.concatenate(P), np.concatenate(T), np.concatenate(Z)
        vm = float(np.mean((P - T) ** 2))
        c_ep, pc_ep = np.corrcoef(P, T)[0, 1], pcorr(P, T, Z)
        print(f"  seed {seed} ep {ep:>2} | val corr {c_ep:+.3f}  pcorr(block) {pc_ep:+.3f}", flush=True)
        if vm < best_mse:
            best_mse = vm; best = (c_ep, pc_ep)
    return best


print("\nseed   raw_corr   block_signal")
res = []
for s in seeds:
    c, pc = run(s); res.append((c, pc))
    print(f"{s:>4}   {c:>+7.3f}   {pc:>+7.3f}")
cs = np.array([c for c, _ in res]); pcs = np.array([pc for _, pc in res])
print(f"mean   {cs.mean():>+7.3f}   {pcs.mean():>+7.3f}")
print(f"std    {cs.std():>7.3f}   {pcs.std():>7.3f}")