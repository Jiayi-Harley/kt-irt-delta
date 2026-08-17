"""
Capacity-matched GENERIC baseline for the Δθ evaluator, to answer "do we need QIKT's structure".

Everything is held identical to models/qikt_delta_mt.py single-task (corr_w=0): same labels, same
sequence-level train/val split, same 5 seeds, same DeltaHead, same training loop / optimizer / epochs /
emb size / K. The ONLY change is the encoder. Here it is two plain LSTMs (a question-side branch and a
concept-side branch) over simple embeddings, matching QIKT's two-branch capacity but WITHOUT QueEmb,
the IRT layer, the readouts, or the auxiliary losses. So a match here means QIKT's specific machinery
is not what produces the raw correlation; a generic two-LSTM sequence model over the full history does.

    python models/plain_delta.py --emb_size 256 --epochs 15 --seeds 0,1,2,3,4 \
        --labels delta_qikt_nips_task34_question_train_valid_W40K10.npz
"""
import os, json, argparse, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/
PYKT = os.path.join(_ROOT, "pykt-toolkit")
from pykt.datasets.que_data_loader import KTQueDataset
from pykt.models.que_base_model import QueEmb

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nips_task34")
ap.add_argument("--labels", default=os.path.join(_ROOT, "delta_qikt_nips_task34_question_train_valid_W40K10.npz"))
ap.add_argument("--emb_size", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.4)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--seeds", default="0")
ap.add_argument("--read_at", default="blockend", choices=["blockend", "preblock"],
                help="which position the Δθ head reads: block-end (bs+K-1) or pre-block (bs-1, causal sanity check)")
ap.add_argument("--emb", default="plain", choices=["plain", "queemb"],
                help="plain = generic q/c/r embeddings; queemb = QIKT's exact QueEmb + KA/KS encoder in this same pipeline")
a = ap.parse_args()
seeds = [int(s) for s in a.seeds.split(",")]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| DeltaKT baseline | encoder:", a.emb, "| read_at:", a.read_at,
      "| emb", a.emb_size, "| seeds", seeds)

# ---- labels: group Δθ windows by row_idx (same as qikt_delta_mt) ----
lab = np.load(a.labels)
recs = lab["records"]; K = int(lab["K"])
windows = defaultdict(list)                 # row_idx -> [(block_start, θ_before, δ), ...]
for r in recs:
    windows[int(r[0])].append((int(r[1]), float(r[2]), float(r[3])))
rows_with_win = sorted(windows.keys())
print(f"{len(recs)} windows over {len(rows_with_win)} student rows, K={K}")

DATA = os.path.join(PYKT, "data", a.dataset)
cfg = json.load(open(os.path.join(PYKT, "configs", "data_config.json")))[a.dataset]
NUM_Q, NUM_C = cfg["num_q"], cfg["num_c"]
ds = KTQueDataset(os.path.join(DATA, cfg["train_valid_file_quelevel"]), input_type=cfg["input_type"],
                  folds=[0, 1, 2, 3, 4], concept_num=cfg["num_c"], max_concepts=cfg["max_concepts"])

# alignment assertion, identical to qikt_delta_mt (dataset[row_idx] block slice == npz block_q)
_it = ds[int(recs[0][0])]
_Q = torch.cat([_it["qseqs"][0:1], _it["shft_qseqs"]]).numpy()
_bs = int(recs[0][1])
assert np.array_equal(_Q[_bs:_bs + K], lab["block_q"][0]), "row_idx alignment broken!"
print("alignment check passed")


class DeltaHead(nn.Module):                 # IDENTICAL to qikt_delta_mt: [h_A, h_B, θ_before] -> Δθ
    def __init__(self, hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * hid + 1, hid), nn.ReLU(), nn.Linear(hid, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class TwoLSTM(nn.Module):
    """generic two-branch encoder: question-side and concept-side LSTM over plain embeddings."""
    def __init__(self, num_q, num_c, emb, dropout):
        super().__init__()
        self.q = nn.Embedding(num_q + 1, emb)   # last index = padding
        self.c = nn.Embedding(num_c + 1, emb)
        self.r = nn.Embedding(3, emb)           # 0,1 responses; 2 = padding
        self.drop = nn.Dropout(dropout)
        self.lstm_a = nn.LSTM(3 * emb, emb, batch_first=True)   # question-side: [q,c,r]
        self.lstm_b = nn.LSTM(2 * emb, emb, batch_first=True)   # concept-side:  [c,r]
        self.numq, self.numc = num_q, num_c

    def forward(self, Q, C, R):
        if C.dim() == 3:                        # multi-concept -> primary concept
            C = C[..., 0]
        Qi = torch.where(Q < 0, torch.full_like(Q, self.numq), Q)
        Ci = torch.where(C < 0, torch.full_like(C, self.numc), C)
        Ri = torch.where(R < 0, torch.full_like(R, 2), R)
        qe, ce, re = self.drop(self.q(Qi)), self.drop(self.c(Ci)), self.drop(self.r(Ri))
        ha, _ = self.lstm_a(torch.cat([qe, ce, re], -1))
        hb, _ = self.lstm_b(torch.cat([ce, re], -1))
        return ha, hb


class QueEmbEncoder(nn.Module):
    """QIKT's single-task encoder, dropped into THIS pipeline: QueEmb + KA LSTM over emb_qca + KS LSTM
    over the response-split concept embedding. Mirrors QIKTNet.forward. If this reproduces
    qikt_delta_mt single-task (~0.32 block), the pipeline is clean and any gap for --emb plain is the
    embedding; if this also comes out high, the gap is a pipeline bug, not the encoder."""
    def __init__(self, num_q, num_c, emb, dropout):
        super().__init__()
        self.que_emb = QueEmb(num_q=num_q, num_c=num_c, emb_size=emb, emb_type="iekt",
                              model_name="qikt", device=device)
        self.lstm_a = nn.LSTM(emb * 4, emb, batch_first=True)   # KA: over emb_qca
        self.lstm_b = nn.LSTM(emb * 2, emb, batch_first=True)   # KS: over response-split emb_c
        self.drop = nn.Dropout(dropout)
        self.emb = emb

    def forward(self, Q, C, R):
        if C.dim() == 2:                        # QueEmb.get_avg_skill_emb expects [B,L,max_c]
            C = C.unsqueeze(-1)
        Qi = torch.where(Q < 0, torch.zeros_like(Q), Q)         # pad->q0; only affects post-read positions
        Rf = torch.where(R < 0, torch.zeros_like(R), R).float()
        _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(Qi, C, Rf)
        ha = self.drop(self.lstm_a(emb_qca)[0])
        emb_ca = torch.cat([emb_c.mul((1 - Rf).unsqueeze(-1).repeat(1, 1, self.emb)),
                            emb_c.mul(Rf.unsqueeze(-1).repeat(1, 1, self.emb))], dim=-1)
        hb = self.drop(self.lstm_b(emb_ca)[0])
        return ha, hb


def pcorr(p, y, z):
    rpy, rpz, ryz = np.corrcoef(p, y)[0, 1], np.corrcoef(p, z)[0, 1], np.corrcoef(y, z)[0, 1]
    d = np.sqrt((1 - rpz ** 2) * (1 - ryz ** 2))
    return (rpy - rpz * ryz) / d if d > 1e-9 else 0.0


def seqs(data):
    # reconstruct the full (q,c,r) sequences the same way qikt_delta_mt reconstructs _Q.
    # C is kept as-is ([B,L] or multi-concept [B,L,max_c]); each encoder handles its own shape.
    Q = torch.cat([data["qseqs"][:, 0:1], data["shft_qseqs"]], 1).long()
    C = torch.cat([data["cseqs"][:, 0:1], data["shft_cseqs"]], 1).long()
    R = torch.cat([data["rseqs"][:, 0:1], data["shft_rseqs"]], 1).long()
    return Q.to(device), C.to(device), R.to(device)


def gather(HA, HB, row_idxs):                   # block-end h_A, h_B, θ_before, δ per window
    L = HA.shape[1]
    A, B, tbs, tgts = [], [], [], []
    for b, ridx in enumerate(row_idxs):
        for bs, tb, d in windows[ridx]:
            pos = (bs + K - 1) if a.read_at == "blockend" else (bs - 1)
            if pos < 0 or pos >= L:
                continue
            A.append(HA[b, pos]); B.append(HB[b, pos]); tbs.append(tb); tgts.append(d)
    A, B = torch.stack(A), torch.stack(B)
    return A, B, A.new_tensor(tbs), A.new_tensor(tgts)


def run(seed):
    # split logic IDENTICAL to qikt_delta_mt
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(rows_with_win)
    nv = len(perm) // 5
    val_idx = sorted(int(x) for x in perm[:nv])
    tr_idx = [int(x) for x in perm[nv:]]
    HID = a.emb_size
    enc = (QueEmbEncoder if a.emb == "queemb" else TwoLSTM)(NUM_Q, NUM_C, HID, a.dropout).to(device)
    dhead = DeltaHead(HID).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dhead.parameters()), lr=1e-3, weight_decay=1e-5)
    mse = nn.MSELoss()
    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=a.batch_size, shuffle=False)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=a.batch_size, shuffle=False)

    best_mse, best = 1e9, (0.0, 0.0)
    for ep in range(a.epochs):
        enc.train(); dhead.train(); ptr = 0
        for data in tr_loader:
            B = data["qseqs"].shape[0]; ridxs = tr_idx[ptr:ptr + B]; ptr += B
            opt.zero_grad()
            HA, HB = enc(*seqs(data))
            A, Bh, tb, y = gather(HA, HB, ridxs)
            loss = mse(dhead(torch.cat([A, Bh, tb.unsqueeze(1)], 1)), y)
            loss.backward(); opt.step()
        enc.eval(); dhead.eval(); ptr = 0
        P, T, Z = [], [], []
        with torch.no_grad():
            for data in val_loader:
                B = data["qseqs"].shape[0]; ridxs = val_idx[ptr:ptr + B]; ptr += B
                HA, HB = enc(*seqs(data))
                A, Bh, tb, y = gather(HA, HB, ridxs)
                dp = dhead(torch.cat([A, Bh, tb.unsqueeze(1)], 1))
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