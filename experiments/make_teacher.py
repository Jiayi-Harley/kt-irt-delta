"""
Precompute the ③-distillation TEACHER signals: KA-change and KS-change from the FROZEN section-17
QIKT (correctness-trained, emb256), per Δθ window, on the train_valid sequences.

These are the same-window signals from the diagnostic (qikt_delta_corr.py): for each window,
  ka[t] = mean over ALL questions of out_question_all(a_t)   (a_t = KA LSTM state, per position)
  ks[t] = mean over ALL concepts  of out_concept_all(g_t)
  KA-change = mean(ka over post-window) - mean(ka over pre-window)   (↔ Δθ +0.44, θ_before-free)
  KS-change = mean(ks over post-window) - mean(ks over pre-window)   (↔ Δθ +0.46)
The teacher "sees the future" (post-window); the student (qikt_delta_mt) does not. ③ distills these
into the student's block-end state (aux head, no θ_before), pushing its block signal 0.32->towards 0.44.

Saves teacher_<dataset>_<item>_<split>_W{W}K{K}.npz with array `teacher` [N, 2] = (KA-change, KS-change)
aligned row-for-row to the label npz `records`.

    python experiments/make_teacher.py --ckpt <saved_model/...nips...256_2> --dataset nips_task34 \
        --emb_size 256 --labels delta_qikt_nips_task34_question_train_valid_W40K10.npz --split train_valid
"""
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/
PYKT = os.path.join(_ROOT, "pykt-toolkit")
from pykt.datasets.que_data_loader import KTQueDataset
from pykt.models.init_model import init_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="frozen QIKT saved_model dir or .ckpt")
ap.add_argument("--dataset", default="nips_task34")
ap.add_argument("--emb_size", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.4)
ap.add_argument("--mlp_layer_num", type=int, default=2)
ap.add_argument("--labels", required=True)
ap.add_argument("--split", default="train_valid", choices=["train_valid", "test"])
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--W", type=int, default=40)
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
DATA = os.path.join(PYKT, "data", a.dataset)
cfg = json.load(open(os.path.join(PYKT, "configs", "data_config.json")))[a.dataset]
file_key = "train_valid_file_quelevel" if a.split == "train_valid" else "test_file_quelevel"
folds = [0, 1, 2, 3, 4] if a.split == "train_valid" else [-1]

ds = KTQueDataset(os.path.join(DATA, cfg[file_key]), input_type=cfg["input_type"], folds=folds,
                  concept_num=cfg["num_c"], max_concepts=cfg["max_concepts"])
loader = DataLoader(ds, batch_size=64, shuffle=False)   # order == row_idx (verified)

other = {"output_mode": "an_irt", "output_q_all_lambda": 1, "output_c_all_lambda": 1,
         "output_c_next_lambda": 1, "output_q_next_lambda": 0}
model = init_model("qikt", dict(emb_size=a.emb_size, dropout=a.dropout,
                                mlp_layer_num=a.mlp_layer_num, other_config=other), cfg, "iekt")
ckpt = a.ckpt if a.ckpt.endswith(".ckpt") else os.path.join(a.ckpt, "iekt_model.ckpt")
model.load_state_dict(torch.load(ckpt, map_location=device))
model.model.eval()
print("loaded", ckpt)

_cap = {}
model.model.que_lstm_layer.register_forward_hook(     lambda m, i, o: _cap.__setitem__("a", o[0].detach()))
model.model.concept_lstm_layer.register_forward_hook( lambda m, i, o: _cap.__setitem__("g", o[0].detach()))

ka_rows, ks_rows = [], []
with torch.no_grad():
    for data in loader:
        model.predict_one_step(data, return_details=True)
        ka = model.model.out_question_all(_cap["a"]).mean(-1).cpu().numpy()   # [B, seqlen-1]
        ks = model.model.out_concept_all(_cap["g"]).mean(-1).cpu().numpy()
        for b in range(ka.shape[0]):
            ka_rows.append(ka[b]); ks_rows.append(ks[b])
print(f"rows: {len(ka_rows)}, per-row length: {len(ka_rows[0])}")

lab = np.load(a.labels)
recs = lab["records"]        # row_idx, block_start, theta_before, delta
K, W = a.K, a.W
teacher = np.full((len(recs), 2), np.nan, dtype=np.float32)   # (KA-change, KS-change) per window
skipped = 0
for i, (row_idx, block_start, _, _) in enumerate(recs):
    r, bs = int(row_idx), int(block_start)
    ka, ks = ka_rows[r], ks_rows[r]
    n = len(ka)
    pre_lo, pre_hi = bs - W - 1, bs - 1               # pre-window output indices
    post_lo, post_hi = bs + K - 1, bs + K + W - 1     # post-window output indices
    if pre_lo < 0 or post_hi > n:                     # need the FULL pre-window (a start window's read is
        skipped += 1; continue                        # too noisy to be a good teacher); leave those as NaN
    teacher[i, 0] = ka[post_lo:post_hi].mean() - ka[pre_lo:pre_hi].mean()   # KA-change
    teacher[i, 1] = ks[post_lo:post_hi].mean() - ks[pre_lo:pre_hi].mean()   # KS-change

ok = ~np.isnan(teacher[:, 0])
d = lab["delta"]
print(f"windows: {len(recs)}, teacher computed: {ok.sum()}, skipped (window out of range): {skipped}")
from scipy.stats import pearsonr
print(f"sanity corr(KA-change, Δθ) = {pearsonr(teacher[ok,0], d[ok])[0]:+.3f}   "
      f"corr(KS-change, Δθ) = {pearsonr(teacher[ok,1], d[ok])[0]:+.3f}")

out = os.path.join(_ROOT, f"teacher_{a.dataset}_{a.split}_W{W}K{K}.npz")
np.savez(out, teacher=teacher, col_names=np.array(["KA_change", "KS_change"]))
print("saved", out)