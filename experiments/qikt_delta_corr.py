"""
Diagnostic: does a trained QIKT's own internals already encode our Δθ, or do they just track
ability? QIKT is trained ONLY on correctness (never sees Δθ), then frozen and read here. If its
internal gain (block-end minus block-start) already correlates with Δθ, reading QIKT off is "not
doing anything new" (Param's point). If instead its value tracks ability θ_before, supervising it
for Δθ is a genuine change. θ_before and Δθ are NEVER fed into QIKT; they are only the yardsticks
we correlate against.

Signals come from the RAW module states, read through QIKT's own trained readout, averaged over
ALL items (no next-item indexing, so no next-question contamination):
    per position t:
        ka[t] = mean over ALL questions of out_question_all(a_t)   (KA module, a_t = que_lstm state)
        ks[t] = mean over ALL concepts  of out_concept_all(g_t)    (KS module, g_t = concept_lstm state)
    per block window [bs, bs+K):
        KA-level  = mean of ka over the block          -> expected to track ability θ_before
        KA-change = ka[block end] - ka[block start]    -> expected to track gain Δθ
        KS-level, KS-change : same with ks

    old (removed) signals block-acquisition / head mastery-change indexed out_question_all /
    out_concept_all at the NEXT question, so they were contaminated by which the next item is.

Alignment: test_sequences_quelevel.csv rows all fold -1; KTQueDataset(folds=[-1]) keeps order, so
DataLoader(shuffle=False) index == row_idx in the Δθ npz. Output index j predicts original position
j+1, so block positions [bs, bs+K) map to output indices [bs-1, bs+K-1).

    python experiments/qikt_delta_corr.py --ckpt <saved_model/.../ dir> --dataset nips_task34 \
        --emb_size 256 --labels delta_qikt_nips_task34_question_test_W40K10.npz
"""
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/
PYKT = os.path.join(_ROOT, "pykt-toolkit")

from pykt.datasets.que_data_loader import KTQueDataset
from pykt.models.init_model import init_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="saved_model/<run>/ dir or the .ckpt file")
ap.add_argument("--dataset", default="assist2009")
ap.add_argument("--emb_size", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.4)
ap.add_argument("--mlp_layer_num", type=int, default=2)
ap.add_argument("--labels", default=os.path.join(_ROOT, "delta_qikt_test_W40K10.npz"))
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--W", type=int, default=40, help="pre/post window used when the Δθ labels were built")
a = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
DATA = os.path.join(PYKT, "data", a.dataset)
data_cfg = json.load(open(os.path.join(PYKT, "configs", "data_config.json")))[a.dataset]

# --- test dataset exactly like pykt (question level, folds=[-1]) ---
test_ds = KTQueDataset(os.path.join(DATA, data_cfg["test_file_quelevel"]),
                       input_type=data_cfg["input_type"], folds=[-1],
                       concept_num=data_cfg["num_c"], max_concepts=data_cfg["max_concepts"])
loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# --- rebuild model, load weights (init_model auto-places on cuda when available) ---
other = {"output_mode": "an_irt", "output_q_all_lambda": 1, "output_c_all_lambda": 1,
         "output_c_next_lambda": 1, "output_q_next_lambda": 0}
model_config = dict(emb_size=a.emb_size, dropout=a.dropout,
                    mlp_layer_num=a.mlp_layer_num, other_config=other)
model = init_model("qikt", model_config, data_cfg, "iekt")
ckpt = a.ckpt if a.ckpt.endswith(".ckpt") else os.path.join(a.ckpt, "iekt_model.ckpt")
model.load_state_dict(torch.load(ckpt, map_location=device))   # ckpt is the QueBaseModel wrapper (keys prefixed "model.")
model.model.eval()
print("loaded", ckpt)

# --- hook BOTH module LSTMs to grab the raw states a_t (KA) and g_t (KS) ---
_cap = {}
model.model.que_lstm_layer.register_forward_hook(     lambda m, i, o: _cap.__setitem__("a", o[0].detach()))
model.model.concept_lstm_layer.register_forward_hook( lambda m, i, o: _cap.__setitem__("g", o[0].detach()))

ka_rows, ks_rows = [], []
with torch.no_grad():
    for data in loader:
        model.predict_one_step(data, return_details=True)   # triggers forward, fills _cap
        # read raw states through QIKT's own trained readout, average over ALL items
        ka = model.model.out_question_all(_cap["a"]).mean(-1).cpu().numpy()   # [B, seqlen-1]
        ks = model.model.out_concept_all(_cap["g"]).mean(-1).cpu().numpy()    # [B, seqlen-1]
        for b in range(ka.shape[0]):
            ka_rows.append(ka[b]); ks_rows.append(ks[b])
print(f"rows extracted: {len(ka_rows)}, per-row output length: {len(ka_rows[0])}")

# --- align to Δθ windows ---
# Two readings, both compared on the SAME windows:
#   same-window (correct): read the QIKT signal on the exact windows the IRT labels use.
#       θ_before is fit on the pre-window W, so KA-level = mean ka over the pre-window.
#       Δθ = θ_after - θ_before spans pre->post, so KA-change = mean(ka over post) - mean(ka over pre).
#   block-internal (old, biased): read the signal only inside the block K. This measures a
#       different, offset span than the labels, so it is the wrong comparison; kept only to show
#       how much the misalignment moved the numbers.
lab = np.load(a.labels)
recs = lab["records"]        # cols: row_idx, block_start, theta_before, delta
K, W = a.K, a.W
# w_* = same-window (correct); b_* = block-internal (old)
w_kal, w_kac, w_ksl, w_ksc = [], [], [], []
b_kal, b_kac, b_ksl, b_ksc = [], [], [], []
tb, delta = [], []
skipped = 0
for row_idx, block_start, theta_before, d in recs:
    r, bs = int(row_idx), int(block_start)
    ka, ks = ka_rows[r], ks_rows[r]
    n = len(ka)
    lo, hi = bs - 1, bs + K - 1                      # block-start / block-end output indices
    pre_lo, pre_hi = bs - W - 1, bs - 1              # pre-window  output indices  [bs-W, bs)
    post_lo, post_hi = bs + K - 1, bs + K + W - 1    # post-window output indices  [bs+K, bs+K+W)
    if pre_lo < 0 or post_hi > n or hi >= n:
        skipped += 1; continue
    # same-window (correct)
    ka_pre, ka_post = ka[pre_lo:pre_hi].mean(), ka[post_lo:post_hi].mean()
    ks_pre, ks_post = ks[pre_lo:pre_hi].mean(), ks[post_lo:post_hi].mean()
    w_kal.append(float(ka_pre)); w_kac.append(float(ka_post - ka_pre))
    w_ksl.append(float(ks_pre)); w_ksc.append(float(ks_post - ks_pre))
    # block-internal (old)
    b_kal.append(float(ka[lo:hi].mean())); b_kac.append(float(ka[hi] - ka[lo]))
    b_ksl.append(float(ks[lo:hi].mean())); b_ksc.append(float(ks[hi] - ks[lo]))
    tb.append(theta_before); delta.append(d)

tb, delta = np.array(tb), np.array(delta)
print(f"aligned windows: {len(delta)}  (skipped {skipped})\n")

def row(name, x):
    x = np.array(x)
    p_d, s_d = pearsonr(x, delta)[0], spearmanr(x, delta)[0]
    p_t = pearsonr(x, tb)[0]
    print(f"{name:12s}  vs Δθ  pearson {p_d:+.3f} spearman {s_d:+.3f}   |   vs θ_before  pearson {p_t:+.3f}")

print("=== same-window (correct): level on pre-window, change = post - pre ===")
row("KA-level",  w_kal); row("KA-change", w_kac)
row("KS-level",  w_ksl); row("KS-change", w_ksc)
print("\n=== block-internal (old, biased): read only inside the block K ===")
row("KA-level",  b_kal); row("KA-change", b_kac)
row("KS-level",  b_ksl); row("KS-change", b_ksc)
print(f"\nbaseline: corr(θ_before, Δθ) pearson {pearsonr(tb, delta)[0]:+.3f}  (mechanical, checks alignment)")