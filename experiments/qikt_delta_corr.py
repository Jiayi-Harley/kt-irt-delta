"""
Diagnostic: does a trained QIKT's own outputs already correlate with our Δθ labels?
If yes, reading QIKT off is "not doing anything new" (Param's point) and novelty must come
from changing the input (his course-graph idea). If no, supervising QIKT for Δθ has real room.

Runs a trained QIKT on the SAME test question-level sequences the Δθ labels were built on,
extracts two per-position signals, aggregates each over the Δθ block windows, correlates.

Signals (both from QIKT's own heads, logit = pre-sigmoid IRT term):
  1. block acquisition : mean logit(y_question_all) over the block positions
                         = QIKT's "knowledge acquisition value" during the block
  2. mastery change    : logit(y_concept_all) just after the block minus just before it
                         = QIKT's own before/after ability change, the structural analog of Δθ

Alignment: test_sequences_quelevel.csv is 874 rows all fold -1; KTQueDataset(folds=[-1])
keeps them in order, so DataLoader(shuffle=False) index == row_idx in the Δθ npz.
Output index j predicts original position j+1, so block positions [bs, bs+K) map to
output indices [bs-1, bs+K-1).

    python experiments/qikt_delta_corr.py --ckpt <saved_model/.../ dir> --emb_size 64
"""
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/
PYKT = os.path.join(_ROOT, "pykt-toolkit")
DATA = os.path.join(PYKT, "data", "assist2009")

from pykt.datasets.que_data_loader import KTQueDataset
from pykt.models.init_model import init_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="saved_model/<run>/ dir or the .ckpt file")
ap.add_argument("--emb_size", type=int, default=64)
ap.add_argument("--dropout", type=float, default=0.4)
ap.add_argument("--mlp_layer_num", type=int, default=2)
ap.add_argument("--labels", default=os.path.join(_ROOT, "delta_qikt_test_W40K10.npz"))
ap.add_argument("--K", type=int, default=10)
a = ap.parse_args()

device = "cpu"
data_cfg = json.load(open(os.path.join(PYKT, "configs", "data_config.json")))["assist2009"]

# --- test dataset exactly like pykt (question level, folds=[-1]) ---
test_ds = KTQueDataset(os.path.join(DATA, data_cfg["test_file_quelevel"]),
                       input_type=data_cfg["input_type"], folds=[-1],
                       concept_num=data_cfg["num_c"], max_concepts=data_cfg["max_concepts"])
loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# --- rebuild model, load weights ---
other = {"output_mode": "an_irt", "output_q_all_lambda": 1, "output_c_all_lambda": 1,
         "output_c_next_lambda": 1, "output_q_next_lambda": 0}
model_config = dict(emb_size=a.emb_size, dropout=a.dropout,
                    mlp_layer_num=a.mlp_layer_num, other_config=other)
model = init_model("qikt", model_config, data_cfg, "iekt")
ckpt = a.ckpt if a.ckpt.endswith(".ckpt") else os.path.join(a.ckpt, "iekt_model.ckpt")
model.load_state_dict(torch.load(ckpt, map_location=device))   # ckpt is the QueBaseModel wrapper (keys prefixed "model.")
model.model.eval()
print("loaded", ckpt)

# --- forward, keep per-row per-position outputs (no flattening) ---
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

# hook the KS-module LSTM to grab the raw knowledge state g_t, then reduce it to a scalar
# "overall mastery" = mean over ALL concepts of out_concept_all(g_t) logits (no next-question indexing,
# so it drops the per-question difficulty noise the head outputs carry)
_cap = {}
model.model.concept_lstm_layer.register_forward_hook(lambda m, i, o: _cap.__setitem__("g", o[0].detach()))

qa_rows, ca_rows, ms_rows = [], [], []
with torch.no_grad():
    for data in loader:
        outputs, _ = model.predict_one_step(data, return_details=True)
        qa = outputs["y_question_all"].detach().cpu().numpy()   # [B, seqlen-1]
        ca = outputs["y_concept_all"].detach().cpu().numpy()
        ms = model.model.out_concept_all(_cap["g"]).mean(-1).detach().cpu().numpy()  # [B, seqlen-1] overall-mastery logit
        for b in range(qa.shape[0]):
            qa_rows.append(qa[b]); ca_rows.append(ca[b]); ms_rows.append(ms[b])
print(f"rows extracted: {len(qa_rows)}, per-row output length: {len(qa_rows[0])}")

# --- align to Δθ windows ---
lab = np.load(a.labels)
recs = lab["records"]        # cols: row_idx, block_start, theta_before, delta
K = a.K
sig1, sig2, sig3, tb, delta = [], [], [], [], []
skipped = 0
for row_idx, block_start, theta_before, d in recs:
    r, bs = int(row_idx), int(block_start)
    qa, ca, ms = qa_rows[r], ca_rows[r], ms_rows[r]
    lo, hi = bs - 1, bs + K - 1                 # output indices for the block
    if lo < 0 or hi >= len(qa):
        skipped += 1; continue
    sig1.append(np.mean(logit(qa[lo:hi])))                       # acquisition over block (head)
    sig2.append(logit(ca[bs + K - 1]) - logit(ca[bs - 1]))      # head mastery after - before
    sig3.append(ms[bs + K - 1] - ms[bs - 1])                    # STATE overall-mastery after - before
    tb.append(theta_before); delta.append(d)

sig1, sig2, sig3, tb, delta = map(np.array, (sig1, sig2, sig3, tb, delta))
print(f"aligned windows: {len(delta)}  (skipped {skipped})\n")

def report(name, x, y):
    pr = pearsonr(x, y)[0]; sr = spearmanr(x, y)[0]
    print(f"{name:38s}  pearson {pr:+.3f}   spearman {sr:+.3f}")

report("corr( block-acquisition , Δθ )", sig1, delta)
report("corr( head mastery-change , Δθ )", sig2, delta)
report("corr( STATE mastery-change , Δθ )", sig3, delta)
report("baseline: corr( theta_before , Δθ )", tb, delta)   # sanity, expect negative (regression to mean)

print("\n-- vs theta_before (ability): does the signal track level rather than gain? --")
report("corr( block-acquisition , theta_before )", sig1, tb)
report("corr( head mastery-change , theta_before )", sig2, tb)
report("corr( STATE mastery-change , theta_before )", sig3, tb)