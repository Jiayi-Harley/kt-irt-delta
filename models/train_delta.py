"""
Train a DKT-style model to predict Δθ (the project's core change).

vs the original DKT:
  - original: LSTM over interactions -> per-step sigmoid -> P(correct), cross-entropy loss
  - here:     LSTM over the block     -> last hidden + θ_before -> linear -> ONE Δθ, MSE loss

Baseline to beat: always predict the training-set mean Δθ.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0); np.random.seed(0)

# --- load samples ---
import sys
npz = sys.argv[1] if len(sys.argv) > 1 else "delta_labels_assist2015.npz"
print("labels:", npz)
d = np.load(npz)
block_c = torch.tensor(d["block_c"], dtype=torch.long)      # (N, K) item ids in block
block_r = torch.tensor(d["block_r"], dtype=torch.long)      # (N, K) responses 0/1
theta_b = torch.tensor(d["theta_before"], dtype=torch.float32)   # (N,)
delta   = torch.tensor(d["delta"], dtype=torch.float32)         # (N,)  the label
n_concepts = int(d["n_items"]); K = block_c.shape[1]
N = len(delta)

# interaction id = concept*2 + response  (same encoding idea as DKT)
inter = block_c * 2 + block_r                               # (N, K), range 0..2*n_concepts-1

# --- train/val split ---
idx = torch.randperm(N)
n_val = N // 5
val_i, tr_i = idx[:n_val], idx[n_val:]

def loader(ii, shuffle):
    return DataLoader(TensorDataset(inter[ii], theta_b[ii], delta[ii]),
                      batch_size=128, shuffle=shuffle)
tr_dl, val_dl = loader(tr_i, True), loader(val_i, False)

# --- model ---
class DeltaKT(nn.Module):
    def __init__(self, n_inter, emb=64, hid=64):
        super().__init__()
        self.emb = nn.Embedding(n_inter, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.head = nn.Linear(hid + 1, 1)          # +1 for θ_before; no sigmoid
    def forward(self, inter, theta_b):
        x = self.emb(inter)                        # (B, K, emb)
        _, (h, _) = self.lstm(x)                   # h: (1, B, hid)
        h = h.squeeze(0)                           # (B, hid)
        z = torch.cat([h, theta_b.unsqueeze(1)], dim=1)
        return self.head(z).squeeze(1)             # (B,) predicted Δθ

model = DeltaKT(2 * n_concepts)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)   # weight_decay = L2 reg
mse = nn.MSELoss()

# --- baseline: predict train mean ---
train_mean = delta[tr_i].mean()
base_val_mse = ((delta[val_i] - train_mean) ** 2).mean().item()

def eval_val():
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(xb, tb) for xb, tb, _ in val_dl])
        ytrue = torch.cat([yb for _, _, yb in val_dl])
    return mse(preds, ytrue).item(), np.corrcoef(preds.numpy(), ytrue.numpy())[0, 1]

# --- train with early stopping (keep the best val epoch) ---
best_mse, best_corr = 1e9, 0.0
for epoch in range(40):
    model.train()
    for xb, tb, yb in tr_dl:
        opt.zero_grad()
        mse(model(xb, tb), yb).backward(); opt.step()
    vm, vc = eval_val()
    if vm < best_mse:
        best_mse, best_corr = vm, vc

print(f"baseline (predict mean) val MSE: {base_val_mse:.3f}")
print(f"model (best epoch)      val MSE: {best_mse:.3f}   beats baseline: {best_mse < base_val_mse}")
print(f"corr(pred, label Δθ) on val:  {best_corr:+.3f}   <- how well the model fits the (windowed) labels; on real data there is no ground-truth Δθ")