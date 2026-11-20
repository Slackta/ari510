# run2: all-mpnet-base-v2 title embeddings + one-hot domain -> PyTorch MLP regressor
#       trained with L1 loss (= the MAE metric), 3-model seed ensemble,
#       epochs / weight decay / shrink-toward-mean chosen automatically by CV
# What Makes Title Intresting - ARI 510 Phase 2b
# Colab: Runtime > Change runtime type > T4 GPU. Upload train.csv and test.csv when asked.
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

SEED = 42
ENCODER = "sentence-transformers/all-mpnet-base-v2"
HIDDEN, DROPOUT, LR, N_ENSEMBLE = 32, 0.3, 1e-3, 3
EPOCHS, WEIGHT_DECAY = 40, 1e-1  # overwritten by the search in step 4
# (epochs, weight decay) candidates. A fixed 40 / 1e-1 overfit on real embeddings (CV MAE 1.532, preds 1.47-4.76)
CONFIGS = [(40, 1e-1), (20, 1e-1), (15, 1.0), (8, 1.0)]
SHRINK = [0.0, 0.25, 0.5, 0.75, 1.0]  # final pred = mean + k * (MLP pred - mean)
DOMAINS = ["article", "podcast", "reddit", "youtube"]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---- 1. Load data ---------------------------------------------------------------------
if not os.path.exists("train.csv"):
    from google.colab import files
    print("Upload train.csv and test.csv"); files.upload()
train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")
y = train["interest"].to_numpy(np.float32)

# ---- 2. Features: 768-d sentence embedding of the title + 4-d one-hot source ----------
def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ENCODER, device="cuda" if torch.cuda.is_available() else "cpu")
    return m.encode(list(texts), batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

def onehot(df):
    return np.stack([(df["domain"] == d).to_numpy(np.float32) for d in DOMAINS], axis=1)

E = embed(pd.concat([train["title"], test["title"]]))
X = np.hstack([E[: len(train)], onehot(train)]).astype(np.float32)
X_test = np.hstack([E[len(train):], onehot(test)]).astype(np.float32)
print("Feature shape:", X.shape)

# ---- 3. MLP regressor. Predicts an offset from the training mean, so an untrained
#         network starts at "guess the average" and only moves away when the data says so.
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, HIDDEN), nn.ReLU(),
                                 nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_one(Xtr, ytr, seed):
    set_seed(seed)
    mu = float(ytr.mean())
    model = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss()  # optimise the competition metric directly
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr - mu)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(Xt), generator=g).split(16):
            opt.zero_grad(); loss_fn(model(Xt[idx]), yt[idx]).backward(); opt.step()
    model.eval()
    return model, mu

def fit_predict(Xtr, ytr, Xte):
    preds = []
    for k in range(N_ENSEMBLE):  # average 5 nets with different seeds -> smoother, less variance
        model, mu = train_one(Xtr, ytr, SEED + k)
        with torch.no_grad():
            preds.append(model(torch.tensor(Xte)).numpy() + mu)
    return np.clip(np.mean(preds, axis=0), 1, 6)

# ---- 4. Pick settings by CV: 5-fold x 3 repeats, out-of-fold predictions ---------------
#         For each (epochs, wd) config, also try shrinking predictions toward the fold's
#         training mean. Keep the combo with the lowest CV MAE.
results = []
for EPOCHS, WEIGHT_DECAY in CONFIGS:
    oofs, mus = [], []
    for r in range(3):
        oof, mu = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(X):
            oof[te] = fit_predict(X[tr], y[tr], X[te]); mu[te] = y[tr].mean()
        oofs.append(oof); mus.append(mu)
    for k in SHRINK:
        ps = [np.clip(m + k * (o - m), 1, 6) for o, m in zip(oofs, mus)]
        mae = np.mean([np.mean(np.abs(p - y)) for p in ps])
        rho = np.mean([spearmanr(o, y).correlation for o in oofs])
        results.append((mae, EPOCHS, WEIGHT_DECAY, k, rho))
        print(f"epochs={EPOCHS:3d} wd={WEIGHT_DECAY:<5} shrink k={k:.2f} -> CV MAE {mae:.3f} | Spearman {rho:+.3f}")

best_mae, EPOCHS, WEIGHT_DECAY, K, best_rho = min(results)
print(f"\nrun2 BEST: epochs={EPOCHS} wd={WEIGHT_DECAY} shrink k={K} | CV MAE: {best_mae:.3f} | Spearman: {best_rho:+.3f}")

# ---- 5. Train on all 120 titles with the best settings, predict test, save -------------
mu_all = float(y.mean())
pred = np.clip(mu_all + K * (fit_predict(X, y, X_test) - mu_all), 1, 6)
pd.DataFrame({"title_id": test["title_id"], "interest": np.round(pred, 3)}).to_csv("submission_run2.csv", index=False)
print("Saved submission_run2.csv | pred range", pred.min().round(2), "-", pred.max().round(2), "| mean", pred.mean().round(2))
try:
    from google.colab import files; files.download("submission_run2.csv")
except ImportError:
    pass
