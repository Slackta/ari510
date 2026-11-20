# run2: all-mpnet-base-v2 sentence embeddings (frozen) + PyTorch MLP (class-weighted)
# Retail Trader Intent Classification - ARI 510 Phase 2b
# Colab: Runtime > Change runtime type > T4 GPU. Put train.csv, validation.csv, test.csv
# in the same folder (or upload them when prompted).

import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score

SEED = 42
ENCODER = "sentence-transformers/all-mpnet-base-v2"
HIDDEN, DROPOUT, LR, WEIGHT_DECAY, EPOCHS = 256, 0.3, 1e-3, 1e-4, 60

LABELS = ["Buy / Increase", "Sell / Decrease / Short", "Hold", "No Clear Trade Intent"]  # ids 0..3
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---- 1. Load data -------------------------------------------------------------------
if not os.path.exists("train.csv"):
    from google.colab import files  # only runs in Colab
    print("Upload train.csv, validation.csv, test.csv")
    files.upload()

train, val, test = pd.read_csv("train.csv"), pd.read_csv("validation.csv"), pd.read_csv("test.csv")
data = pd.concat([train, val], ignore_index=True)
y = data["label"].map(LABEL_TO_ID).to_numpy()
print("Posts to learn from:", len(data), "| class counts:", np.bincount(y))

# ---- 2. Turn every post into a 768-number vector (pretrained, frozen) ----------------
def embed(texts):
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(ENCODER, device=device)
    return model.encode(list(texts), batch_size=32, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)

all_vecs = embed(pd.concat([data["text"], test["text"]]))
X, X_test = all_vecs[: len(data)], all_vecs[len(data):]
print("Embedding shape:", X.shape)

# ---- 3. The neural network we train ---------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, n_out=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_in, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, n_out),
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(Xtr, ytr):
    set_seed()
    model = MLP(Xtr.shape[1])
    # class weights: rare classes (Hold, Sell) count more in the loss
    counts = np.bincount(ytr, minlength=4).astype(float)
    weights = torch.tensor(len(ytr) / (4 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(Xt), generator=g).split(16):
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model

def predict(model, Xp):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xp, dtype=torch.float32)).argmax(1).numpy()

# ---- 4. Dev score: same 5-fold x 3 CV as run1 -----------------------------------------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scores = []
for tr_idx, te_idx in cv.split(X, y):
    m = train_mlp(X[tr_idx], y[tr_idx])
    scores.append(f1_score(y[te_idx], predict(m, X[te_idx]), average="macro"))
print(f"\nrun2 CV macro F1: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

# ---- 5. Train on everything, predict test, save submission ----------------------------
final = train_mlp(X, y)
sub = pd.DataFrame({"id": test["id"], "label": predict(final, X_test)})
sub.to_csv("submission_run2.csv", index=False)
print("Saved submission_run2.csv | predicted counts:", np.bincount(sub["label"], minlength=4))

try:
    from google.colab import files
    files.download("submission_run2.csv")
except ImportError:
    pass
