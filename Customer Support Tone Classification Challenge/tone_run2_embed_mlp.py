# run2: all-mpnet-base-v2 embeddings of last_customer_message AND conversation (concatenated,
#       1536-d) -> PyTorch MLP, class-weighted cross-entropy. Epochs / weight decay chosen by CV.
# Customer Support Tone Classification - ARI 510 Phase 2b
# Colab: Runtime > Change runtime type > T4 GPU. Upload train.csv, validation.csv, test.csv.
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score

SEED = 42
ENCODER = "sentence-transformers/all-mpnet-base-v2"
HIDDEN, DROPOUT, LR = 256, 0.3, 1e-3
CONFIGS = [(10, 1e-4), (20, 1e-4), (40, 1e-4), (40, 1e-2), (80, 1e-2)]  # (epochs, weight decay) candidates
N_CLASSES = 5  # labels 1..5 are shifted to 0..4 inside the model

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---- 1. Load data -----------------------------------------------------------------------
if not os.path.exists("train.csv"):
    from google.colab import files
    print("Upload train.csv, validation.csv, test.csv"); files.upload()
train, val, test = pd.read_csv("train.csv"), pd.read_csv("validation.csv"), pd.read_csv("test.csv")
data = pd.concat([train, val], ignore_index=True)
y = data["label"].to_numpy() - 1
print("Conversations:", len(data), "| class counts (1-5):", np.bincount(y, minlength=5))

# ---- 2. Embed the last message and the whole conversation separately, then concatenate ----
def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ENCODER, device="cuda" if torch.cuda.is_available() else "cpu")
    return m.encode(list(texts), batch_size=32, show_progress_bar=True,
                    normalize_embeddings=True, convert_to_numpy=True)

both = pd.concat([data, test], ignore_index=True)
E_last = embed(both["last_customer_message"])
E_conv = embed(both["conversation"])  # long conversations are truncated by the encoder (384 tokens)
E = np.hstack([E_last, E_conv]).astype(np.float32)
X, X_test = E[: len(data)], E[len(data):]
print("Feature shape:", X.shape)

# ---- 3. The neural network we train --------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_in, HIDDEN), nn.ReLU(),
                                 nn.Dropout(DROPOUT), nn.Linear(HIDDEN, N_CLASSES))
    def forward(self, x):
        return self.net(x)

def train_mlp(Xtr, ytr, epochs, wd):
    set_seed()
    model = MLP(Xtr.shape[1])
    counts = np.bincount(ytr, minlength=N_CLASSES).astype(float)
    weights = torch.tensor(len(ytr) / (N_CLASSES * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)  # rare tones count more
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt), generator=g).split(16):
            opt.zero_grad(); loss_fn(model(Xt[idx]), yt[idx]).backward(); opt.step()
    return model

def predict(model, Xp):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xp)).argmax(1).numpy()

# ---- 4. Pick (epochs, weight decay) by 5-fold x 3 CV macro F1 ------------------------------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
results = []
for epochs, wd in CONFIGS:
    s = [f1_score(y[te], predict(train_mlp(X[tr], y[tr], epochs, wd), X[te]), average="macro")
         for tr, te in cv.split(X, y)]
    results.append((np.mean(s), np.std(s), epochs, wd))
    print(f"epochs={epochs:3d} wd={wd:<6} -> CV macro F1 {np.mean(s):.3f} (+/- {np.std(s):.3f})")

best_f1, best_sd, EPOCHS, WD = max(results)
print(f"\nrun2 BEST: epochs={EPOCHS} wd={WD} | CV macro F1: {best_f1:.3f} (+/- {best_sd:.3f})")

# ---- 5. Train on all 405 with the best settings, predict test, save -------------------------
pred = predict(train_mlp(X, y, EPOCHS, WD), X_test) + 1  # back to labels 1-5
pd.DataFrame({"id": test["id"], "label": pred}).to_csv("submission_run2.csv", index=False)
print("Saved submission_run2.csv | predicted counts (1-5):", np.bincount(pred, minlength=6)[1:])
try:
    from google.colab import files; files.download("submission_run2.csv")
except ImportError:
    pass
