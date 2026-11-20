# run1: char TF-IDF (2-5 grams) on title + one-hot source domain -> Ridge regression (alpha=3)
# What Makes Title Intresting - ARI 510 Phase 2b
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

SEED, ALPHA = 42, 3.0
train, test = pd.read_csv("train.csv"), pd.read_csv("test.csv")
y = train["interest"].to_numpy(float)

def fit_predict(tr_df, ytr, te_df):
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    onehot = OneHotEncoder(handle_unknown="ignore")
    Xtr = hstack([tfidf.fit_transform(tr_df["title"]), onehot.fit_transform(tr_df[["domain"]])])
    Xte = hstack([tfidf.transform(te_df["title"]), onehot.transform(te_df[["domain"]])])
    pred = Ridge(alpha=ALPHA, random_state=SEED).fit(Xtr, ytr).predict(Xte)
    return np.clip(pred, 1, 6)  # scores must stay on the 1-6 scale

# Dev score: 5-fold CV repeated 5 times, out-of-fold MAE + Spearman
maes, rhos = [], []
for r in range(5):
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(train):
        oof[te] = fit_predict(train.iloc[tr], y[tr], train.iloc[te])
    maes.append(np.mean(np.abs(oof - y))); rhos.append(spearmanr(oof, y).correlation)
base = np.mean([np.mean(np.abs(y.mean() - y))])
print(f"Predict-the-mean MAE (in-sample reference): {base:.3f}")
print(f"run1 CV MAE: {np.mean(maes):.3f} | Spearman: {np.mean(rhos):+.3f}")

pred = fit_predict(train, y, test)
pd.DataFrame({"title_id": test["title_id"], "interest": np.round(pred, 3)}).to_csv("submission_run1.csv", index=False)
print("Saved submission_run1.csv | pred range", pred.min().round(2), "-", pred.max().round(2), "| mean", pred.mean().round(2))
