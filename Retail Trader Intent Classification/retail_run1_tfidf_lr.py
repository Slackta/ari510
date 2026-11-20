# run1: TF-IDF (char n-grams 2-5) + Logistic Regression (C=1, class-balanced)
# Retail Trader Intent Classification - ARI 510 Phase 2b (run1)
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier

SEED = 42

# 1. Load data (train + validation merged -> more data to learn from)
train = pd.read_csv("train.csv")
val = pd.read_csv("validation.csv")
test = pd.read_csv("test.csv")
data = pd.concat([train, val], ignore_index=True)
X, y = data["text"], data["label"]
print("Posts to learn from:", len(data))
print(y.value_counts(), "\n")

# 2. The model: turn text into word/character counts, then a simple classifier
def build_model():
    # character chunks (2-5 letters) handle slang/typos like "HODL", "loading up"
    features = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    clf = LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=SEED)
    return make_pipeline(features, clf)

# 3. Our own "dev score": 5-fold cross-validation, repeated 3 times
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
dummy = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=cv, scoring="f1_macro")
scores = cross_val_score(build_model(), X, y, cv=cv, scoring="f1_macro")
print(f"Always-guess-'No Clear Trade Intent' macro F1: {dummy.mean():.3f}")
print(f"run1 CV macro F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 4. Train on everything, predict the test posts, save the submission file
model = build_model().fit(X, y)
# Kaggle wants numbers, not label names
LABEL_TO_ID = {"Buy / Increase": 0, "Sell / Decrease / Short": 1, "Hold": 2, "No Clear Trade Intent": 3}
preds = model.predict(test["text"])
sub = pd.DataFrame({"id": test["id"], "label": [LABEL_TO_ID[p] for p in preds]})
sub.to_csv("submission_run1.csv", index=False)
print("\nSaved submission_run1.csv")
print(sub["label"].value_counts())
