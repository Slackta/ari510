# run1: char TF-IDF (2-5 grams) on last_customer_message AND on full conversation (separate
#       vectorizers) -> Logistic Regression (C=0.25, class-balanced)
# Customer Support Tone Classification - ARI 510 Phase 2b
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

SEED = 42
COLS = ["conversation", "last_customer_message"]

train, val, test = pd.read_csv("train.csv"), pd.read_csv("validation.csv"), pd.read_csv("test.csv")
data = pd.concat([train, val], ignore_index=True)  # 405 labeled conversations
X, y = data[COLS], data["label"]                     # labels are already 1-5
print("Class counts:", y.value_counts().sort_index().to_dict())

def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)

def build_model():
    features = ColumnTransformer([
        ("last", char_tfidf(), "last_customer_message"),  # the message being labeled
        ("conv", char_tfidf(), "conversation"),           # earlier turns = context
    ])
    clf = LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000, random_state=SEED)
    return make_pipeline(features, clf)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scores = cross_val_score(build_model(), X, y, cv=cv, scoring="f1_macro")
print(f"run1 CV macro F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

model = build_model().fit(X, y)
sub = pd.DataFrame({"id": test["id"], "label": model.predict(test[COLS])})
sub.to_csv("submission_run1.csv", index=False)
print("Saved submission_run1.csv |", sub["label"].value_counts().sort_index().to_dict())
