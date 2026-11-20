# ARI 510 Phase 2b – Competition Code

## Retail Trader Intent Classification
Kaggle: https://www.kaggle.com/competitions/retail-trader-intent-classification

Put `train.csv`, `validation.csv`, `test.csv` (from the competition's Data tab) in the same folder as the script, then run it.

| Run ID | File | Approach | Settings | CV macro F1 |
| --- | --- | --- | --- | --- |
| run1 | `retail_run1_tfidf_lr.py` | TF-IDF char 2–5 grams + Logistic Regression | C=1, class_weight=balanced, min_df=2, seed 42 | 0.305 |
| run2 | `retail_run2_embed_mlp.py` | all-mpnet-base-v2 embeddings (frozen) + PyTorch MLP | hidden 256, dropout 0.3, AdamW lr 1e-3, wd 1e-4, batch 16, 60 epochs, class-weighted CE, seed 42 | 0.327 |

- Dev score = repeated stratified 5-fold CV (5×3, seed 42) on train + validation merged (205 posts).
- Each script prints its CV score and writes `submission_runN.csv` (labels 0–3).
- run1 runs on CPU (`pip install scikit-learn pandas`).
- run2 was run on Google Colab (T4 GPU). It needs `sentence-transformers` and `torch`, both preinstalled on Colab.
