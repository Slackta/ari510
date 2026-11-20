# ARI 510 Phase 2b – Competition Code (Ian Slackta)

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

## What Makes Title Intresting
Kaggle: https://www.kaggle.com/competitions/what-makes-title-intresting

Put `train.csv` and `test.csv` in the same folder as the script, then run it.

| Run ID | File | Approach | Settings | CV MAE |
| --- | --- | --- | --- | --- |
| run1 | `title_run1_tfidf_ridge.py` | char TF-IDF (2–5) on title + one-hot domain -> Ridge | alpha=3, min_df=2, seed 42 | 1.479 |
| run2 | `title_run2_embed_mlp.py` | all-mpnet-base-v2 title embeddings + one-hot domain -> PyTorch MLP (L1 loss, 3-seed ensemble) | CV search over (epochs, weight decay) x shrink k; selected 20 epochs, wd 0.1, k=1.0; hidden 32, dropout 0.3, lr 1e-3, seed 42 | 1.471 |

- Dev score = 5-fold CV (run1: 5 repeats, run2: 3 repeats) on the 120 single-rated training titles. Test labels are 6-rater means, so CV MAE is much higher than leaderboard MAE.
- run2 was run on Google Colab (T4 GPU); it picks its own settings by CV and prints the full search table.

## Customer Support Tone Classification
Kaggle: invite-only competition (Group 9). Labels 1-5: Appreciative, Neutral, Help-seeking, Frustrated, Escalatory.

Put `train.csv`, `validation.csv`, `test.csv` in the same folder as the script, then run it.

| Run ID | File | Approach | Settings | CV macro F1 |
| --- | --- | --- | --- | --- |
| run1 | `tone_run1_tfidf_lr.py` | char TF-IDF (2–5) on last_customer_message + separate char TF-IDF on conversation -> Logistic Regression | C=0.25, class_weight=balanced, min_df=2, seed 42 | 0.433 |
| run2 | `tone_run2_embed_mlp.py` | all-mpnet-base-v2 embeddings of last message + conversation (concatenated, 1536-d) -> PyTorch MLP | CV search over (epochs, wd); selected 80 epochs, wd 0.01; hidden 256, dropout 0.3, lr 1e-3, class-weighted CE, seed 42 | 0.383 |

- Dev score = repeated stratified 5-fold CV (5x3, seed 42) on train + validation merged (405 conversations).
- run2 was run on Google Colab (T4 GPU); it picks its own settings by CV and prints the search table.
