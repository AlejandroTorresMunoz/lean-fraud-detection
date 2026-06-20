# Data Exploration — Findings

Conclusions from the exploratory analysis in
[`notebooks/eda_sparkov.ipynb`](../notebooks/eda_sparkov.ipynb), on the **Sparkov** dataset
(~1.85M transactions, 999 cards). Every number below is reproduced by that notebook; this doc is the
written summary and the bridge from *what the data says* to *the modelling decisions*.

## 1. Dataset at a glance
- **1,852,394** transactions across **999** cards; **overall fraud rate 0.52%**.
- Strict **time-based** split (no future in train), with fraud rate **declining** over time:

  | Split | Rows | Fraud rate |
  |---|--:|--:|
  | train | 1,296,677 | 0.579% |
  | val | 185,239 | 0.430% |
  | test | 370,478 | 0.364% |

## 2. Severe class imbalance → metric & loss
Fraud is **~0.5%** of transactions. Consequences that drive the modelling choices:
- **Accuracy is useless** (predicting "never fraud" scores ~99.5%).
- **Report PR-AUC, not ROC-AUC** — with so many negatives, the false-positive rate barely moves, so
  ROC-AUC looks optimistic; PR-AUC reflects real minority-class performance.
- **Train with focal / class-weighted loss** (and XGBoost `scale_pos_weight`) so the gradient isn't
  swamped by the easy majority of legit transactions.

## 3. Temporal drift → the test set is genuinely harder
The monthly fraud rate roughly **halves over the ~2-year span** (≈1.0% in early 2012 → ≈0.2% by end
2013), which is why the time-ordered splits show a falling fraud rate (0.58% → 0.36%). This is real
**distribution shift**: the model is evaluated on the most recent, lower-fraud period, so reported
metrics are honest rather than flattered by an easier random split.

## 4. Per-card history → `sequence_length = 32`
Transactions per card (n = 999): **median 1,471**, mean 1,854, IQR 740–2,917, min 6, max 4,392.
- **90.9%** of cards have ≥ 32 transactions, and **~100%** of *transactions* belong to a card with
  ≥ 32 transactions — so a 32-step window is almost always **full**.
- `sequence_length = 32` is therefore a *recent-history* slice that captures behavior while keeping the
  TCN small and low-latency — the project's "efficiency beats scale" thesis.

## 5. Feature signal & leakage check
Single-feature **ROC-AUC on the train split** (direction-agnostic, `max(auc, 1−auc)`):

| Feature | AUC | | Feature | AUC |
|---|--:|---|---|--:|
| amt / amt_log | 0.835 | | category_code | 0.575 |
| dt | 0.652 | | dow | 0.536 |
| amt_count | 0.646 | | gender_code | 0.525 |
| amt_roll_mean | 0.600 | | state_code | 0.506 |
| hour | 0.585 | | geo_dist | 0.501 |

- **Amount is the strongest single signal** (median legit **$47** vs fraud **$390**, ~8×).
- **Leakage check passes:** the highest single-feature AUC is **0.835 < 0.95** — no feature is a label
  proxy, confirming the pipeline's anti-leakage design.
- **`geo_dist` alone is ≈ random (0.50)** in Sparkov — a candidate to confirm/prune in the post-model
  feature-importance step.
- **Redundancy:** `amt`/`amt_log` (and `amt_count`/`amt_roll_mean`) are highly correlated (see the
  correlation heatmap) — also pruning candidates for the lean model.

## 6. Implications for modelling
1. Headline metric = **PR-AUC**; also report precision/recall/F1. Keep ROC-AUC secondary.
2. Loss = **focal / weighted**; XGBoost `scale_pos_weight ≈ negatives/positives`.
3. Keep **`sequence_length = 32`**; windows are effectively always full.
4. Evaluate on the **time-based test split** (don't switch to a random split — it would hide the drift).
5. After the first model, run **feature importance** (SHAP / permutation) to confirm whether
   `geo_dist` and the redundant pairs can be dropped without hurting PR-AUC.

> See [DATA_PIPELINE.md](DATA_PIPELINE.md) for how the features and splits are built, and the notebook
> for the charts behind each point.
