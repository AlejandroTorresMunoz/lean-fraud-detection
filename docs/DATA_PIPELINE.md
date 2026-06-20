# Data Pipeline — from raw transactions to model-ready sequences

How the **Sparkov** credit-card transactions become a clean, leakage-free, model-ready feature
table. The pipeline is a small **ETL** (Extract → Transform → Load): the orchestrator
[`build_sequences.py`](../src/lean_fraud/data/build_sequences.py) only loads and writes, and
delegates each Transform stage to a focused, unit-testable module under
[`data/transform/`](../src/lean_fraud/data/transform/).

```
Kaggle ──► download ──► [raw CSVs] ──► build_sequences (orchestrator) ──► sequences.npz + meta.json
 (Extract)                                   │
                                             ├─ transform/features.py   causal numeric features
                                             ├─ transform/split.py      strict time-based split
                                             └─ transform/encode.py     cat. codes + scaling (fit on train)

(train time) sequences.npz ──► windows.make_windows ──► (batch, seq_len, n_features) tensors
```

## Module map

| Module | Stage | Public API | Responsibility |
|---|---|---|---|
| [`data/download.py`](../src/lean_fraud/data/download.py) | Extract | `main()` | Pull the Sparkov CSVs via the Kaggle API into `data/raw/`. |
| [`data/build_sequences.py`](../src/lean_fraud/data/build_sequences.py) | Orchestrate | `main()` | Load + per-card ordering, compose the transforms, write outputs. |
| [`data/transform/features.py`](../src/lean_fraud/data/transform/features.py) | Transform | `treat_num_features(df, feats)` | Causal numeric feature engineering. |
| [`data/transform/split.py`](../src/lean_fraud/data/transform/split.py) | Transform | `time_split(t, test_size, val_size)` | Strict time-based train/val/test split. |
| [`data/transform/encode.py`](../src/lean_fraud/data/transform/encode.py) | Transform | `encode_categoricals`, `fit_scaler`, `apply_scaler` | Categorical → int codes; standardize numerics — **fit on train only**. |
| [`data/windows.py`](../src/lean_fraud/data/windows.py) | Train-time | `make_windows(x, user, seq_len, indices)` | Build causal per-user sequence windows lazily. |

## Dataset

**Sparkov** (`kartik2112/fraud-detection`) — ~1.85M synthetic credit-card transactions from **999
cards**, ~0.5% fraud. The card number (`cc_num`) is the per-user key; `unix_time` orders a card's
transactions. The two shipped CSVs (`fraudTrain.csv` + `fraudTest.csv`) are **merged**; we ignore
the dataset's own split and make our own (see below). Only a curated subset of columns is read; the
raw PII-ish fields are deliberately dropped.

## Stage by stage

### 1. Extract & order (`build_sequences.py`)
Load the CSVs, build the `user` key, and **sort by `(user, unix_time)`**. This ordering is a
precondition every downstream stage relies on (the groupby/rolling ops and the windowing assume it).

### 2. Transform — causal features (`transform/features.py`)
`treat_num_features` returns `(df, num_cols)` with these **causal** numeric features (each uses only
the current row and that card's *past* rows — never the future):

| Feature | Meaning |
|---|---|
| `amt`, `amt_log` | transaction amount and `log1p(amt)` |
| `dt` | seconds since the card's previous transaction |
| `amt_roll_mean` | mean spend of the card's **prior** transactions (`shift()` excludes the current one) |
| `amt_count` | running count of the card's prior transactions |
| `geo_dist` | cardholder ↔ merchant distance (a classic fraud signal) |
| `hour`, `dow` | the transaction's own hour-of-day / day-of-week |

The subtle bit: `amt_roll_mean` uses `.shift()` so the current transaction never contributes to its
own rolling mean — otherwise the label-bearing row would leak into its own feature.

### 3. Transform — strict time split (`transform/split.py`)
`time_split` assigns each transaction a label (`0=train, 1=val, 2=test`) **by its own timestamp**:
the earliest 70% of transactions → train, next 10% → val, most recent 20% → test. This mirrors
production (train on the past, score the future) and guarantees
`t[train].max() < t[val].min() < t[test].min()` — **no future leaks into train**.

### 4. Transform — encode & scale, *fit on train only* (`transform/encode.py`)
The leakage-critical stage, split into explicit fit/apply:
- `encode_categoricals` maps `category`, `gender`, `state` to integer codes. The map is built from
  values **seen in train**; categories appearing only in val/test map to `0` (unknown).
- `fit_scaler` computes mean/std of the numeric block over **train rows only**; `apply_scaler`
  standardizes all rows with those train statistics. Categorical codes are **not** scaled.

### 5. Load — one table, not three (`build_sequences.py`)
The output is a single time-sorted table, `data/processed/sequences.npz`:

| Array | Type | Shape | Meaning |
|---|---|---|---|
| `X` | float32 | `(n, n_features)` | engineered + scaled features |
| `y` | int8 | `(n,)` | `is_fraud` label |
| `user` | int64 | `(n,)` | contiguous card id (for grouping) |
| `t` | int64 | `(n,)` | `unix_time` (ordering within a card) |
| `split` | int8 | `(n,)` | 0=train, 1=val, 2=test |

plus `meta.json` (feature names + order, scaler stats, category maps, per-split row counts and fraud
rates). We keep **one** table on purpose: a val/test row's causal window may legitimately include
that card's earlier *train* rows — that is past context, **not** leakage. Splitting into three tables
would amputate that history at the time boundary.

### 6. Window building (`data/windows.py`, train time)
`make_windows` expands each target row into the `seq_len` (default 32) transactions ending at it
within the same card, **left-padded with zeros**. It is built **lazily per batch** from the 2-D
table, avoiding a multi-GB `(n, seq_len, n_features)` materialization. Not called by
`build_sequences` — it belongs to the training/scoring path.

## Anti-leakage invariants (what a reviewer should check)
1. **Causal features** — every feature uses only current/past rows of the same card (`shift()` on
   rolling spend; per-card `diff()` for `dt`).
2. **Time-ordered split** — assignment by timestamp; `t[train].max() < t[val].min() < t[test].min()`.
3. **Train-only fitting** — category maps and scaler statistics come from the train split alone.
4. **Causal windows** — each window ends at the target row; padding is on the left (oldest side).

These are the targets of the `test_features` / `test_split` / `test_windows` unit tests.

## How to run
```bash
uv sync --group data                              # Kaggle client for the download
uv run python -m lean_fraud.data.download         # -> data/raw/*.csv (idempotent)
uv run python -m lean_fraud.data.build_sequences  # -> data/processed/sequences.npz + meta.json
```

## Exploratory analysis
The design choices above are justified empirically in
[`notebooks/eda_sparkov.ipynb`](../notebooks/eda_sparkov.ipynb): class imbalance (→ PR-AUC + focal
loss), transactions-per-card (→ `sequence_length`), per-signal fraud patterns, and a **leakage sanity
check** (no single engineered feature scores a near-perfect single-feature ROC-AUC on train). Run it
after the pipeline with `uv sync --group eda` then open it in Jupyter.

## Imbalanced-class toolkit: PR-AUC and focal loss

With fraud at ~0.5% of transactions, a model and a metric that treat every transaction equally get
dominated by the legit majority. Two standard choices address this.

**PR-AUC — the headline metric (not ROC-AUC).**
- *ROC-AUC* is the area under the curve of true-positive rate vs **false-positive rate**. Its weakness
  on rare positives: the false-positive rate divides by the huge number of negatives, so thousands of
  false alarms barely move it — ROC-AUC can read 0.95+ while the system is unusable in practice.
- *PR-AUC* is the area under the **precision–recall** curve. **Precision = TP / (TP + FP)** directly
  penalizes false positives relative to the few true frauds; **recall = TP / (TP + FN)** measures how
  much fraud is caught. A trivial classifier's PR-AUC is ≈ the fraud base rate (~0.005), not 0.5, so
  PR-AUC is an honest summary of minority-class performance. **It is the number we report.**

**Focal loss — the training objective.**
- Plain cross-entropy averages the loss over all examples; with 99.5% easy negatives, the gradient is
  swamped by transactions the model already gets right, so it barely learns the rare fraud.
- *Class weighting* (and XGBoost's `scale_pos_weight`) scales up the positive class so the minority
  gets proportional attention.
- *Focal loss* (Lin et al., 2017) goes further: it multiplies each example's loss by a factor
  **(1 − p_t)^γ** (where `p_t` is the predicted probability of the correct class). That factor shrinks
  toward zero for confident, well-classified examples and stays ~1 for hard, misclassified ones, so the
  model spends its capacity on the **hard cases** (the fraud and borderline legit) instead of the sea
  of easy negatives. `γ` (gamma) sets how aggressively easy examples are down-weighted; an optional `α`
  term adds class weighting on top.

> **What is the per-feature "AUC" in the EDA?** A *single-feature ROC-AUC*: each feature's value is used
> directly as the fraud score and scored against the label on the train split (we take `max(auc, 1−auc)`
> so a feature predictive in either direction counts). It answers "how well does this one feature alone
> rank fraud above legit?" — **0.5 = random, 1.0 = perfect**. Two uses: ranking feature strength, and a
> **leakage alarm** — a lone feature near 1.0 would mean it encodes the label. ROC-AUC is fine for this
> quick, symmetric separability screen; PR-AUC stays reserved for evaluating the trained *model*.

## Next phases
The stage unit tests (`test_features` / `test_split` / `test_windows`) and the EDA notebook are done.
Remaining:
1. **MLflow dataset tracking** — log dataset version/hash, per-split row counts & fraud rates, and
   store `meta.json` (scaler + feature order) as a run artifact, so each model is reproducibly tied
   to the exact data and preprocessing that produced it.
2. **Modelling** — a `SequenceDataset` over `make_windows`, then `train`/`evaluate`/`benchmark`
   (TCN vs Transformer vs baselines) reporting quality **and** efficiency (params, p50/p99 latency).
3. **Phase 2 (see [ARCHITECTURE.md](ARCHITECTURE.md))** — SQS + Postgres + Airflow DAGs for the batch
   inference pipeline; serving loads the `Production` model from the MLflow registry.
