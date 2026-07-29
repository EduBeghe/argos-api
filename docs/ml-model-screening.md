# ML Model Screening

## Purpose

The model-screening job asks whether Ridge, LightGBM, or CatBoost can predict numeric future return reproducibly from the validated 126-feature ML dataset. A screening result is predictive evidence, not a Strategy or trading recommendation.

Training fits an engine on historical features and exact future returns. Ridge is a regularized linear baseline with fold-local standardization. It replaces logistic regression here because the contract predicts a number, not a class. LightGBM and CatBoost provide bounded nonlinear comparisons.

## Temporal contract

A future duration is the elapsed time between a decision boundary and the return being predicted. All 80 contract durations are supported. Only the seven chronological development folds are used. Preprocessing and fitting use training rows only; validation rows are later in time and rows are never shuffled. A guard rejects overlap with the protected 2024-2026 final test, preserving it for later candidate validation.

Eligibility protects the complete target interval as well as its starting boundary. A training row is used only when `boundary + future duration <= training_end`; validation applies the equivalent rule against `validation_end`. Target construction is restricted to those bounded development rows, so a long horizon near the end of 2023 cannot read a 2024 final-test candle. This target bound is additional to the existing seven-day embargo.

## Fixed version 1 configuration

All engines use deterministic seed `1729` and at most two threads by default. Ridge uses `alpha=1`. LightGBM uses 120 trees, depth 5, 24 leaves, and learning rate 0.05. CatBoost uses 120 iterations, depth 5, and learning rate 0.05. Tree engines use CPU execution and no early stopping. These conservative configurations compare engine families rather than optimize hyperparameters.

## Evidence

Each fold records MAE, RMSE, prediction and actual means, Pearson and Spearman correlation, directional accuracy, zero-prediction MAE, improvement over zero, sign counts, samples, and elapsed time. Undefined correlations remain null; non-finite values fail closed.

Aggregates preserve median and worst-fold predictive evidence. The Pareto set maximizes median and worst MAE improvement, median Spearman correlation, and folds beating zero. Training time only orders ties. Pareto retention is not approval for trading.

## Persistence, reuse, and operations

The experiment key contains contract and screening versions, dataset fingerprint, complete instrument identity, ordered engines and durations, exact configurations and library versions, and seed. Identical completed runs are reused; incomplete or failed runs are not. No executable model is serialized.

One-duration canary:

```bash
python -u -m app.jobs.screen_ml_models --asset BTC/USDT --engine all --future-duration-seconds 1800
```

All durations:

```bash
python -u -m app.jobs.screen_ml_models --asset BTC/USDT --engine all --all-future-durations
```

`--dry-run` validates dependencies, dataset identity, and planned work without fitting or writes. `--force-recompute` bypasses completed-run reuse. Training is sequential, targets are lazy by duration, fold predictions are released promptly, and `ML_SCREENING_THREAD_COUNT` defaults to two. Full screening is CPU-intensive, so Railway operation should begin with a canary.

Dry-run imports scikit-learn, LightGBM, and CatBoost themselves, which verifies their native modules can load rather than checking package metadata alone. Completion logs distinguish computed and reused executions. Peak-memory reporting prefers Linux `VmHWM`, which includes native engine allocations, and falls back to Python traced memory where `/proc` is unavailable.

Untouched final-test evaluation, candidate Strategy construction, after-cost economic testing, model-artifact design, and explicit activation remain future work. Live Decisions and Paper Trading are unchanged.
# Read-only screening evidence API

Persisted model-screening evidence is available through bounded, read-only optimization endpoints:

- `GET /optimization/ml-screening-runs`
- `GET /optimization/ml-screening-runs/latest?asset=BTC%2FUSDT&provider=binance&market_type=spot&interval=5m`
- `GET /optimization/ml-screening-runs/{run_id}`
- `GET /optimization/ml-screening-runs/{run_id}/results`
- `GET /optimization/ml-screening-runs/{run_id}/results/{result_id}`

Run and result collections support the documented filters plus `limit`/`offset` pagination, with a maximum page size of 200 and deterministic ordering. The result detail reads only the seven fold rows matching the result's run, engine, and future duration.

Evidence classification version 1 derives a label exclusively from the seven chronological development folds:

- **Promising** means every consistency, error-improvement, directional-accuracy, and positive-correlation gate passes.
- **Insufficient** means the complete result has partial combined positive evidence but misses one or more promising gates.
- **Discarded** means the complete result does not contain enough combined positive evidence under screening version 1. It is neither deleted nor declared permanently unusable.
- **Incomplete** means seven folds or required median correlations are missing.

Each result exposes the stable identifiers of failed promising rules and a concise explanation. Pareto retention remains a separate relative comparison: non-dominated does not mean predictive, approved, or suitable for trading.

These endpoints do not read the protected 2024-2026 final-test period, generate Strategies, serialize models, create wallets, or make trading recommendations.

