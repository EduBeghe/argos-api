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

## Parameterized refinement

Screening compares one fixed configuration for every engine and horizon. Refinement starts from one persisted screening result and searches a normalized JSON-defined set of hyperparameters, feature policies, prediction objectives, confidence buckets, costs, and budgets. The example at `docs/examples/ml-refinement-catboost-30m.json` is deliberately bounded and contains placeholder UUIDs rather than production identifiers.

The supported objectives are numeric future-return regression, binary future-direction classification, and cost-adjusted `up`/`no_trade`/`down` classification. Binary positive-class probabilities are converted to the signed directional score `2p - 1`: negative values predict down, positive values predict up, and magnitude represents confidence. Cost-adjusted labels use the configured round-trip fee and slippage assumptions; they are experimental predictions, not Paper Trading instructions.

`all` uses the complete versioned variable registry. `explicit` validates names against that registry. `top_k` calculates relevance independently inside each fold using training rows only and persists the selected names with that fold. Confidence thresholds are also frozen from training predictions before validation metrics are calculated. Positive and negative signal evidence is persisted separately. Economic evidence is aggregated for the same confidence bucket across folds; classification uses its total observations, majority consistency, median, worst, weighted net return, and coverage rather than mixing maxima from unrelated folds or buckets. These rules prevent validation information from selecting variables or confidence thresholds.

Refinement uses two deliberately separate dataset roles. First it rebuilds canonical identity metadata over `2020-01-01T00:00:00Z` through `2026-01-01T00:00:00Z` (exclusive), exactly matching the persisted screening fingerprint. That full dataset is released after retaining only immutable metadata; its features, targets, and final-test indices never reach candidate evaluation. The worker then builds a distinct development evaluation dataset ending at `2024-01-01T00:00:00Z` (exclusive). Only this second dataset supplies the seven chronological folds, feature selection, fitting, targets, metrics, confidence evidence, and classification. Matching identity fingerprints authorizes the experiment identity, not access to protected final-test evidence. Live candles at or after the exclusive 2026 identity end cannot change the fingerprint.

Validate and inspect the deterministic plan without training or writes:

```bash
python -m app.jobs.refine_ml_screening_result --config docs/examples/ml-refinement-catboost-30m.json --dry-run
```

Execute and persist the same plan:

```bash
python -m app.jobs.refine_ml_screening_result --config docs/examples/ml-refinement-catboost-30m.json
```

The normalized configuration, dataset fingerprint, source result, costs, contract versions, library versions, and seed form the experiment key. Before work begins, the worker requires the source screening, queued refinement, and canonically rebuilt fingerprints to agree. A mismatch identifies the conflicting pair and records the instrument, closed range, candle count, and contract versions. Live ingestion beyond the exclusive end is harmless, but any insertion, update, or deletion inside the pinned range changes the rebuilt identity and requires a new screening run. Evaluation reuse additionally freezes the variable registry, missing-data policy, classification contract, relevant engine-library version, and deterministic seed. An identical completed experiment is returned without recomputation. Compatible completed individual evaluations are copied as reused evidence. A failed run retains its transactionally complete evaluations; explicit retry keeps the same run ID and experiment identity, reuses compatible evidence, and calculates only missing combinations. Trial and runtime budgets stop creation of new evaluations only between transactional writes and record computed, reused, and omitted counts plus the termination reason. Dry-run output reports any engine, objective, duration, or feature policy left uncovered by a budget too small for the requested dimensions.

Run states are `queued`, `running`, `completed`, `failed`, and `cancelled`. Evaluation classifications are `promising_for_strategy_simulation`, `insufficient_evidence`, `discarded`, and `incomplete`. Promising evidence only permits later economic simulation: it is not a Strategy approval and still requires protected final-test evaluation in a separate, explicitly governed workflow.

The private API exposes the authoritative contract and durable queue controls:

- `GET /optimization/ml-refinement-contract`
- `POST /optimization/ml-refinement-runs`
- `POST /optimization/ml-refinement-runs/{run_id}/cancel`
- `POST /optimization/ml-refinement-runs/{run_id}/retry`
- `GET /optimization/ml-refinement-runs`
- `GET /optimization/ml-refinement-runs/{run_id}`
- `GET /optimization/ml-refinement-runs/{run_id}/evaluations`
- `GET /optimization/ml-refinement-runs/{run_id}/evaluations/{evaluation_id}`

Enqueue validates the source and deterministic plan but never loads candles or trains a model. Identical submissions converge on one run and report whether it was queued, already active, requeued, or reused. Cancellation is immediate before claim and cooperative between evaluations after claim. Failed and cancelled runs can be explicitly retried while retaining compatible evaluation evidence.

Run the durable Railway `argos-research` worker as one replica initially:

```bash
python -m app.jobs.process_ml_refinement_queue
```

`--poll-seconds` accepts 1 through 300, `--stale-after-seconds` accepts 300 through 86400, and `--once` processes at most one available run. PostgreSQL row locks make multiple workers safe, but one replica controls research compute cost. Deploy the migration and API first, then start or redeploy this worker. A stale lease is marked failed and requires explicit retry; it is never silently executed again.
