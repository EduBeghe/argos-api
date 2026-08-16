# Current System Reference

This document describes the implementation on `master` after PR #79. It distinguishes operational behavior from planned work and should be updated with every material backend change. It was last verified against the deployed Railway environment (all six services) and the router/job source, not only against prose in other documents.

## Instruments and providers

An instrument is the complete identity `(asset, provider, market_type, interval)`. Display symbols are metadata, not identity. The catalog currently contains BTC/USDT and ETH/USDT on Binance Spot with five-minute candles. Both enable ingestion, scoring, decisions, and Paper Trading. The API catalog reports all four capabilities; internally Paper Trading currently derives eligibility from the decision-enabled projection rather than maintaining a separate flag. The code exposes capability-specific catalog resolution instead of symbol branches.

Binance is the only implemented provider. BTC and ETH are the current MVP configuration, not permanent architectural constants. SPY, stocks, ETFs, other providers, market calendars, and provider-specific jobs are future work. `/market/instruments` is the catalog source and is private once API authentication enforcement is enabled (it is enabled in production; see [API authentication](015-authentication.md)).

## Deployment topology

Argos runs as **six Railway services** from two repositories, not two one-shot jobs plus an API. All backend services build from this repository (`argos-api`); only their start command differs.

| Service | Start command | Schedule | Role |
| --- | --- | --- | --- |
| `argos-api` | `scripts/start_api.sh` (Uvicorn) | continuously serving | Read-only HTTP API for `argos-web`; applies Alembic migrations on boot; never contacts Binance, executes trades, or starts research jobs. |
| `argos-ingest` | `python -m app.jobs.ingest_closed_candles` | cron, every 5 minutes | Persists newly closed Binance Spot candles, then bounded live Feature and baseline Score calculation. Does not generate Decisions. |
| `argos-decide` | `python -m app.jobs.generate_market_decisions` | cron, `2,32 * * * *` UTC | Evaluates the latest 30-minute boundary for every configured instrument **and** every active ML Paper Deployment; creates Decision Cycles/Snapshots, executes Paper Trading, records equity snapshots. |
| `argos-research` | `python -m app.jobs.process_ml_refinement_queue` | continuously running (single replica) | Durable worker that claims and executes queued ML screening/refinement evaluations, ML search campaign children, and protected final-validation runs. This is the service that performs all CPU-heavy ML work; the API only enqueues and reads. |
| `argos-web` | static Angular build served by Caddy | continuously serving | Read-only SPA, separate repository, resolves instruments from the catalog. |
| `Postgres` | managed Railway Postgres | continuously serving | Shared transactional evidence store for every service above. |

`argos-research` did not exist when this document was last accurate; it is now the service that runs the largest share of total compute (ML screening, refinement, and search-campaign evaluation described below). See [Railway operations](012-railway-operations.md) for the full runbook, including `argos-research`'s operational constraints (single replica; stale-lease recovery).

## Market data and backfill

The live job requests Binance Spot `GET /api/v3/klines`, accepts only closed five-minute candles, normalizes the 12-position response, and persists exact Decimals. Natural uniqueness is asset, provider, market type, interval, and open time. Repeated responses are harmless duplicates.

`BINANCE_BASE_URL` is configurable. `https://api.binance.com` is the default; `https://data-api.binance.vision` is useful for public market-data access where the main endpoint returns HTTP 451 because of network-location restrictions. The URL changes transport only, not instrument identity or normalization.

Historical backfill is candle-only. It does not create historical Feature or Score rows. Existing measurements observed approximately 308–327 bytes of additional table-and-index storage per candle; this is an environment-specific estimate, not a database guarantee. Persisting raw normalized candles preserves reproducible evidence, while historical Feature matrices are constructed in bounded memory for analysis. Live Feature/Score work remains restricted to the recent ingestion target window.

The 1,000-candle live context limit covers the current EMA 20, EMA 50, RSI 14, ATR 14, return 1, return 12, and relative-volume 20 model with conservative recovery room. It prevents an historical backfill from making live runtime and memory proportional to the full table.

## Features and baseline Score

`baseline_features` version 1 calculates and persists seven same-candle technical measurements:

| Feature | Meaning |
| --- | --- |
| `ema_20` / `ema_50` | Fast and slow exponentially smoothed close levels |
| `rsi_14` | Wilder relative-strength balance over 14 close changes |
| `atr_14` | Wilder average true range, including gaps from the previous close |
| `return_1` / `return_12` | Fractional close return over one and twelve candles |
| `relative_volume_20` | Current quote volume divided by the previous 20-candle mean |

The exact formulas and alignment rules are in [Analytical Features](005-analytical-features.md). These seven Features are used only by the baseline heuristic Score below. The independent 126-variable ML registry (`app/ml/registry.py`) described in the ML section is a separate, much larger feature set computed only inside bounded historical/ML jobs — it is never persisted per candle and never feeds `baseline_technical`.

`baseline_technical` version 1 consumes all seven Features plus close price. It derives trend, RSI, one-candle return, twelve-candle return, volume activity, and volatility ratio. Directional weights are 0.55 trend, 0.1575 RSI, 0.09 return 1, and 0.2025 return 12; they sum exactly to 1. Volume confirmation has weight 0.25 and scales the directional score by a factor from 0.75 to 1. Persisted Decimals use scale 18 and round-half-even.

`overall_score = directional_score × volume_confirmation_factor`; `signal_strength = abs(overall_score)`. Interpretations range from strongly bearish through neutral to strongly bullish. They describe measured evidence, not an automatic instruction. The Strategy alone decides requested action. See [Baseline Score](006-baseline-score-model.md).

## Baseline Strategy and action vocabulary

`baseline_long_only` version 1, configuration version 1, consumes `baseline_technical` version 1. When no position exists, an overall Score of at least 0.60 requests `open_position` with a 15% target portfolio allocation. When a position exists, a Score at or above 0.20 produces `no_action`; below 0.20 it requests `sell_all` for 100% of that position.

The enum also reserves `increase_position` and `sell_partial`, but version 1 rejects and never emits them. "Open position" is intent at the eligible boundary, not an assertion that the price is a market peak or trough. The threshold uses several normalized signals and can occur during an established trend. A Decision Snapshot preserves requested sizing, rule, evidence identity, and explanation; a Trade is the separate simulated execution record.

Limitations include fixed thresholds and weights, long-only exposure, one position per instrument, one cadence, no order-book model, and no guarantee that historical behavior persists. This is the only Strategy currently receiving real capital-simulation traffic on the **primary** Paper Deployment; every ML Strategy runs on its own separate, non-primary Paper Deployment (see below).

## Paper Trading

Paper Trading uses versioned experiments ("runs"), not real wallets or exchange accounts. Version 1 defaults to 10,000 USDT, a 0.10% fee per execution notional, and 0.05% adverse slippage. Buys use observed price multiplied by `1 + slippage_rate`; sells use observed price multiplied by `1 - slippage_rate`. Slippage is a simplified unfavorable execution assumption, not an exchange quote or a second fee.

A run stores initial/current cash, configuration, status, timestamps, optional predecessor, Positions, immutable Trades, Decision Cycles, equity snapshots, and performance history. Valuation uses persisted compatible candles. Realized P&L comes from closed quantity; unrealized P&L marks open quantity; equity is cash plus marked position value. Performance includes return, P&L, drawdown, costs, activity, and snapshot boundaries.

Resetting through `app.jobs.reset_paper_account` stops the active run, starts a successor, links `predecessor_run_id`, and preserves all history. It does not synthesize sells or erase open-at-stop Positions. Changing initial funds therefore creates a new experiment rather than rewriting the old one.

### Paper Deployments (parallel wallets)

Contrary to earlier versions of this document, **parallel Strategy wallets are implemented.** `PaperDeploymentRecord` (`app/paper/deployment_models.py`) is the deployment lifecycle wrapper around a Paper Trading run:

- `lifecycle` is `active`, `paused`, or `archived`. Only a `paused` deployment may be archived; archiving stops its current run.
- `strategy_kind` is `baseline` (the heuristic above) or `ml_fixed_horizon_long_only` (a promoted ML Strategy).
- Exactly one non-archived deployment may be `is_primary` at a time; the dashboard and default paper-trading views read the primary deployment.
- Creating an ML deployment (`POST /paper/deployments`) requires a **completed, `passed`** protected final validation for the source Strategy specification, plus a trusted, hash-verified model artifact whose feature order matches the specification. There is no path to deploy an ML Strategy that has not cleared final validation.
- Transitions (`pause`, `resume`, `archive`, `make-primary`) are exposed at `POST /paper/deployments/{deployment_id}/{action}` and are guarded (e.g. the primary deployment cannot be archived directly; it must be replaced first).

`argos-decide` runs one decision cycle per active deployment per boundary: `run_instrument_cycle` for `baseline` deployments, `run_ml_instrument_cycle` for `ml_fixed_horizon_long_only` deployments. As of this writing, production has exactly one active deployment (the primary `baseline_long_only` wallet); no ML Strategy has yet cleared final validation to be promoted to its own deployment.

### Historical optimization

#### Reproducible dataset and simulation

A historical dataset is identified by complete instrument and half-open UTC range. Readers enforce ordering, candle continuity, closed-candle eligibility, and a deterministic fingerprint. Feature matrices are built from that dataset without persisting millions of historical Feature rows. Historical simulation reuses the production Score, Strategy, fees, slippage, capital, and cadence, and compares results with a buy-and-hold benchmark.

#### Refinement and experiment memory

Refinement separates training, validation, and isolated test evidence; explores bounded Score/Strategy parameter grids; evaluates controlled one-at-a-time ablations; and can persist compact runs, candidates, evaluations, benchmarks, and walk-forward evidence. `experiment_key` fingerprints all effective inputs. An identical completed graph can be reused, while incomplete or incompatible evidence is rejected or replaced atomically. Persistence is memory, not learning or activation: it avoids repeating identical experiments and lets later planning use version-compatible history.

This system (`app/optimization/memory.py`, `refinement.py`, `experiment_service.py`) is the shared base that the `walk_forward` search variant builds directly on top of. `composite_search`, `edge_guided`, `targeted_refinement`, and `predictive_edge` are separate, independently versioned search variants layered above it — see [Architecture](002-architecture.md#research-and-optimization-module) for how they relate and what is and is not shared between them.

#### Predictive Edge Map

The Predictive Edge Map examines versioned predictors across horizons, conditions, and walk-forward windows. Training freezes centers, mappings, volatility thresholds, and quantile bounds. Validation selects individual bin/action identities; isolated test confirms only those same frozen bins. Evidence may be promising, unstable, or not detectable. Temporal isolation, fingerprints, frozen mappings, and segment-specific observations limit leakage; none proves causality. See [Predictive Edge Map](009-predictive-edge-map.md) for the full v4/v4.1 contract, including regime-conditioned discovery.

#### Complete edge-guided Strategies and qualification v3

Generation is bounded, but a `long_entry` is an independent long-only entry rule rather than only a filter on `baseline_long_only`. Complete candidates freeze an entry, a timed/signal/baseline exit, cadence, holding bounds, cooldown, allocation, costs, and evidence. `exit_or_avoid` remains a separate defensive overlay.

Qualification contract v3 assigns:

- `rejected_validation` when training, validation, or the minimum valid-window evidence fails;
- `rejected_test` when training and validation pass but isolated test, confirmed-window count, or test stability fails;
- `confirmed_candidate` only with sufficient trades, strictly positive net returns in training/validation/test, strict outperformance of production in validation/test, acceptable drawdown, enough windows, and required stability.

Buy and hold remains an independent comparison and is not required to be beaten by this stage. Test metrics are null when validation does not authorize isolated test. Earlier evidence remains readable as legacy but is not eligible under v3. See [Weekly strategy research](014-weekly-strategy-research.md).

## Machine learning pipeline

This is the largest addition since the document's previous revision and the current center of gravity of the project. It is a distinct pipeline from the heuristic Score/Strategy above: it does not touch `baseline_technical`, `baseline_long_only`, live Features, or live Decisions unless and until an ML Strategy clears final validation and is explicitly promoted to its own Paper Deployment (see above).

### Dataset contract

`app/ml/contract.py` freezes the temporal split, half-open and UTC:

- development: `[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`;
- qualification: `[2024-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`;
- protected final test: `[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.

`app/ml/registry.py` is the authoritative 126-variable registry: price returns, EMA distance/slope/spread, breakout/range position, volatility, volume, candle shape, and time-of-day encodings, each computed at multiple systematic lookback scales (see [ML experiment dataset contract](ml-experiment-contract.md)). A fixed seven-day embargo gap separates training from validation, matching the longest supported target duration, so no training target can reach validation evidence. `eligible_indices()` is the only safe public split operation and cannot accidentally return final-test rows to a training consumer.

### Screening

`app/ml/screening/` fits Ridge, LightGBM, and CatBoost with a fixed, deliberately unoptimized version-1 configuration (seed `1729`) on the seven chronological development folds only, across all 80 registered future durations, per instrument. It reports MAE/RMSE, correlation, directional accuracy, and Pareto-set retention. See [ML model screening](ml-model-screening.md). This stage never reads the protected final-test period.

### Refinement

`app/ml/refinement/` starts from one persisted screening result and searches hyperparameters, feature policies (`all`, or top 16/32/64/96 variables by training-only relevance), prediction objectives (numeric regression, binary direction, cost-adjusted three-class), and confidence buckets, still restricted to development data. `argos-research` runs `python -m app.jobs.process_ml_refinement_queue` as a durable single-replica worker that claims queued refinement runs with a database lease.

### Search campaigns

`app/ml/campaign/` coordinates large-scale, durable search campaigns over the refinement queue:

- `exhaustive_ohlcv_v1` creates 3,600 immutable families per screening source (3 engines × 80 durations × 3 objectives × 5 feature policies), gives every family one evaluation, then deepens economically surviving families through successive budgets of 4, 16, and 64 configurations.
- `economic_policy_search_v2` (`app/ml/campaign/economic_v2_*.py`) adds 432 families per source (3 engines × 12 horizons × 3 objectives × 4 feature policies), each evaluated against 45 economic policies (5 entry-confidence fractions × 3 holding multipliers × 3 regime policies), plus a distinct **qualification stage** that reads a separate, still-protected `[2024-01-01, 2025-01-01)` window and writes to `ml_economic_qualifications` — never the reserved 2025–2026 evidence.

Promotion at every stage requires beating production and passing explicit economic gates, never predictive accuracy alone; see [ML search campaigns](ml-search-campaign.md) for the full gate contract, diagnostic re-qualification of stage-3 near-misses, and the read-only `audit_ml_search_near_misses` development re-audit tool.

### Protected final validation

`app/ml/protected/` is the only code path allowed to read `[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`. `app/ml/protected/temporal.py` reconstructs the complete development identity to prove no leakage before any protected read is permitted; `app/ml/protected/policy.py` (`failed_final_validation_rules`) gates promotion on: at least 20 round trips, total net return of at least 10%, all four half-year evidence blocks present, at least three of those four half-years positive, no half-year worse than −5%, maximum drawdown at or below 25%, and strictly positive return under both 1.5× and 2× cost stress. A model that clears every gate produces an `MLStrategySpecificationRecord` and a trusted, hash-verified `MLModelArtifactRecord`, which is the only artifact shape `POST /paper/deployments` will accept.

**As of this writing, no ML Strategy has cleared protected final validation.** The active research effort (`argos-research`, continuously running) is still in the screening/refinement/campaign stages for BTC/USDT and ETH/USDT.

## Current state and next steps

Implemented: live ingestion; bounded Features/Scores; Decisions; Paper execution; parallel Paper Deployments (baseline and ML) with pause/resume/archive/make-primary lifecycle; portfolio/performance and equity history; dashboard API; separate frontend; candle backfill; reproducible simulation; refinement persistence/reuse; Predictive Edge Discovery v4/v4.1 diagnostics and bounded funnel contracts; complete edge-guided Strategy research and qualification v3; weekly research execution; the full ML pipeline (dataset contract, screening, refinement, exhaustive and economic-policy search campaigns, protected final validation); and candidate/campaign/validation read APIs and UI (`argos-web` `/strategies`, `/ml-refinement`, `/ml-search-campaigns`, `/paper-strategies`).

Not implemented: additional providers/assets beyond BTC/USDT and ETH/USDT on Binance Spot; news/sentiment inputs; order-book or on-chain data; multi-user identity; and real-money trading. Single-administrator API authentication is implemented **and enforcement is enabled in production** (`ARGOS_AUTH_ENFORCEMENT_ENABLED=true`); see [API authentication](015-authentication.md). No ML Strategy has yet cleared protected final validation, so every ML search campaign's practical output to date is research evidence, not a deployed Strategy. These are possible next steps, not production commitments.
