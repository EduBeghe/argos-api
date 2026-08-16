# Railway Operations Runbook

## Services and schedules

Deploy `argos-api`, `argos-ingest`, `argos-decide`, and `argos-research` from this repository, PostgreSQL as the shared datastore, and `argos-web` from its separate repository — six services in total. `argos-api`, `argos-research`, `argos-web`, and PostgreSQL are continuously serving deployments; `argos-ingest` and `argos-decide` are Railway Cron Runs. Configure external cron schedules:

- ingest: every five minutes;
- decide: `2,32 * * * *` (UTC), allowing boundary candles and Scores to be persisted before the 30-minute cycle;
- research: **not** a cron. `argos-research` runs `python -m app.jobs.process_ml_refinement_queue` as a long-lived, continuously serving process (a durable queue worker), not a Cron Run. Deploy it as exactly **one replica** — PostgreSQL row locks make concurrent workers safe, but a second replica only doubles compute cost without doubling throughput, since only one worker can hold a given run's lease at a time.

The weekly Predictive Edge Discovery v4 cron remains intentionally pending until its
first-run timing and diagnostics have been validated on Railway. Do not infer a
profitable pattern from a successful job or enable a candidate automatically.

Weekly research logs distinguish `edge_guided_run_id` from
`predictive_edge_run_id`; predictive diagnostic and near-miss URLs always use the
predictive ID. Verify first run, exact reuse, and weekly range extension by comparing
current computed/reused work with original-run work and compute avoided. Historical
work must not be reported as newly computed.

### `argos-research` operational notes

`process_ml_refinement_queue` claims queued ML refinement runs, search-campaign children, and protected final-validation work through a database lease (`--poll-seconds`, 1–300; `--stale-after-seconds`, 300–86400; `--once` processes at most one available run and exits). A stale lease — a worker that died mid-claim — is marked failed and requires an explicit retry through the corresponding `POST .../retry` endpoint; it is never silently re-executed. Redeploying `argos-research` (e.g. after a code change) is safe: in-flight work resumes from its last committed evaluation on the next poll, and completed evaluations are never recomputed. Deploy the migration and API before starting or redeploying this worker, since it depends on the current schema.

Screening and refinement runs are CPU-bound (Ridge/LightGBM/CatBoost fits across up to seven folds per configuration); a full `exhaustive_ohlcv_v1` or `economic_policy_search_v2` campaign can occupy the worker for an extended period. Start with a one-duration canary (`python -m app.jobs.screen_ml_models --engine all --future-duration-seconds 1800`) or `--dry-run` before enqueueing a full campaign on a new instrument or after a registry/contract-version change.

Railway’s Cron Run is a one-shot execution created by a schedule or **Run now**. It is distinct from a continuously serving deployment. **Active** means the process is currently running; **Completed** means it exited successfully; **Failed** means a non-zero exit or platform failure. Open the run’s **Deploy Logs** for structured job output and traceback. Use **Run now** for one controlled retry. Disable a schedule before maintenance or expensive historical work and re-enable it only after verifying resources and configuration.

The API container runs `scripts/start_api.sh`: migrations must succeed before Uvicorn starts, and `${PORT:-8000}` is honored. Jobs create their own resources and exit. `railway ssh` opens a shell in the selected service environment; confirm the service before running commands and never paste secrets into logs.

## Environment

Required/shared values include `DATABASE_URL` (normalized to asyncpg), `BINANCE_BASE_URL`, Binance connect/read timeouts and ingestion limit, `CORS_ALLOWED_ORIGINS`, `LOG_LEVEL`, and optional service/version values. The API service additionally uses the authentication variables documented in [API authentication](015-authentication.md); jobs do not require them. Use `https://data-api.binance.vision` for public market data if the normal Binance endpoint returns HTTP 451 from the deployment location. Do not commit secrets or project UUIDs.

## Operational commands

Replace dates and instrument values deliberately. Run BTC and ETH sequentially on small services.

```sh
# Live jobs
python -m app.jobs.ingest_closed_candles
python -m app.jobs.generate_market_decisions

# Start a successor Paper run while preserving history
python -m app.jobs.reset_paper_account --initial-cash 10000 --reason "operator reset"

# Candle-only backfill; inspect first with --dry-run
python -m app.jobs.backfill_market_candles --from 2025-01-01T00:00:00Z --to 2025-06-01T00:00:00Z --asset BTC/USDT --provider binance --market-type spot --interval 5m --page-size 1000 --dry-run
python -m app.jobs.backfill_market_candles --from 2025-01-01T00:00:00Z --to 2025-06-01T00:00:00Z --asset BTC/USDT --provider binance --market-type spot --interval 5m --page-size 1000

# Validate reproducible dataset
python -m app.jobs.validate_historical_dataset --from 2025-01-01T00:00:00Z --to 2025-06-01T00:00:00Z --asset BTC/USDT --provider binance --market-type spot --interval 5m --page-size 1000

# Historical simulation
python -m app.jobs.simulate_historical_strategy --from 2025-01-01T00:00:00Z --to 2025-06-01T00:00:00Z --asset BTC/USDT --provider binance --market-type spot --interval 5m --initial-cash 10000 --fee-rate 0.001 --slippage-rate 0.0005

# Refinement with persistent memory and walk-forward evidence
python -m app.jobs.refine_historical_strategy --from 2025-01-01T00:00:00Z --to 2025-06-01T00:00:00Z --asset BTC/USDT --provider binance --market-type spot --interval 5m --initial-cash 10000 --fee-rate 0.001 --slippage-rate 0.0005 --persist --walk-forward

# Compact refinement storage diagnostics (no flags)
python -m app.jobs.refinement_storage

# Predictive Edge Map and persisted edge-guided qualification
python -m app.jobs.analyze_predictive_edges --asset BTC/USDT --provider binance --market-type spot --interval 5m --from 2025-01-01T00:00:00Z --to 2026-01-01T00:00:00Z --training-days 180 --validation-days 60 --test-days 60 --step-days 60 --maximum-windows 6 --bins 5 --persist --edge-guided-candidates --max-edge-guided-candidates 12 --initial-cash 10000

# ML pipeline (see docs/ml-experiment-contract.md, ml-model-screening.md, ml-search-campaign.md)
python -m app.jobs.validate_ml_experiment_dataset --asset BTC/USDT --contract-only
python -m app.jobs.screen_ml_models --asset BTC/USDT --engine all --future-duration-seconds 1800
python -m app.jobs.screen_ml_models --asset BTC/USDT --engine all --all-future-durations
python -m app.jobs.refine_ml_screening_result --config docs/examples/ml-refinement-catboost-30m.json --dry-run
python -m app.jobs.enqueue_ml_search_campaign --source-screening-run-id <run-id> --preset exhaustive_ohlcv_v1 --dry-run
python -m app.jobs.audit_ml_search_near_misses --campaign-id <campaign-id> --maximum-candidates 15 --output development-reaudit.json

# Durable ML worker (argos-research; run as one continuously serving replica, not a cron)
python -m app.jobs.process_ml_refinement_queue --poll-seconds 5 --stale-after-seconds 900
```

Use each command’s `--help` as the parser-authoritative flag reference. `refinement_storage` intentionally has no flags.

## Safety and diagnosis

- Confirm migrations with `python -m alembic upgrade head`; a migration failure prevents API startup.
- Check `/health` separately from `/health/database`.
- Inspect structured ingest totals for received/inserted/duplicates, context candles, target candles, Feature values, and Scores.
- Candle duplicates are expected during retries and do not imply failure.
- A downstream analytics failure must leave the earlier committed candle stage intact; rerun ingest to recover recent missing Features/Scores.
- Verify database size and available memory before large backfills or analyses. Historical analyses have taken roughly 30 minutes per asset in the observed environment; duration varies with data and resources.
- Do not run BTC and ETH historical analysis simultaneously on a small Railway service. Stop or stagger schedules to avoid competing with live jobs.
- Do not run candidate analysis with `--persist` unless the intended experiment identity and database capacity have been reviewed.
