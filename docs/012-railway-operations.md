# Railway Operations Runbook

## Services and schedules

Deploy `argos-api`, `argos-ingest`, and `argos-decide` from this repository, PostgreSQL as the shared datastore, and `argos-web` from its separate repository. Configure external cron schedules:

The weekly Predictive Edge Discovery v4 cron remains intentionally pending until its
first-run timing and diagnostics have been validated on Railway. Do not infer a
profitable pattern from a successful job or enable a candidate automatically.

- ingest: every five minutes;
- decide: `2,32 * * * *` (UTC), allowing boundary candles and Scores to be persisted before the 30-minute cycle.

Railway’s Cron Run is a one-shot execution created by a schedule or **Run now**. It is distinct from a continuously serving deployment. **Active** means the process is currently running; **Completed** means it exited successfully; **Failed** means a non-zero exit or platform failure. Open the run’s **Deploy Logs** for structured job output and traceback. Use **Run now** for one controlled retry. Disable a schedule before maintenance or expensive historical work and re-enable it only after verifying resources and configuration.

The API container runs `scripts/start_api.sh`: migrations must succeed before Uvicorn starts, and `${PORT:-8000}` is honored. Jobs create their own resources and exit. `railway ssh` opens a shell in the selected service environment; confirm the service before running commands and never paste secrets into logs.

## Environment

Required/shared values include `DATABASE_URL` (normalized to asyncpg), `BINANCE_BASE_URL`, Binance connect/read timeouts and ingestion limit, `CORS_ALLOWED_ORIGINS`, `LOG_LEVEL`, and optional service/version values. Use `https://data-api.binance.vision` for public market data if the normal Binance endpoint returns HTTP 451 from the deployment location. Do not commit secrets or project UUIDs.

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


## Weekly Strategy research

Create a separate `argos-research` service with start command
`python -m app.jobs.run_weekly_strategy_research` and suggested UTC cron
`17 3 * * 0`. It shares the backend database and logging variables, runs separately
from ingest and decide, and uses a PostgreSQL advisory lock. Operational interpretation,
retry behavior, resource bounds, and flags are documented in
[weekly Strategy research](014-weekly-strategy-research.md).
