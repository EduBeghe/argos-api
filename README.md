# Argos API

Argos is a private, asset-agnostic quantitative research and predictive decision-support platform. It estimates probabilities from measurable evidence; it does not claim to predict the future with certainty or guarantee profitability.

The platform will ingest market observations, calculate explicit and explainable component metrics, generate simulated decisions, execute paper trades, and evaluate results after fees and simulated slippage. Potential analytical components include technical analysis, momentum, volatility, market context, news, sentiment, and machine-learning outputs. These independent scores may later contribute to an aggregated Argos Score.

Every generated signal must retain its component scores, supporting metrics, and reasoning. The initial MVP will study BTC/USDT and ETH/USDT on Binance Spot, while keeping the domain model suitable for other providers, instruments, and asset classes.

## Documentation

- [Project vision](docs/000-project-vision.md)
- [MVP scope](docs/001-mvp-scope.md)
- [Architecture overview](docs/002-architecture.md)
- [Domain model](docs/003-domain-model.md)
- [Binance market-data contract](docs/004-binance-market-data-contract.md)
- [Architecture decisions](docs/decisions/)

## Current phase

The project is in its documentation-first phase. No application implementation, live-money trading, or profitability claim is part of this phase.

## Development

### Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker with Docker Compose, for the containerized environment

### Installation

Copy `.env.example` to `.env`, replace the example credentials, and install the locked dependencies:

```sh
cp .env.example .env
uv sync
```

### Local execution

```sh
make run
```

The API is available at `http://localhost:8000`. Process health is reported by `GET /health`; PostgreSQL connectivity is reported independently by `GET /health/database`.

### Tests and quality checks

```sh
make test
make lint
make typecheck
```

Run `make format` to format the codebase.

### PostgreSQL integration tests

After configuring `.env`, start PostgreSQL, apply migrations, and run only the integration tests:

```sh
docker compose up -d postgres
uv run alembic upgrade head
uv run pytest -m integration -o addopts=-q
```

Optionally stop PostgreSQL afterward:

```sh
docker compose stop postgres
```

### One-shot closed-candle ingestion

With PostgreSQL migrated and the environment configured, run the latest fully
closed five-minute Binance Spot candle ingestion, complete Feature pack, and
baseline technical Score calculation once:

```sh
python -m app.jobs.ingest_closed_candles
```

`BINANCE_INGESTION_LIMIT` controls the requested candles per asset and defaults
to 12 (one hour of five-minute candles). The command processes BTC/USDT followed
by ETH/USDT and commits candles, Features, and Scores in three explicit
transactions before closing its resources and exiting. A later-stage failure
preserves earlier committed stages and can recover on the next run. It never
ingests the current open candle. Repeated
execution must be scheduled by the external deployment platform; Argos embeds
no scheduler.

### Read-only market data

Fetch the latest persisted BTC/USDT five-minute candles:

```sh
curl "http://localhost:8000/market/candles?asset=BTC%2FUSDT&interval=5m"
```

Request candles in ascending chronological order:

```sh
curl "http://localhost:8000/market/candles?asset=BTC%2FUSDT&interval=5m&order=asc"
```

Inspect ingestion freshness for both supported assets:

```sh
curl "http://localhost:8000/market/status"
```

These endpoints only read persisted candles. They never trigger ingestion.

### Analytical feature calculation

The ingestion job calculates EMA 20, EMA 50, RSI 14, ATR 14, one- and
twelve-candle fractional returns, and twenty-candle relative quote volume after
committing each closed-candle batch. Each feature is persisted only when its
history requirement is satisfied; these measurements are not trading decisions.
For an explicit repair or historical calculation after at least 20 closed
candles exist, run the standalone command:

```sh
python -m app.jobs.calculate_market_features
```

Feature calculation retains complete candle history for deterministic indicators,
filters identities already persisted, and sends only missing values to PostgreSQL.
Large historical reconstruction is split into bounded 1,000-row inserts while the
caller continues to own the transaction.

Repeated execution is idempotent. Fetch the latest same-candle snapshot with:

```sh
curl "http://localhost:8000/market/features/latest?asset=BTC%2FUSDT&interval=5m"
```

See [the analytical feature reference](docs/005-analytical-features.md) for exact
formulas, warm-up periods, interpretations, limitations, and versions.

### Baseline technical score

After committing features, the ingestion job calculates and commits the
`baseline_technical` version-1 score for complete same-candle feature sets. The
three transactions are explicit: candles -> features -> scores. A later-stage
failure preserves earlier committed stages, and repeated execution recovers
missing immutable rows idempotently.

Reconstruct missing scores explicitly with:

```sh
python -m app.jobs.calculate_market_scores
```

Read the latest persisted score with:

```sh
curl "http://localhost:8000/market/scores/latest?asset=BTC%2FUSDT&interval=5m"
```

Interpretations are `strongly_bearish`, `bearish`, `neutral`, `bullish`, and
`strongly_bullish`. They describe analytical direction, not a trading action.
See [the score-model reference](docs/006-baseline-score-model.md) for formulas,
weights, boundaries, explanations, and limitations.

The pure, provider-independent `baseline_long_only` version-1 policy is
implemented in `app.strategies.baseline_long_only`. It converts a validated
`baseline_technical` version-1 overall score and open-position state into typed
Decision intent without persistence, execution, scheduling, or provider access.

### Decision Snapshots

Decision generation is a separate responsibility from ingestion. The planned
Railway one-shot command is:

```sh
python -m app.jobs.generate_market_decisions
```

It runs one persisted Paper Decision Cycle for each configured instrument and
contains no internal scheduler. The external Railway cron is `2,32 * * * *`:
minute 02 evaluates the completed :00 UTC boundary and minute 32 evaluates :30.
This delay allows the separate five-minute ingestion job to persist closed-candle
Features and Scores first. It never defaults `position_is_open` to false and
never infers execution from an earlier Decision.

Persisted Decision Snapshots retain the source Score, market evidence, supplied
position state, Strategy and configuration versions, requested sizing, applied
rule, and explanation. `OPEN_POSITION` requests a 15% target allocation,
`SELL_ALL` requests 100% liquidation of an existing position, and `NO_ACTION`
leaves the portfolio unchanged. These are requested intents, not executed
trades; `HOLD` is not an action.

Read the latest persisted snapshot without triggering generation:

```sh
curl "http://localhost:8000/market/decisions/latest?asset=BTC%2FUSDT&interval=5m"
```

### Paper portfolio valuation

Read the active Paper Trading portfolio, or select a historical run, without
creating Decisions or Trades:

```sh
curl "http://localhost:8000/paper/portfolio"
curl "http://localhost:8000/paper/portfolio?run_id=<uuid>"
```

The response is a read-only mark-to-market valuation from persisted closed
candles. For each open Position, `cost_basis` is entry notional plus accumulated
entry fees, `market_value` is quantity times the latest compatible candle close,
and `unrealized_pnl` is market value minus cost basis. Portfolio `equity` is
current cash plus all Position market values; `total_pnl` is equity minus initial
cash. Returns divide their corresponding P&L by a positive cost basis or initial
cash and are otherwise null.

`valued_at` is the UTC time when Argos assembled the response. Each Position also
reports the exact `mark_candle_open_time`, so prices from different candle times
are never presented as simultaneous evidence. Missing compatible persisted
evidence fails explicitly. Unrealized P&L does not deduct hypothetical exit fees
or slippage in version 1. BTC/USDT and ETH/USDT are current MVP configuration,
not special cases in the valuation algorithm.

## Production deployment

Provide `DATABASE_URL` and review the production values for `BINANCE_BASE_URL`,
`BINANCE_CONNECT_TIMEOUT_SECONDS`, `BINANCE_READ_TIMEOUT_SECONDS`,
`BINANCE_INGESTION_LIMIT`, `LOG_LEVEL`, and `CORS_ALLOWED_ORIGINS`. The CORS
setting is a comma-separated list of exact HTTP origins; wildcards and
credentialed origins are rejected. Development-only `POSTGRES_*` variables are
not needed when `DATABASE_URL` is supplied.

Run the API process with:

```sh
scripts/start_api.sh
```

The script applies `alembic upgrade head` before starting one Uvicorn worker on
`0.0.0.0:${PORT:-8000}`. A migration failure prevents the server from starting.
Deployment configuration must avoid running the same migration concurrently in
multiple API instances.

Run ingestion as a separate one-shot process:

```sh
python -m app.jobs.ingest_closed_candles
```

The external platform should invoke it once every five minutes, shortly after a
five-minute UTC boundary. The overlap window and PostgreSQL natural uniqueness
make repeated or overlapping invocations idempotent; this is not an exactly-once
scheduling guarantee. Argos contains no internal scheduler or infinite loop.

Run Decision generation and Paper execution as a separate one-shot process:

```sh
python -m app.jobs.generate_market_decisions
```

Each instrument has an independently committed cycle. PostgreSQL uniqueness
makes completed retries and overlapping invocations no-ops; this is not an
exactly-once scheduling claim. `NO_ACTION` is a completed auditable cycle without
a Trade. The job contacts no provider or exchange.

Use `/health` for API liveness and `/health/database` for PostgreSQL readiness.
Neither endpoint contacts Binance or exposes connection details.

### Production smoke test

With the API running and example host adjusted for the deployment:

```sh
curl http://localhost:8000/health
curl http://localhost:8000/health/database
python -m app.jobs.ingest_closed_candles
curl http://localhost:8000/market/status
curl "http://localhost:8000/market/candles?asset=BTC%2FUSDT&interval=5m"
```

### Docker

After configuring `.env`, build and start the API and PostgreSQL:

```sh
make docker
```

The API starts independently of PostgreSQL readiness. Its database health endpoint returns HTTP 503 while PostgreSQL is unavailable.

### Paper Trading administration

Start or reset the active simulated account without deleting prior experiments:

```sh
python -m app.jobs.reset_paper_account --initial-cash 10000 --reason "baseline before feature X"
```

The command parses money as `Decimal`, stops the active experiment, and creates a fresh linked run in one transaction. Stopping a run does not liquidate it: open Positions retain their quantity and entry state, and no synthetic Trade is created. Historical runs are queryable through `GET /paper/runs` and `GET /paper/runs/{run_id}`; positions and trades accept an optional `run_id`. Reset remains CLI-only until authenticated mutation endpoints exist, and a future authenticated frontend action must reuse this use case.
