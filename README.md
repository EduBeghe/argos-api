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
closed five-minute Binance Spot candle ingestion and EMA 20 calculation once:

```sh
python -m app.jobs.ingest_closed_candles
```

`BINANCE_INGESTION_LIMIT` controls the requested candles per asset and defaults
to 12 (one hour of five-minute candles). The command processes BTC/USDT followed
by ETH/USDT and commits those candles atomically. It then calculates EMA 20 for
both assets in a separate atomic transaction before closing its resources and
exiting. A feature failure preserves committed candles and can recover on the
next run. It never ingests the current open candle. Repeated
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

Repeated execution is idempotent. Fetch the latest same-candle snapshot with:

```sh
curl "http://localhost:8000/market/features/latest?asset=BTC%2FUSDT&interval=5m"
```

See [the analytical feature reference](docs/005-analytical-features.md) for exact
formulas, warm-up periods, interpretations, limitations, and versions.

### Baseline technical score

After committing features, the ingestion job calculates and commits the
`baseline_technical` version-1 score for complete same-candle feature sets. The
three transactions are explicit: candles â†’ features â†’ scores. A later-stage
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
