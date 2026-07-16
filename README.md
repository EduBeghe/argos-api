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
closed five-minute Binance Spot candle ingestion once:

```sh
python -m app.jobs.ingest_closed_candles
```

`BINANCE_INGESTION_LIMIT` controls the requested candles per asset and defaults
to 12 (one hour of five-minute candles). The command processes BTC/USDT followed
by ETH/USDT, commits the run atomically through ingestion orchestration, closes
its resources, and exits. It never ingests the current open candle. Repeated
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

### Docker

After configuring `.env`, build and start the API and PostgreSQL:

```sh
make docker
```

The API starts independently of PostgreSQL readiness. Its database health endpoint returns HTTP 503 while PostgreSQL is unavailable.
