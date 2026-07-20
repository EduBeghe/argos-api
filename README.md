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

### Read-only instrument history

The historical collections use the complete catalog identity: `asset`,
`provider`, `market_type`, and `interval`. For backward compatibility,
`provider` and `market_type` may be omitted when `asset` plus `interval` resolves
to exactly one enabled catalog instrument. Unknown instruments return HTTP 404;
ambiguous partial identities return HTTP 422 and must be completed explicitly.

```sh
curl "http://localhost:8000/market/candles?asset=BTC%2FUSDT&provider=binance&market_type=spot&interval=5m&from=2026-07-15T12%3A00%3A00Z&to=2026-07-15T14%3A00%3A00Z&order=asc&limit=500"
curl "http://localhost:8000/market/scores?asset=BTC%2FUSDT&provider=binance&market_type=spot&interval=5m&order=desc&limit=100"
curl "http://localhost:8000/market/decisions?asset=BTC%2FUSDT&provider=binance&market_type=spot&interval=5m&run_id=<paper-run-id>&order=asc&limit=100"
curl "http://localhost:8000/paper/trades?run_id=<paper-run-id>&asset=BTC%2FUSDT&provider=binance&market_type=spot&interval=5m&order=asc&limit=100"
```

`from` and `to` are optional, inclusive, timezone-aware ISO-8601 bounds;
offset-aware values are normalized to UTC and naive timestamps are rejected.
`order` defaults to `desc`, `limit` defaults to 100, and the maximum is 1000.
Ordering is deterministic by each collection's event time and immutable ID.
Valid identities with no matching rows return HTTP 200 with `[]`.

These endpoints return persisted Candles, Scores, immutable Decision Snapshots,
and simulated Paper Trades only. Reads never contact a provider, calculate
analytics, generate Decisions, infer Trades, or execute Paper Trading actions.
Score responses retain model, contribution, and explanation fields; Decision
responses retain source evidence, versions, sizing intent, rule, explanation,
and position state; Trade responses retain prices, quantities, costs, cash
impact, and immutable run, Decision, and Position links. `NO_ACTION` Decisions
therefore appear in Decision history but never fabricate a Trade.

The older candle parameters `start_time` and `end_time` remain supported. Do not
combine them with `from` and `to` respectively.

Inspect ingestion freshness for both supported assets:

```sh
curl "http://localhost:8000/market/status"
```

List the configured market instruments and their enabled capabilities:

```sh
curl "http://localhost:8000/market/instruments"
```

Load the data needed for the initial dashboard in one read-only request:

```sh
curl "http://localhost:8000/dashboard/overview"
```

The overview builds instrument cards from catalog display order and returns
market freshness, the latest persisted Score and Decision, and the active Paper
Run summary when available. Missing market or analytical evidence is represented
explicitly with null or stale fields. If an active Paper portfolio cannot be
valued from persisted candles, the endpoint returns HTTP 409 instead of inventing
a value. Detailed endpoints remain the source for equity history, trades, and
other historical collections.

An asset is Argos's canonical market identity; an instrument combines that asset
with provider, market type, interval, provider symbol, presentation metadata,
and enabled capabilities. The immutable in-code catalog is the MVP source of
truth. The current argos-ingest job consumes only the validated Binance Spot
five-minute projection; it fails before I/O if that projection is empty, mixed,
duplicated, or incompatible. Feature/score calculation and Decision jobs
discover work through their capabilities, while existing read endpoints resolve
provider and market type from the same catalog. The API exposes the metadata for
a data-driven frontend. Paper Trading capability is derived from Decision
capability in the MVP.

BTC/USDT and ETH/USDT are current configuration, not special application or
frontend entities. SPY and Alpaca are outside the current MVP. A future equity
source will require its own catalog projection, provider adapter/job, and
market-session rules, but
not symbol-specific UI components. The catalog can move to persistence without
changing its read contract if instrument volume or dynamic administration later
justifies it.

The frontend should render dashboard cards from the catalog rather than contain
BTC- or ETH-specific components. Adding a future catalog instrument does not
change the dashboard response contract.

These endpoints are read-only. They never trigger ingestion or calculations.

### Analytical feature calculation

The ingestion job calculates EMA 20, EMA 50, RSI 14, ATR 14, one- and
twelve-candle fractional returns, and twenty-candle relative quote volume after
committing each closed-candle batch. Each feature is persisted only when its
history requirement is satisfied; these measurements are not trading decisions.
Live calculation targets only the current ingestion window and loads at most
1,000 preceding/current candles per configured instrument as analytical context.
This keeps live memory and query cost independent of total historical candle
storage. Candle-only backfills are intentionally excluded; the overlap window
still retries missing recent Features and Scores after a partial pipeline failure.
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

After every successfully completed 30-minute Decision boundary, `argos-decide`
persists one immutable equity snapshot for the Paper Run. The snapshot records
cash, Position value, equity, P&L, return, and relational detail for every open
Position. `scheduled_boundary` is the market boundary being valued;
`recorded_at` is when the snapshot was persisted. Marks use only compatible
persisted candles closed at or before the boundary.

```sh
curl "http://localhost:8000/paper/equity-snapshots"
curl "http://localhost:8000/paper/equity-snapshots?run_id=<uuid>&order=asc"
curl "http://localhost:8000/paper/equity-snapshots/<snapshot-id>"
```

Snapshots are unique by run and boundary, so retries are idempotent. Instrument
cycles retain their independent commits; if any cycle fails, no aggregate
snapshot is created. Snapshot persistence uses its own atomic transaction and a
failure does not undo completed cycles, but it does fail the job for a later
retry. The formulas match the live read-only valuation and do not estimate
hypothetical exit fees or slippage. All instruments in a run must currently be
valuable in its account currency; explicit multi-currency conversion is future
work. Configured BTC and ETH instruments are not special cases.

Read the historical performance summary for the active Paper Run or a selected
active or stopped run:

```sh
curl "http://localhost:8000/paper/performance"
curl "http://localhost:8000/paper/performance?run_id=<uuid>"
```

Evaluate the active Paper Run, or one preserved historical run, using only
persisted evidence through the latest successful Equity Snapshot:

```sh
curl "http://localhost:8000/paper/evaluation"
curl "http://localhost:8000/paper/evaluation?run_id=<uuid>"
```

The response separates run metadata, Strategy results, the
`equal_weight_buy_and_hold_v1` benchmark, their comparison, and deterministic
per-instrument results ke{}¼×Kh‘éì¶»§q«^uµ…±±}‰Õ‘•Ğ ¤°(€€€€€€€€¤(€€€…ÍÍ•ÉĞÉ•ÍÕ±Ğ¹ÍÉ••¹¥¹}Í•µ•¹Ğ¹‘•¥Í¥½¹}½Õ¹Ğ€ôô€Ì(€€€…ÍÍ•ÉĞ…±°¡Í¥é”€ôô€Ì™½ÈÍ¥é”°|°|¥¸…±±ÍlèÄÜÙt¤(€€€…ÍÍ•ÉĞ€ (€€€€€€€µ…ÑÉ¥•ÍmI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹Y1%Q%=9t¹É½İÍlÁt(€€€€€€€¹½Ğ¥¸µ…ÑÉ¥•ÍmI•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QI%9%9t¹É½İÌ(€€€€¤(()‘•˜Ñ•ÍÑ}‰Õ‘•Ñ}±¥µ¥ÑÍ}™…¥±}±½Í•‘}İ¥Ñ¡½ÕÑ}Í¥±•¹Ñ}ÑÉÕ¹…Ñ¥½¸ ¤€´ø9½¹”è(€€€İ¥Ñ ÁåÑ•ÍĞ¹É…¥Í•Ì¡!¥ÍÑ½É¥…±I•™¥¹•µ•¹ÑÉÉ½È°µ…Ñ ô‰Í½É”½¹™¥ÕÉ…Ñ¥½¹Ìˆ¤è(€€€€€€€É•™¥¹•}¡¥ÍÑ½É¥…±}ÍÑÉ…Ñ•ä (€€€€€€€€€€€‘…Ñ…Í•Ğõ‘…Ñ…Í•Ğ ¤°(€€€€€€€€€€€™•…ÑÕÉ•}µ…ÑÉ¥àõµ…ÑÉ¥à ¤°(€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ•¥µ…° ˆÄÀÀÀÀˆ¤°(€€€€€€€€€€€™••}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÄˆ¤°(€€€€€€€€€€€Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÀÔˆ¤°(€€€€€€€€€€€‰Õ‘•ĞõI•™¥¹•µ•¹Ñ	Õ‘•Ğ Ä°€Ä°€ÄÜÔ°€ØÀ¤°(€€€€€€€€¤(€€€İ¥Ñ ÁåÑ•ÍĞ¹É…¥Í•Ì¡!¥ÍÑ½É¥…±I•™¥¹•µ•¹ÑÉÉ½È°µ…Ñ ô‰©½¥¹Ğ…¹‘¥‘…Ñ•Ìˆ¤è(€€€€€€€É•™¥¹•}¡¥ÍÑ½É¥…±}ÍÑÉ…Ñ•ä (€€€€€€€€€€€‘…Ñ…Í•Ğõ‘…Ñ…Í•Ğ ¤°(€€€€€€€€€€€™•…ÑÕÉ•}µ…ÑÉ¥àõµ…ÑÉ¥à ¤°(€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ•¥µ…° ˆÄÀÀÀÀˆ¤°(€€€€€€€€€€€™••}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÄˆ¤°(€€€€€€€€€€€Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÀÔˆ¤°(€€€€€€€€€€€‰Õ‘•ĞõI•™¥¹•µ•¹Ñ	Õ‘•Ğ È°€Ä°€ÈÀÀ°€ÄÄä¤°(€€€€€€€€¤(()‘•˜Ñ•ÍÑ}½¹±å}É•Ñ…¥¹•‘}Í½É•Í}É•…¡}©½¥¹Ñ}É•™¥¹•µ•¹Ñ}…¹‘}µ…ÑÉ¥á}‰Õ¥±‘Í}½¹” ¤€´ø9½¹”è(€€€…±±Ì°™…­”€ô™…­•}Í¥µÕ±…Ñ½È ¤(€€€İ¥Ñ € (€€€€€€€Á…Ñ  (€€€€€€€€€€€€‰…ÁÀ¹½ÁÑ¥µ¥é…Ñ¥½¸¹É•™¥¹•µ•¹Ğ¹‰Õ¥±‘}¡¥ÍÑ½É¥…±}™•…ÑÕÉ•}µ…ÑÉ¥àˆ°(€€€€€€€€€€€É•ÑÕÉ¹}Ù…±Õ”õµ…ÑÉ¥à ¤°(€€€€€€€€¤…Ì‰Õ¥±°(€€€€€€€Á…Ñ  (€€€€€€€€€€€€‰…ÁÀ¹½ÁÑ¥µ¥é…Ñ¥½¸¹É•™¥¹•µ•¹Ğ¹Í¥µÕ±…Ñ•}¡¥ÍÑ½É¥…±}™•…ÑÕÉ•}µ…ÑÉ¥àˆ°(€€€€€€€€€€€Í¥‘•}•™™•Ğõ™…­”°(€€€€€€€€¤°(€€€€¤è(€€€€€€€É•ÍÕ±Ğ€ôÉ•™¥¹•}¡¥ÍÑ½É¥…±}‘…Ñ…Í•Ğ (€€€€€€€€€€€‘…Ñ…Í•Ğõ‘…Ñ…Í•Ğ ¤°(€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ•¥µ…° ˆÄÀÀÀÀˆ¤°(€€€€€€€€€€€™••}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÄˆ¤°(€€€€€€€€€€€Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÀÔˆ¤°(€€€€€€€€€€€‰Õ‘•ĞõÍµ…±±}‰Õ‘•Ğ ¤°(€€€€€€€€¤(€€€‰Õ¥±¹…ÍÍ•ÉÑ}…±±•‘}½¹” ¤(€€€…ÍÍ•ÉĞÉ•ÍÕ±Ğ¹½Õ¹ÑÌ¹Í½É•}½¹™¥ÕÉ…Ñ¥½¹Í}É•Ñ…¥¹•€ôô€Ä(€€€…ÍÍ•ÉĞÉ•ÍÕ±Ğ¹½Õ¹ÑÌ¹©½¥¹Ñ}…¹‘¥‘…Ñ•Í}•Ù…±Õ…Ñ•€ôô€ØÀ(€€€É•Ñ…¥¹•€ôÍ•Ğ¡É•ÍÕ±Ğ¹É•Ñ…¥¹•‘}Í½É•}Á…É…µ•Ñ•ÉÌ¤(€€€…ÍÍ•ÉĞì(€€€€€€€¥Ñ•´¹…¹‘¥‘…Ñ”¹Í½É•}Á…É…µ•Ñ•ÉÌ™½È¥Ñ•´¥¸É•ÍÕ±Ğ¹ÑÉ…¥¹¥¹}É…¹­¥¹œ(€€€ô€ôôÉ•Ñ…¥¹•(€€€…ÍÍ•ÉĞ…±°¡…±±lÅt¥¸É•Ñ…¥¹•™½È…±°¥¸…±±ÍlÄÜØèÈÌÙt¤(()‘•˜Ñ•ÍÑ}É•½µµ•¹‘…Ñ¥½¹}‰•…ÑÍ}‰…Í•±¥¹•}…¹‘}Ñ•ÍÑ}…¹¹½Ñ}¡…¹•}Í•±•Ñ¥½¸ ¤€´ø9½¹”è(€€€Í•±•Ñ•€ômt(€€€ÍÑ…ÑÕÍ•Ì€ômt(€€€™½ÈÑ•ÍÑ}É•ÑÕÉ¸¥¸€¡•¥µ…° ˆÀ¸ÀÌˆ¤°•¥µ…° ˆ´À¸ÀÌˆ¤¤è(€€€€€€€|°™…­”€ô™…­•}Í¥µÕ±…Ñ½È¡Ñ•ÍÑ}É•ÑÕÉ¸¤(€€€€€€€İ¥Ñ Á…Ñ  (€€€€€€€€€€€€‰…ÁÀ¹½ÁÑ¥µ¥é…Ñ¥½¸¹É•™¥¹•µ•¹Ğ¹Í¥µÕ±…Ñ•}¡¥ÍÑ½É¥…±}™•…ÑÕÉ•}µ…ÑÉ¥àˆ°(€€€€€€€€€€€Í¥‘•}•™™•Ğõ™…­”°(€€€€€€€€¤è(€€€€€€€€€€€É•ÍÕ±Ğ€ôÉ•™¥¹•}¡¥ÍÑ½É¥…±}ÍÑÉ…Ñ•ä (€€€€€€€€€€€€€€€‘…Ñ…Í•Ğõ‘…Ñ…Í•Ğ ¤°(€€€€€€€€€€€€€€€™•…ÑÕÉ•}µ…ÑÉ¥àõµ…ÑÉ¥à ¤°(€€€€€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ•¥µ…° ˆÄÀÀÀÀˆ¤°(€€€€€€€€€€€€€€€™••}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÄˆ¤°(€€€€€€€€€€€€€€€Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÀÔˆ¤°(€€€€€€€€€€€€€€€‰Õ‘•ĞõÍµ…±±}‰Õ‘•Ğ ¤°(€€€€€€€€€€€€¤(€€€€€€€Í•±•Ñ•¹…ÁÁ•¹ (€€€€€€€€€€€É•ÍÕ±Ğ¹É•½µµ•¹‘…Ñ¥½¸¹…¹‘¥‘…Ñ”¥˜É•ÍÕ±Ğ¹É•½µµ•¹‘…Ñ¥½¸•±Í”9½¹”(€€€€€€€€¤(€€€€€€€ÍÑ…ÑÕÍ•Ì¹…ÁÁ•¹¡É•ÍÕ±Ğ¹É•½µµ•¹‘…Ñ¥½¹}ÍÑ…ÑÕÌ¤(€€€€€€€…ÍÍ•ÉĞ€ (€€€€€€€€€€€É•ÍÕ±Ğ¹Ù…±¥‘…Ñ¥½¹}É•ÍÕ±ÑÍlÁt¹Ñ½Ñ…±}É•ÑÕÉ¸(€€€€€€€€€€€€øÉ•ÍÕ±Ğ¹ÁÉ½‘ÕÑ¥½¹}‰…Í•±¥¹”¹Ù…±¥‘…Ñ¥½¸¹Ñ½Ñ…±}É•ÑÕÉ¸(€€€€€€€€¤(€€€€€€€…ÍÍ•ÉĞ€‰Í½É”İ•¥¡ÑÌˆ¥¸É•ÍÕ±Ğ¹É•½µµ•¹‘…Ñ¥½¹}•áÁ±…¹…Ñ¥½¸(€€€€€€€…ÍÍ•ÉĞ€‰ÁÉ½‘ÕÑ¥½¸‰…Í•±¥¹”ˆ¥¸É•ÍÕ±Ğ¹É•½µµ•¹‘…Ñ¥½¹}•áÁ±…¹…Ñ¥½¸(€€€€€€€…ÍÍ•ÉĞ€‰‰Õä…¹¡½±ˆ¥¸É•ÍÕ±Ğ¹É•½µµ•¹‘…Ñ¥½¹}•áÁ±…¹…Ñ¥½¸(€€€…ÍÍ•ÉĞÍ•±•Ñ•‘lÁt€ôôÍ•±•Ñ•‘lÅt(€€€…ÍÍ•ÉĞÍÑ…ÑÕÍ•Ì€ôôl(€€€€€€€I•½µµ•¹‘…Ñ¥½¹MÑ…ÑÕÌ¹I=559}9}QMQ}=9%I5°(€€€€€€€I•½µµ•¹‘…Ñ¥½¹MÑ…ÑÕÌ¹I=559}	UQ}QMQ}%1°(€€€t(()‘•˜Ñ•ÍÑ}İ½ÉÍ•}Ñ¡…¹}‰…Í•±¥¹•}¥Í}¹½Ñ}É•½µµ•¹‘• ¤€´ø9½¹”è(€€€|°™…­”€ô™…­•}Í¥µÕ±…Ñ½È ¤((€€€‘•˜İ½ÉÍ” ¨©­İ…ÉÌè½‰©•Ğ¤€´ø!¥ÍÑ½É¥…±M¥µÕ±…Ñ¥½¹I•ÍÕ±Ğè(€€€€€€€Ù…±Õ”€ô…ÍĞ¡M¥µÁ±•9…µ•ÍÁ…”°™…­” ¨©­İ…ÉÌ¤¤(€€€€€€€µ…ÑÉ¥á}Ù…±Õ”€ô…ÍĞ¡!¥ÍÑ½É¥…±•…ÑÕÉ•5…ÑÉ¥à°­İ…ÉÍl‰™•…ÑÕÉ•}µ…ÑÉ¥à‰t¤(€€€€€€€Í½É”€ô…ÍĞ¡	…Í•±¥¹•Q•¡¹¥…±A…É…µ•Ñ•ÉÌ°­İ…ÉÍl‰Í½É•}Á…É…µ•Ñ•ÉÌ‰t¤(€€€€€€€ÍÑÉ…Ñ•ä€ô…ÍĞ¡	…Í•±¥¹•1½¹=¹±åA…É…µ•Ñ•ÉÌ°­İ…ÉÍl‰ÍÑÉ…Ñ•å}Á…É…µ•Ñ•ÉÌ‰t¤(€€€€€€€‰…Í•±¥¹”€ô€ (€€€€€€€€€€€Í½É”€ôô	M1%9}Q!9%1}XÅ}AI5QIL(€€€€€€€€€€€…¹ÍÑÉ…Ñ•ä€ôô	M1%9}1=9}=91e}XÅ}AI5QIL(€€€€€€€€¤(€€€€€€€¥˜€ (€€€€€€€€€€€±•¸¡µ…ÑÉ¥á}Ù…±Õ”¹É½İÌ¤€ôô€È(€€€€€€€€€€€…¹µ…ÑÉ¥á}Ù…±Õ”¹É½İÍlÁt¹…¹‘±•}½Á•¹}Ñ¥µ”¹¡½ÕÈ€ôô€Ì(€€€€€€€€¤è(€€€€€€€€€€€Ù…±Õ”¹Ñ½Ñ…±}É•ÑÕÉ¸€ô•¥µ…° ˆÀ¸ÀÈˆ¤¥˜‰…Í•±¥¹”•±Í”•¥µ…° ˆÀ¸ÀÄˆ¤(€€€€€€€É•ÑÕÉ¸…ÍĞ¡!¥ÍÑ½É¥…±M¥µÕ±…Ñ¥½¹I•ÍÕ±Ğ°Ù…±Õ”¤((€€€İ¥Ñ Á…Ñ  (€€€€€€€€‰…ÁÀ¹½ÁÑ¥µ¥é…Ñ¥½¸¹É•™¥¹•µ•¹Ğ¹Í¥µÕ±…Ñ•}¡¥ÍÑ½É¥…±}™•…ÑÕÉ•}µ…ÑÉ¥àˆ°(€€€€€€€Í¥‘•}•™™•Ğõİ½ÉÍ”°(€€€€¤è(€€€€€€€É•ÍÕ±Ğ€ôÉ•™¥¹•}¡¥ÍÑ½É¥…±}ÍÑÉ…Ñ•ä (€€€€€€€€€€€‘…Ñ…Í•Ğõ‘…Ñ…Í•Ğ ¤°(€€€€€€€€€€€™•…ÑÕÉ•}µ…ÑÉ¥àõµ…ÑÉ¥à ¤°(€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ•¥µ…° ˆÄÀÀÀÀˆ¤°(€€€€€€€€€€€™••}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÄˆ¤°(€€€€€€€€€€€Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÀÔˆ¤°(€€€€€€€€€€€‰Õ‘•ĞõÍµ…±±}‰Õ‘•Ğ ¤°(€€€€€€€€¤(€€€…ÍÍ•ÉĞÉ•ÍÕ±Ğ¹É•½µµ•¹‘…Ñ¥½¸¥Ì9½¹”(€€€…ÍÍ•ÉĞÉ•ÍÕ±Ğ¹É•½µµ•¹‘…Ñ¥½¹}ÍÑ…ÑÕÌ¥ÌI•½µµ•¹‘…Ñ¥½¹MÑ…ÑÕÌ¹9=}Y1%Q%=9}9%Q(()‘•˜Ñ•ÍÑ}…‰±…Ñ¥½¹}½Ù•ÉÍ}•Ù•Éå}Í¥¹…±}…¹‘}É•ÍÕ±Ñ}¥Í}‘•Ñ•Éµ¥¹¥ÍÑ¥Œ ¤€´ø9½¹”è(€€€É•ÍÕ±ÑÌ€ômt(€€€™½È|¥¸É…¹” È¤è(€€€€€€€|°™…­”€ô™…­•}Í¥µÕ±…Ñ½È ¤(€€€€€€€İ¥Ñ Á…Ñ  (€€€€€€€€€€€€‰…ÁÀ¹½ÁÑ¥µ¥é…Ñ¥½¸¹É•™¥¹•µ•¹Ğ¹Í¥µÕ±…Ñ•}¡¥ÍÑ½É¥…±}™•…ÑÕÉ•}µ…ÑÉ¥àˆ°(€€€€€€€€€€€Í¥‘•}•™™•Ğõ™…­”°(€€€€€€€€¤è(€€€€€€€€€€€É•ÍÕ±ÑÌ¹…ÁÁ•¹ (€€€€€€€€€€€€€€€É•™¥¹•}¡¥ÍÑ½É¥…±}ÍÑÉ…Ñ•ä (€€€€€€€€€€€€€€€€€€€‘…Ñ…Í•Ğõ‘…Ñ…Í•Ğ ¤°(€€€€€€€€€€€€€€€€€€€™•…ÑÕÉ•}µ…ÑÉ¥àõµ…ÑÉ¥à ¤°(€€€€€€€€€€€€€€€€€€€¥¹¥Ñ¥…±}…Í õ•¥µ…° ˆÄÀÀÀÀˆ¤°(€€€€€€€€€€€€€€€€€€€™••}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÄˆ¤°(€€€€€€€€€€€€€€€€€€€Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÀÔˆ¤°(€€€€€€€€€€€€€€€€€€€‰Õ‘•ĞõÍµ…±±}‰Õ‘•Ğ ¤°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€…ÍÍ•ÉĞÉ•ÍÕ±ÑÍlÁt€ôôÉ•ÍÕ±ÑÍlÅt(€€€…ÍÍ•ÉĞí¥Ñ•´¹‘¥Í…‰±•‘}Í¥¹…°™½È¥Ñ•´¥¸É•ÍÕ±ÑÍlÁt¹…‰±…Ñ¥½¹}ÍÕµµ…Éåô€ôôì(€€€€€€€€‰ÑÉ•¹ˆ°(€€€€€€€€‰ÉÍ¤ˆ°(€€€€€€€€‰É•ÑÕÉ¹|Äˆ°(€€€€€€€€‰É•ÑÕÉ¹|ÄÈˆ°(€€€€€€€€‰Ù½±Õµ•}½¹™¥Éµ…Ñ¥½¸ˆ°(€€€ô(()‘•˜Ñ•ÍÑ}‰Õå}…¹‘}¡½±‘}…ÁÁ±¥•Í}•¹ÑÉå}•á¥Ñ}™••Í}…¹‘}Í±¥ÁÁ…” ¤€´ø9½¹”è(€€€Ù…±Õ”€ô}‰Õå}…¹‘}¡½± (€€€€€€€!¥ÍÑ½É¥…±•…ÑÕÉ•5…ÑÉ¥à ‰™¥¹•ÉÁÉ¥¹Ğˆ°µ…ÑÉ¥à ¤¹É½İÍlèÉt¤°(€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€%9MQIU59P°(€€€€€€€•¥µ…° ˆÄÀÀÀÀ¸ÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀˆ¤°(€€€€€€€M¥µÕ±…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¸ (€€€€€€€€€€€™••}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÄˆ¤°Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° ˆÀ¸ÀÀÀÔˆ¤(€€€€€€€€¤°(€€€€¤(€€€İ¥Ñ¡½ÕÑ}½ÍÑÌ€ô}‰Õå}…¹‘}¡½± (€€€€€€€!¥ÍÑ½É¥…±•…ÑÕÉ•5…ÑÉ¥à ‰™¥¹•ÉÁÉ¥¹Ğˆ°µ…ÑÉ¥à ¤¹É½İÍlèÉt¤°(€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¹QMP°(€€€€€€€%9MQIU59P°(€€€€€€€•¥µ…° ˆÄÀÀÀÀ¸ÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀˆ¤°(€€€€€€€M¥µÕ±…Ñ¥½¹½¹™¥ÕÉ…Ñ¥½¸¡™••}É…Ñ”õ•¥µ…° À¤°Í±¥ÁÁ…•}É…Ñ”õ•¥µ…° À¤¤°(€€€€¤(€€€…ÍÍ•ÉĞÙ…±Õ”¹ÑÉ…‘•}½Õ¹Ğ€ôô€È(€€€…ÍÍ•ÉĞÙ…±Õ”¹™••Í}Á…¥€ø€À(€€€…ÍÍ•ÉĞÙ…±Õ”¹™¥¹…±}•ÅÕ¥Ñä€ğİ¥Ñ¡½ÕÑ}½ÍÑÌ¹™¥¹…±}•ÅÕ¥Ñä