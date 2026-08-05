# Argos API

Argos is a private, asset-agnostic quantitative research and predictive decision-support platform. It estimates probabilities from measurable evidence; it does not claim to predict the future with certainty or guarantee profitability.

The current research phase includes Predictive Edge Discovery v4: a bounded
training/validation funnel, isolated test confirmation, explicit rejection margins,
near-miss and threshold-sensitivity diagnostics, and read-only evidence APIs. It does
not activate Strategies or claim that historical relationships will persist.

The platform will ingest market observations, calculate explicit and explainable component metrics, generate simulated decisions, execute paper trades, and evaluate results after fees and simulated slippage. Potential analytical components include technical analysis, momentum, volatility, market context, news, sentiment, and machine-learning outputs. These independent scores may later contribute to an aggregated Argos Score.

Every generated signal must retain its component scores, supporting metrics, and reasoning. The initial MVP will study BTC/USDT and ETH/USDT on Binance Spot, while keeping the domain model suitable for other providers, instruments, and asset classes.

## Documentation

- [Documentation index](docs/README.md)
- [Current system reference](docs/010-current-system.md)
- [API and frontend contract](docs/011-api-and-frontend-contract.md)
- [Railway operations runbook](docs/012-railway-operations.md)
- [Glossary](docs/013-glossary.md)
- [Project vision](docs/000-project-vision.md)
- [MVP scope](docs/001-mvp-scope.md)
- [Architecture overview](docs/002-architecture.md)
- [Domain model](docs/003-domain-model.md)
- [Binance market-data contract](docs/004-binance-market-data-contract.md)
- [Experiment memory and experience-guided search](docs/008-experiment-memory.md)
- [ML experiment dataset contract](docs/ml-experiment-contract.md)
- [ML model screening and refinement](docs/ml-model-screening.md)
- [API authentication and Railway rollout](docs/015-authentication.md)
- [Architecture decisions](docs/decisions/)

## Current phase

Argos currently runs closed-candle ingestion, bounded live Feature and Score
calculation, 30-minute decision cycles, versioned Paper Trading, portfolio and
performance history, historical simulation/refinement, Predictive Edge Map
analysis, and edge-guided candidate qualification v2. The API exposes current
and historical evidence to the separately deployed `argos-web` SPA. Candidate
approval, parallel Strategy wallets, scheduled exploration, additional
providers, ML, sentiment/news, multi-user identity, and real-money trading are not
implemented. See the [current system reference](docs/010-current-system.md).

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

The API is available at `http://localhost:8000`. Public process health is exactly
`{"status":"ok"}` from `GET /health`; PostgreSQL connectivity is reported independently
by `GET /health/database` and becomes private when authentication enforcement is enabled.

### API authentication rollout

Argos includes a single-administrator, HttpOnly-cookie session foundation. Enforcement
defaults to `false` only as temporary migration scaffolding until the separately deployed
frontend supports login. Do not describe a deployment as private until enforcement is
enabled and unauthenticated data requests have been verified to return `401`.

Generate the administrator Argon2id hash interactively:

```sh
python -m app.jobs.hash_admin_password
```

Set `ARGOS_ADMIN_USERNAME`, `ARGOS_ADMIN_PASSWORD_HASH`, a random
`ARGOS_JWT_SECRET` of at least 32 UTF-8 bytes, and the exact frontend origin in
`CORS_ALLOWED_ORIGINS`. See [API authentication and Railway rollout](docs/015-authentication.md)
for the complete cookie, CSRF, rollout, rollback, and secret-rotation contract.

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

Validated ML Paper deployments use a separate immutable ML Decision Snapshot.
They do not create synthetic Market Scores or weaken the baseline snapshot
contract. Paper cycles and Trades reference a common typed decision identity
that resolves to exactly one baseline or ML snapshot; database constraints
reject absent, dual, and type-mismatched evidence.

Protected ML final validation and ML Paper decisions share the same UTC-aligned
30-minute cadence. A configured confidence bucket such as `0.05` is a development
coverage fraction, not a literal prediction threshold: Argos freezes the
corresponding absolute-score quantile from pre-2024 development predictions and
persists that concrete cutoff separately as `signed_score_threshold`. A passed
deployment serializes the exact pre-2024 fitted model and ordered feature set that
scored the protected 2024-2026 evidence; protected rows are never used to refit it.

Read the latest persisted snapshot without triggering generation:

```sh
curl "http://localhost:8000/markço:¶‰ËkºwµçEÀ¥¹Ù…±¥‘…Ñ•ÌÑ¡”‘…Ñ…Í•ĞÉ…Ñ¡•ÈÑ¡…¸‰•¥¹œ™¥±±•°)¥¹Ñ•ÉÁ½±…Ñ•°½È¡¥‘‘•¸¸()Q¡”½µµ…¹…±Õ±…Ñ•ÌÑ¡”ÕÉÉ•¹ĞÍ•Ù•¸µ™•…ÑÕÉ”µ…ÑÉ¥à½¹”¥¸µ•µ½ÉäÕÍ¥¹œ)Ñ¡”Í…µ”ÁÕÉ”¥¹‘¥…Ñ½È™Õ¹Ñ¥½¹Ì…Ì½Á•É…Ñ¥½¹…°¥¹•ÍÑ¥½¸¸]…É´µÕÀ…¹‘±•Ì)İ¥Ñ¡½ÕĞ„½µÁ±•Ñ”•…ÑÕÉ”Í•Ğ…É”½µ¥ÑÑ•¸%Ğ‘½•Ì¹½ĞÁ•ÉÍ¥ÍĞ‘•É¥Ù••…ÑÕÉ”)É½İÌ°…±Õ±…Ñ”M½É•Ì°•Ù…±Õ…Ñ”MÑÉ…Ñ•¥•Ì½ÈÁÉ½™¥Ñ…‰¥±¥Ñä°½ÈÍ¥µÕ±…Ñ”)QÉ…‘•Ì¸AÉ½•ÍÌ½¹”¥¹ÍÑÉÕµ•¹Ğ…Ğ„Ñ¥µ”Í¼¥ÑÌ‘…Ñ…Í•Ğ…¸‰”É•±•…Í•‰•™½É”)Ñ¡”¹•áĞ¥¹ÍÑÉÕµ•¹Ğ¥Ì±½…‘•¸()Q¡”É•Á½ÉÑ•M!´ÈÔØ™¥¹•ÉÁÉ¥¹Ğ½Ù•ÉÌÑ¡”½µÁ±•Ñ”¥¹ÍÑÉÕµ•¹Ğ¥‘•¹Ñ¥Ñä°)É…¹”°•…ÑÕÉ”µ½‘•°½Ù•ÉÍ¥½¸°…¹…¹½¹¥…°½É‘•É•=!1X•Ù¥‘•¹”¸%‘•¹Ñ¥…°)•Ù¥‘•¹”ÁÉ½‘Õ•ÌÑ¡”Í…µ”™¥¹•ÉÁÉ¥¹Ğì…¹ä…¹‘±”¡…¹”ÁÉ½‘Õ•Ì„‘¥™™•É•¹Ğ)½¹”¸Q¡”ÕÉÉ•¹ĞÉ½Õ¡±ä€ÈØà°ÀÀÀ™¥Ù”µµ¥¹ÕÑ”…¹‘±•ÌÁ•È…ÍÍ•Ğ…É”É•…¥¸(Ô°ÀÀÀµÉ½ÜÁ…•Ì¸%¸µµ•µ½ÉäÕÍ…”É•µ…¥¹Ì±¥¹•…È¥¸Ñ¡…Ğ½¹”¥¹ÍÑÉÕµ•¹ĞÌÉ…¹”ì)•á…Ğµ•µ½Éä‘•Á•¹‘Ì½¸AåÑ¡½¸…¹•¥µ…°½‰©•Ğ½Ù•É¡•…°Í¼ÁÉ½‘ÕÑ¥½¸ÉÕ¹Ì)Í¡½Õ±½‰Í•ÉÙ”½¹Ñ…¥¹•Èµ•µ½Éä‰•™½É”¥¹É•…Í¥¹œÑ¡”Í•±•Ñ•É…¹”¸((ŒŒŒ•Ñ•Éµ¥¹¥ÍÑ¥Œ¡¥ÍÑ½É¥…°ÍÑÉ…Ñ•äÍ¥µÕ±…Ñ¥½¸()M¥µÕ±…Ñ”‰…Í•±¥¹•}Ñ•¡¹¥…±€Ù•ÉÍ¥½¸€Ä…¹‰…Í•±¥¹•}±½¹}½¹±å€Ù•ÉÍ¥½¸€Ä™½È)½¹”Ù…±¥‘…Ñ•¡¥ÍÑ½É¥…°¥¹ÍÑÉÕµ•¹Ğİ¥Ñ¡½ÕĞµ½‘¥™å¥¹œ½Á•É…Ñ¥½¹…°½ÈA…Á•È)QÉ…‘¥¹œÍÑ…Ñ”è()Í )ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹Í¥µÕ±…Ñ•}¡¥ÍÑ½É¥…±}ÍÑÉ…Ñ•äp(€€´µ™É½´€ÈÀÈĞ´ÀÄ´ÀÅPÀÀèÀÀèÀÁhp(€€´µÑ¼€ÈÀÈĞ´ÀĞ´ÀÅPÀÀèÀÀèÀÁhp(€€´µ…ÍÍ•Ğ	Q½UMPp(€€´µÁÉ½Ù¥‘•È‰¥¹…¹”p(€€´µµ…É­•ĞµÑåÁ”ÍÁ½Ğp(€€´µ¥¹Ñ•ÉÙ…°€Õ´p(€€´µ¥¹¥Ñ¥…°µ…Í €ÄÀÀÀÀp(€€´µ™•”µÉ…Ñ”€À¸ÀÀÄp(€€´µÍ±¥ÁÁ…”µÉ…Ñ”€À¸ÀÀÀÔ)€()Q¡”•¹¥¹”‰Õ¥±‘ÌÑ¡”•…ÑÕÉ”µ…ÑÉ¥à½¹”…¹•Ù…±Õ…Ñ•Ì½¹±äUQ€ÌÀµµ¥¹ÕÑ”)‰½Õ¹‘…É¥•Ì¸Ğ•… ‰½Õ¹‘…Éä¥ĞÕÍ•ÌÑ¡”™¥Ù”µµ¥¹ÕÑ”…¹‘±”Ñ¡…Ğ¡…Ì©ÕÍĞ)±½Í•èÑ¡”Í½ÕÉ”…¹‘±”Ì±½Í•}ÁÉ¥•€¥Ì‰½Ñ Ñ¡”½‰Í•ÉÙ••¥Í¥½¸•Ù¥‘•¹”)…¹Ñ¡”¥¹ÁÕĞÑ¼Ñ¡”•á¥ÍÑ¥¹œA…Á•È•á•ÕÑ¥½¸…±Õ±…Ñ¥½¸¸9¼±…Ñ•È…¹‘±”½È)•…ÑÕÉ”É½Ü¥ÌÙ¥Í¥‰±”°ÁÉ•Ù•¹Ñ¥¹œ±½½¬µ…¡•…¸=Á•¸µÁ½Í¥Ñ¥½¸…¹Í•±°µ…±°•á•ÕÑ¥½¸ÁÉ¥•Ì°)™••Ì°Í±¥ÁÁ…”°Í¥é¥¹œ°…¹@™0É•ÕÍ”Ñ¡”Í…µ”ÁÕÉ”ÁÉ½‘ÕÑ¥½¸™Õ¹Ñ¥½¹Ì¸()…Í °Ñ¡”±½¹œµ½¹±äA½Í¥Ñ¥½¸°QÉ…‘•Ì°É•…±¥é•½Õ¹É•…±¥é•@™0°…¹Ñ¡”‰½Õ¹‘…Éä)•ÅÕ¥ÑäÕÉÙ”É•µ…¥¸¥µµÕÑ…‰±”¥¸µµ•µ½ÉäÉ•ÍÕ±ÑÌ¸Q¡”½µµ…¹±½Ì½¹±ä…¸)…É•…Ñ”ÍÕµµ…Éä…¹‘…Ñ…Í•Ğ™¥¹•ÉÁÉ¥¹Ğ¸%ĞÁ•É™½ÉµÌ¹¼İÉ¥Ñ•Ì°•áÑ•É¹…°)É•ÅÕ•ÍÑÌ°½È±½¬µ‘•Á•¹‘•¹Ğ…±Õ±…Ñ¥½¹Ì¥¹Í¥‘”Ñ¡”Í¥µÕ±…Ñ¥½¸¸()Y•ÉÍ¥½¸€ÄÍÕÁÁ½ÉÑÌ½¹”¥¹ÍÑÉÕµ•¹Ğ°½¹”€ÌÀµµ¥¹ÕÑ”…‘•¹”°…¹Ñ¡”•á¥ÍÑ¥¹œ)MÑÉ…Ñ•ä…Ñ¥½¹Ì€¡=A9}A=M%Q%=9€°M11}11€°…¹9=}Q%=9€¤¸±Ñ¡½Õ Ñ¡”)‘½µ…¥¸Ù½…‰Õ±…ÉäÉ•Í•ÉÙ•ÌÁ…ÉÑ¥…°Í•±±¥¹œ°‰…Í•±¥¹•}±½¹}½¹±å€Ù•ÉÍ¥½¸€Ä‘½•Ì)¹½Ğ•µ¥Ğ¥Ğ¸Q¡¥Ì¥Ì„‘•Ñ•Éµ¥¹¥ÍÑ¥ŒÍ¥µÕ±…Ñ½È°¹½Ğ‰…­Ñ•ÍÑ¥¹œ¥¹™É…ÍÑÉÕÑÕÉ”)™½ÈµÕ±Ñ¥Á±”MÑÉ…Ñ•¥•Ì…¹¹½Ğ…¸½ÁÑ¥µ¥é•È½ÈÁÉ½™¥Ñ…‰¥±¥ÑäÕ…É…¹Ñ•”¸((ŒŒŒ•Ñ•Éµ¥¹¥ÍÑ¥ŒÍÑÉ…Ñ•äÉ•™¥¹•µ•¹Ğ()IÕ¸Ñ¡”™¥ÉÍĞ‰½Õ¹‘•°É•…µ½¹±äÉ•™¥¹•È™½È‰…Í•±¥¹•}±½¹}½¹±å€è()Í )ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹É•™¥¹•}¡¥ÍÑ½É¥…±}ÍÑÉ…Ñ•äp(€€´µ™É½´€ÈÀÈĞ´ÀÄ´ÀÅPÀÀèÀÀèÀÁhp(€€´µÑ¼€ÈÀÈĞ´ÀĞ´ÀÅPÀÀèÀÀèÀÁhp(€€´µ…ÍÍ•Ğ	Q½UMPp(€€´µÁÉ½Ù¥‘•È‰¥¹…¹”p(€€´µµ…É­•ĞµÑåÁ”ÍÁ½Ğp(€€´µ¥¹Ñ•ÉÙ…°€Õ´p(€€´µ¥¹¥Ñ¥…°µ…Í €ÄÀÀÀÀp(€€´µ™•”µÉ…Ñ”€À¸ÀÀÄp(€€´µÍ±¥ÁÁ…”µÉ…Ñ”€À¸ÀÀÀÔp(€€´µÙ…±¥‘…Ñ¥½¸µÑ½Àµ¬€ÄÀp(€€´µÍ½É”µÍÉ••¹¥¹œµÑ½Àµ¬€ÄÈp(€€´µµ…àµÍ½É”µ½¹™¥ÕÉ…Ñ¥½¹Ì€ÈÀÀp(€€´µµ…àµ©½¥¹Ğµ…¹‘¥‘…Ñ•Ì€ÄÀÀÀ)€()Q¡”É•™¥¹•È½¹ÍÑÉÕÑÌÑ¡”¡¥ÍÑ½É¥…°•…ÑÕÉ”µ…ÑÉ¥à½¹”¸M½É”ÍÉ••¹¥¹œ™¥ÉÍĞ)Ñ•ÍÑÌÑ¡”€ÌÔ‘¥É•Ñ¥½¹…°½µ‰¥¹…Ñ¥½¹Ì½¸„€À¸ÈÕ€Í¥µÁ±•à™½ÈÑÉ•¹°IM$°)É•ÑÕÉ¸´Ä°…¹É•ÑÕÉ¸´ÄÈ¸… ¥Ì½µ‰¥¹•)İ¥Ñ Ù½±Õµ”µ½¹™¥Éµ…Ñ¥½¸İ•¥¡ÑÌ€Á€°€À¸ÄÈÕ€°€À¸ÈÕ€°€À¸ÌÜÕ€°…¹€À¸ÔÁ€°…¹)Ñ¡”•á…ĞÁÉ½‘ÕÑ¥½¸Í½É”İ•¥¡ÑÌ…É”¥¹±Õ‘•½¹”¸MÉ••¹¥¹œÕÍ•Ì½¹±äÑ¡”)™¥ÉÍĞ¡É½¹½±½¥…°¡…±˜½˜ÑÉ…¥¹¥¹œ…¹É•Ñ…¥¹Ì€ÄÈÍ½É”½¹™¥ÕÉ…Ñ¥½¹Ì‰ä)‘•™…Õ±Ğ¸())½¥¹ĞÉ•™¥¹•µ•¹Ğ½µ‰¥¹•ÌÉ•Ñ…¥¹•Í½É”½¹™¥ÕÉ…Ñ¥½¹Ìİ¥Ñ •¹ÑÉäÑ¡É•Í¡½±‘Ì)€À¸ÔÕ€°€À¸ØÁ€°€À¸ØÕ€°…¹€À¸ÜÁ€ì•á¥ĞÑ¡É•Í¡½±‘Ì€À¸ÄÁ€Ñ¡É½Õ €À¸ÌÁ€¥¸)€À¸ÀÕ€ÍÑ•ÁÌì…¹Ñ…É•Ğ…±±½…Ñ¥½¹Ì€ÄÁ€°€ÄÕ€°…¹€ÈÁ€Á•É•¹Ğ¸Q¡”ÁÉ½‘ÕÑ¥½¸)Í½É”İ•¥¡ÑÌ€À¸ÔÔ¼À¸ÄÔÜÔ¼À¸Àä¼À¸ÈÀÈÕ€İ¥Ñ Ù½±Õµ”€À¸ÈÕ€…¹MÑÉ…Ñ•ä)Á…É…µ•Ñ•ÉÌ€À¸ØÀ¼À¸ÈÀ¼ÄÔ•€É•µ…¥¸Õ¹¡…¹•…¹•áÁ±¥¥Ñ±ä¥‘•¹Ñ¥™¥•¸()•¥Í¥½¸‰½Õ¹‘…É¥•Ì…É”ÍÁ±¥Ğ¡É½¹½±½¥…±±ä¥¹Ñ¼€ØÀ”ÑÉ…¥¹¥¹œ°€ÈÀ”)Ù…±¥‘…Ñ¥½¸°…¹€ÈÀ”Õ¹Ñ½Õ¡•Ñ•ÍĞ•Ù¥‘•¹”¸… Í•µ•¹ĞÍÑ…ÉÑÌİ¥Ñ Ñ¡”Í…µ”)…Í …¹¹¼½Á•¸A½Í¥Ñ¥½¸¸•…ÑÕÉ•Ìµ…äÕÍ”•…É±¥•È…¹‘±•Ì™½Èİ…É´µÕÀ°İ¡¥±”)•… Í•µ•¹Ğ½¹Ñ…¥¹Ì½¹±ä¥ÑÌ½İ¸…±¥¹••¥Í¥½¸‰½Õ¹‘…É¥•Ì°ÁÉ•Ù•¹Ñ¥¹œ)™ÕÑÕÉ”•Ù¥‘•¹”…¹Ñ•ÍĞÉ•ÍÕ±ÑÌ™É½´¥¹™±Õ•¹¥¹œÍ•±•Ñ¥½¸¸()QÉ…¥¹¥¹œÉ…¹­Ì•Ù•Éä©½¥¹Ğ…¹‘¥‘…Ñ”‰ä¹•ĞÉ•ÑÕÉ¸°‘É…İ‘½İ¸°ÑÉ…‘”½Õ¹Ğ°…¹„)ÍÑ…‰±”%‘•É¥Ù•™É½´…±°Í½É”…¹MÑÉ…Ñ•äÁ…É…µ•Ñ•ÉÌ¸=¹±äÑ¡”½¹™¥ÕÉ•Ñ½À),É•…¡•ÌÙ…±¥‘…Ñ¥½¸¸É•½µµ•¹‘…Ñ¥½¸µÕÍĞ…±Í¼‰•…ĞÑ¡”ÁÉ½‘ÕÑ¥½¸‰…Í•±¥¹”)Ù…±¥‘…Ñ¥½¸É•ÑÕÉ¸…™Ñ•ÈÑ¡”ÍÕÁÁ±¥•™••Ì…¹Í±¥ÁÁ…”°­••À‘É…İ‘½İ¸¹¼É•…Ñ•È)Ñ¡…¸€ÄÀ”°…¹•á•ÕÑ”…Ğ±•…ÍĞ½¹”QÉ…‘”¸Q•ÍĞ•Ù…±Õ…Ñ•Ì½¹±äÑ¡”Í•±•Ñ•)…¹‘¥‘…Ñ”°ÁÉ½‘ÕÑ¥½¸‰…Í•±¥¹”°…¹„Í•Á…É…Ñ•±ä…±Õ±…Ñ•‰Õäµ…¹µ¡½±)‰•¹¡µ…É¬ì¥Ğ½¹™¥ÉµÌ½ÈÉ•©•ÑÌÑ¡”™¥á•É•½µµ•¹‘…Ñ¥½¸…¹¹•Ù•È¡…¹•Ì¥Ğ¸)	Õä…¹¡½±•¹Ñ•ÉÌ…¹•á¥ÑÌ…ĞÍ•µ•¹Ğ‰½Õ¹‘…É¥•Ìİ¥Ñ ¥‘•¹Ñ¥…°™••Ì…¹)Í±¥ÁÁ…”¸%ÑÌ¹•…ÈµÑ½Ñ…°•¹ÑÉäÉ•Í•ÉÙ•Ì„€À¸ÀÀÀÀÀÅ€…Í µÉ…Ñ”‰Õ™™•ÈÍ½±•±äÑ¼)…‰Í½Éˆ•¥µ…°ÅÕ…¹Ñ¥é…Ñ¥½¸•™™•ÑÌ°±•…Ù¥¹œ„µ¥¹¥µ…°…Í É•Í¥‘Õ”ìÑ¡”‰Õ™™•È)¥Ì¹½Ğ…¸…‘‘¥Ñ¥½¹…°™•”¸I•ÑÕÉ¸½Ù•È‘É…İ‘½İ¸¥ÌÉ•Á½ÉÑ•…Ìé•É¼İ¡•¸µ…á¥µÕ´‘É…İ‘½İ¸¥Ìé•É¼°)…Ù½¥‘¥¹œ…¸Õ¹‘•™¥¹•É…Ñ¥¼¸¥Ù”…‘‘¥Ñ¥½¹…°½¹ÑÉ½±±•½¹”µ…Ğµ„µÑ¥µ”…‰±…Ñ¥½¹Ì)ÕÍ”Ñ¡”ÁÉ½‘ÕÑ¥½¸MÑÉ…Ñ•ä½¸Ñ¡”Í…µ”ÍÉ••¹¥¹œÍ•µ•¹Ğ¸‘¥É•Ñ¥½¹…°Í¥¹…°)¥ÌÍ•ĞÑ¼é•É¼…¹¥ÑÌÁÉ½‘ÕÑ¥½¸İ•¥¡Ğ¥ÌÉ•‘¥ÍÑÉ¥‰ÕÑ•ÁÉ½Á½ÉÑ¥½¹…±±ä…É½ÍÌ)Ñ¡”½Ñ¡•ÈÑ¡É•”‘¥É•Ñ¥½¹…°Í¥¹…±ÌÕÍ¥¹œ•¥µ…°°€ÄàµÁ±…”I=U9}!1}Y8)ÅÕ…¹Ñ¥é…Ñ¥½¸°…¹„‘•Ñ•Éµ¥¹¥ÍÑ¥ŒÉ•Í¥‘Õ…°½¸Ñ¡”±…ÍĞ•¹…‰±•İ•¥¡ĞÍ¼Ñ¡•¥È)ÍÕ´É•µ…¥¹Ì•á…Ñ±ä½¹”¸Q¡”Ù½±Õµ”…‰±…Ñ¥½¸¡…¹•Ì½¹±ä¥ÑÌ½İ¸İ•¥¡Ğ¸Q¡•Í”)½µÁ…É¥Í½¹ÌÉ•Á½ÉĞ¡¥ÍÑ½É¥…°µ•ÑÉ¥Œ‘¥™™•É•¹•Ì…¹‘¼¹½Ğ‘•µ½¹ÍÑÉ…Ñ”)…ÕÍ…±¥Ñä¸()Q¡”‘•™…Õ±Ğ½µÁÕÑ…Ñ¥½¹…°‰Õ‘•ĞÉ•Ñ…¥¹Ì€ÄÈ½˜…Ğµ½ÍĞ€ÈÀÀ•¹•É…Ñ•Í½É”)½¹™¥ÕÉ…Ñ¥½¹Ì°•Ù…±Õ…Ñ•Ì…Ğµ½ÍĞ€Ä°ÀÀÀ©½¥¹Ğ…¹‘¥‘…Ñ•Ì°…¹Í•¹‘Ì€ÄÀÑ¼)Ù…±¥‘…Ñ¥½¸¸•¹•É…Ñ¥½¸™…¥±Ì±•…É±ä¥¹ÍÑ•…½˜Í¥±•¹Ñ±äÑÉÕ¹…Ñ¥¹œİ¡•¸„±¥µ¥Ğ)İ½Õ±‰”•á••‘•¸=Ù•ÉÉ¥‘”Ñ¡•Í”‰½Õ¹‘Ìİ¥Ñ €´µÍ½É”µÍÉ••¹¥¹œµÑ½Àµ­€°)€´µµ…àµÍ½É”µ½¹™¥ÕÉ…Ñ¥½¹Í€°€´µµ…àµ©½¥¹Ğµ…¹‘¥‘…Ñ•Í€°…¹)€´µÙ…±¥‘…Ñ¥½¸µÑ½Àµ­€¸()±°É•™¥¹•µ•¹Ğ¥Ì¥¸µ•µ½Éä…™Ñ•È½¹”¡¥ÍÑ½É¥…°É•…¸Q¡”½µµ…¹•µ¥ÑÌ½µÁ…Ğ)ÍÑÉÕÑÕÉ•µ•ÑÉ¥Ì…¹„‘•Ñ•Éµ¥¹¥ÍÑ¥Œ•áÁ±…¹…Ñ¥½¸°‰ÕĞÁ•ÉÍ¥ÍÑÌ¹¼•áÁ•É¥µ•¹Ğ°)•…ÑÕÉ”°M½É”°•¥Í¥½¸°QÉ…‘”°A½Í¥Ñ¥½¸°½ÈA…Á•ÈIÕ¸…¹¹•Ù•ÈÁÉ½µ½Ñ•Ì„)½¹™¥ÕÉ…Ñ¥½¸¸Y•ÉÍ¥½¸€Ä¥ÌÍ¥¹±”µ¥¹ÍÑÉÕµ•¹Ğ°±½¹œµ½¹±ä°Í•ÅÕ•¹Ñ¥…°°ÕÍ•Ì½¹”)™¥á•Í½É”µ½‘•°…¹„™¥¹¥Ñ”É¥°…¹µ…­•Ì¹¼±…¥´½˜™ÕÑÕÉ”ÁÉ½™¥Ñ…‰¥±¥Ñä¸)M½É”µµ½‘•°…¹MÑÉ…Ñ•äÁ…É…µ•Ñ•ÉÌÉ•µ…¥¸Í•Á…É…Ñ”Í¼™ÕÑÕÉ”…¹‘¥‘…Ñ•Ì…¸…‘)Ñ•¡¹¥…°°Í•¹Ñ¥µ•¹Ğ°¹•İÌ°½È50½µÁ½¹•¹Ğİ•¥¡ÑÌİ¥Ñ¡½ÕĞ½ÕÁ±¥¹œÑ¡½Í”)™ÕÑÕÉ”½µÁ½¹•¹ÑÌÑ¼MÑÉ…Ñ•äÑ¡É•Í¡½±‘Ìì¹½¹”½˜Ñ¡½Í”½µÁ½¹•¹ÑÌ¥Ì¥¹ÑÉ½‘Õ•)¡•É”¸()Q¡”½µµ…¹É•µ…¥¹ÌÉ•…µ½¹±ä‰ä‘•™…Õ±Ğ‰ÕĞµ…äÕÍ”½µÁ…Ñ¥‰±”½µÁ±•Ñ••áÁ•É¥µ•¹Ğ)¡¥ÍÑ½ÉäÑ¼‰Õ¥±„‘•Ñ•Éµ¥¹¥ÍÑ¥ŒÍ•…É Á±…¸¸‘€´µÁ•ÉÍ¥ÍÑ€Ñ¼…Ñ½µ¥…±±äÉ•Ñ…¥¸Ñ¡”)ÉÕ¸°…¹½¹¥…°…¹‘¥‘…Ñ•Ì°½µÁ…Ğ•Ù…±Õ…Ñ¥½¹Ì°‰…Í•±¥¹”°…¹‰Õäµ…¹µ¡½±•Ù¥‘•¹”¸)¸•á…Ğ½µÁ±•Ñ••áÁ•É¥µ•¹Ğ¥ÌÉ•ÕÍ•İ¥Ñ¡½ÕĞÍ¥µÕ±…Ñ¥½¸¸‘€´µ™½É”µÉ•½µÁÕÑ•€)İ¥Ñ €´µÁ•ÉÍ¥ÍÑ€Ñ¼½µÁ…É”„™É•Í ‘•Ñ•Éµ¥¹¥ÍÑ¥ŒÉ•ÍÕ±Ğ……¥¹ÍĞÍÑ½É••Ù¥‘•¹”İ¥Ñ¡½ÕĞ)É•…Ñ¥¹œ„‘ÕÁ±¥…Ñ”¸IÕ¸ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹É•™¥¹•µ•¹Ñ}ÍÑ½É…•€Ñ¼¥¹ÍÁ•Ğ•áÁ•É¥µ•¹Ğ°)…¹‘¥‘…Ñ”°•Ù…±Õ…Ñ¥½¸°…¹ÍÑ½É…”Ñ½Ñ…±Ì¸M•”)máÁ•É¥µ•¹Ğ5•µ½Éä…¹áÁ•É¥•¹”µÕ¥‘•M•…É¡t¡‘½Ì¼ÀÀàµ•áÁ•É¥µ•¹Ğµµ•µ½Éä¹µ¤™½È)½µÁ…Ñ¥‰¥±¥Ñä°±…ÍÍ¥™¥…Ñ¥½¹Ì°Á±…¹¹•È…±±½…Ñ¥½¸°¹•¥¡‰½É¡½½ÉÕ±•Ì°É•©•Ñ•)…¹‘¥‘…Ñ”É•½¹Í¥‘•É…Ñ¥½¸°ÑÉ…¹Í…Ñ¥½¸‰•¡…Ù¥½È°…¹±¥µ¥Ñ…Ñ¥½¹Ì¸()IÕ¸•¥Í¥½¸•¹•É…Ñ¥½¸…¹A…Á•È•á•ÕÑ¥½¸…Ì„Í•Á…É…Ñ”½¹”µÍ¡½ĞÁÉ½•ÍÌè()Í )ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹•¹•É…Ñ•}µ…É­•Ñ}‘•¥Í¥½¹Ì)€()… ¥¹ÍÑÉÕµ•¹Ğ¡…Ì…¸¥¹‘•Á•¹‘•¹Ñ±ä½µµ¥ÑÑ•å±”¸A½ÍÑÉ•ME0Õ¹¥ÅÕ•¹•ÍÌ)µ…­•Ì½µÁ±•Ñ•É•ÑÉ¥•Ì…¹½Ù•É±…ÁÁ¥¹œ¥¹Ù½…Ñ¥½¹Ì¹¼µ½ÁÌìÑ¡¥Ì¥Ì¹½Ğ…¸)•á…Ñ±äµ½¹”Í¡•‘Õ±¥¹œ±…¥´¸9=}Q%=9€¥Ì„½µÁ±•Ñ•…Õ‘¥Ñ…‰±”å±”İ¥Ñ¡½ÕĞ)„QÉ…‘”¸Q¡”©½ˆ½¹Ñ…ÑÌ¹¼ÁÉ½Ù¥‘•È½È•á¡…¹”¸()UÍ”€½¡•…±Ñ¡€™½ÈA$±¥Ù•¹•ÍÌ…¹€½¡•…±Ñ ½‘…Ñ…‰…Í•€™½ÈA½ÍÑÉ•ME0É•…‘¥¹•ÍÌ¸)9•¥Ñ¡•È•¹‘Á½¥¹Ğ½¹Ñ…ÑÌ	¥¹…¹”½È•áÁ½Í•Ì½¹¹•Ñ¥½¸‘•Ñ…¥±Ì¸((ŒŒŒAÉ½‘ÕÑ¥½¸Íµ½­”Ñ•ÍĞ()]¥Ñ Ñ¡”A$ÉÕ¹¹¥¹œ…¹•á…µÁ±”¡½ÍĞ…‘©ÕÍÑ•™½ÈÑ¡”‘•Á±½åµ•¹Ğè()Í )ÕÉ°¡ÑÑÀè¼½±½…±¡½ÍĞèàÀÀÀ½¡•…±Ñ )ÕÉ°¡ÑÑÀè¼½±½…±¡½ÍĞèàÀÀÀ½¡•…±Ñ ½‘…Ñ…‰…Í”)ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹¥¹•ÍÑ}±½Í•‘}…¹‘±•Ì)ÕÉ°¡ÑÑÀè¼½±½…±¡½ÍĞèàÀÀÀ½µ…É­•Ğ½ÍÑ…ÑÕÌ)ÕÉ°€‰¡ÑÑÀè¼½±½…±¡½ÍĞèàÀÀÀ½µ…É­•Ğ½…¹‘±•Ìı…ÍÍ•Ğõ	Q”ÉUMP™¥¹Ñ•ÉÙ…°ôÕ´ˆ)€((ŒŒŒ½­•È()™Ñ•È½¹™¥ÕÉ¥¹œ€¹•¹Ù€°‰Õ¥±…¹ÍÑ…ÉĞÑ¡”A$…¹A½ÍÑÉ•ME0è()Í )µ…­”‘½­•È)€()Q¡”A$ÍÑ…ÉÑÌ¥¹‘•Á•¹‘•¹Ñ±ä½˜A½ÍÑÉ•ME0É•…‘¥¹•ÍÌ¸%ÑÌ‘…Ñ…‰…Í”¡•…±Ñ •¹‘Á½¥¹ĞÉ•ÑÕÉ¹Ì!QQ@€ÔÀÌİ¡¥±”A½ÍÑÉ•ME0¥ÌÕ¹…Ù…¥±…‰±”¸((ŒŒŒA…Á•ÈQÉ…‘¥¹œ…‘µ¥¹¥ÍÑÉ…Ñ¥½¸()MÑ…ÉĞ½ÈÉ•Í•ĞÑ¡”…Ñ¥Ù”Í¥µÕ±…Ñ•…½Õ¹Ğİ¥Ñ¡½ÕĞ‘•±•Ñ¥¹œÁÉ¥½È•áÁ•É¥µ•¹ÑÌè()Í )ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹É•Í•Ñ}Á…Á•É}…½Õ¹Ğ€´µ¥¹¥Ñ¥…°µ…Í €ÄÀÀÀÀ€´µÉ•…Í½¸€‰‰…Í•±¥¹”‰•™½É”™•…ÑÕÉ”`ˆ)€()Q¡”½µµ…¹Á…ÉÍ•Ìµ½¹•ä…Ì•¥µ…±€°ÍÑ½ÁÌÑ¡”…Ñ¥Ù”•áÁ•É¥µ•¹Ğ°…¹É•…Ñ•Ì„™É•Í ±¥¹­•ÉÕ¸¥¸½¹”ÑÉ…¹Í…Ñ¥½¸¸MÑ½ÁÁ¥¹œ„ÉÕ¸‘½•Ì¹½Ğ±¥ÅÕ¥‘…Ñ”¥Ğè½Á•¸A½Í¥Ñ¥½¹ÌÉ•Ñ…¥¸Ñ¡•¥ÈÅÕ…¹Ñ¥Ñä…¹•¹ÑÉäÍÑ…Ñ”°…¹¹¼Íå¹Ñ¡•Ñ¥ŒQÉ…‘”¥ÌÉ•…Ñ•¸!¥ÍÑ½É¥…°ÉÕ¹Ì…É”ÅÕ•Éå…‰±”Ñ¡É½Õ P€½Á…Á•È½ÉÕ¹Í€…¹P€½Á…Á•È½ÉÕ¹Ì½íÉÕ¹}¥‘õ€ìÁ½Í¥Ñ¥½¹Ì…¹ÑÉ…‘•Ì…•ÁĞ…¸½ÁÑ¥½¹…°ÉÕ¹}¥‘€¸I•Í•ĞÉ•µ…¥¹Ì1$µ½¹±äÕ¹Ñ¥°…ÕÑ¡•¹Ñ¥…Ñ•µÕÑ…Ñ¥½¸•¹‘Á½¥¹ÑÌ•á¥ÍĞ°…¹„™ÕÑÕÉ”…ÕÑ¡•¹Ñ¥…Ñ•™É½¹Ñ•¹…Ñ¥½¸µÕÍĞÉ•ÕÍ”Ñ¡¥ÌÕÍ”…Í”¸((ŒŒŒ]••­±äMÑÉ…Ñ•äÉ•Í•…É ()]••­±äÉ•Í•…É ¥¹±Õ‘•Ì„‰½Õ¹‘•ÁÉ½É•ÍÍ¥Ù”½µÁ½Í¥Ñ”µ™•…ÑÕÉ”Í•…É …™Ñ•ÈÑ…É•Ñ•)É•™¥¹•µ•¹Ğ¸QÉ…¥¹¥¹œµ½¹±ä…ÍÍ½¥…Ñ¥½¸‘•Ñ•Éµ¥¹•ÌÍ¥¹•İ•¥¡ÑÌ°…¹Í•Á…É…Ñ”•½¹½µ¥Œ)…¹•áÁ±½É…Ñ½Éä‰•…µÌÉ½ÜÑ¡É½Õ É•…°Á…É•¹Ğ±¥¹•…”…É½ÍÌ‘•ÁÑ¡Ì€ËŠLÜ¸áÁ±½É…Ñ½Éä)Á…É•¹ÑÌµ…äÁÉ½‘Õ”‘•Í•¹‘…¹ÑÌ‰ÕĞ…¹¹½Ğ•¹Ñ•ÈÙ…±¥‘…Ñ¥½¸½ÈÑ•ÍĞÑ¡•µÍ•±Ù•Ì¸… )İ…±¬µ™½Éİ…Éİ¥¹‘½Ü™É••é•Ì¥ÑÌ½İ¸)ÑÉ…¥¹¥¹œµ•‘¥…¸½%EH¸I•Ñ…¥¹•½¹™¥ÕÉ…Ñ¥½¹ÌÕÍ”Ñ¡”¡¥ÍÑ½É¥…°A…Á•ÈQÉ…‘¥¹œÍ¥µÕ±…Ñ½È°)Í¼™••Ì°Í±¥ÁÁ…”°…‘•¹”°Ñ¡É•Í¡½±‘Ì°¡½±‘¥¹œÁ•É¥½‘Ì°½½±‘½İ¸°…Í …¹Á½Í¥Ñ¥½¹Ì…É”)Á…ÉĞ½˜Ù…±¥‘…Ñ¥½¸…¹¥Í½±…Ñ•µÑ•ÍĞ•Ù¥‘•¹”¸Q¡”ÍÑ…”¹•Ù•È…Ñ¥Ù…Ñ•ÌMÑÉ…Ñ•¥•Ì½È)¡…¹•ÌA…Á•ÈQÉ…‘¥¹œ¸M•”)m‘½Ì¼ÀÄĞµİ••­±äµÍÑÉ…Ñ•äµÉ•Í•…É ¹µ‘t¡‘½Ì¼ÀÄĞµİ••­±äµÍÑÉ…Ñ•äµÉ•Í•…É ¹µ¤¸()Q¡”½…ÉÍ”µ…ÑÉ¥à½Ù•ÉÌ€ÌÁ´¼Å ¼É ¼Ñ ¼Ù ¼á ¼ÄÉ …‘•¹•Ì…¹(ÌÁ´¼Å ¼É ¼Ñ ¼á ¼ÄÉ ¼ÈÑ ¡½É¥é½¹Ìİ¥Ñ¡½ÕĞ„…ÉÑ•Í¥…¸ÁÉ½‘ÕĞ¸I•µ…¥¹¥¹œÑÉ…¥¹¥¹œÍ±½ÑÌ)•Ù…±Õ…Ñ”‰½Õ¹‘•±½…°½Á•É…Ñ¥½¹…°¹•¥¡‰½ÉÌìÙ…±¥‘…Ñ¥½¸…¹Ñ•ÍĞ¹•Ù•È¡½½Í”½ÈÉ•ÑÕ¹”)Ñ¡½Í”½¹™¥ÕÉ…Ñ¥½¹Ì¸()½µÁ½Í¥Ñ”ÍÕµµ…É¥•Ì•áÁ½Í”Í•Á…É…Ñ”¡•…À°ÑÉ…¥¹¥¹œ°Ù…±¥‘…Ñ¥½¸…¹Ñ•ÍĞ½Õ¹Ñ•ÉÌ°É•…°)É½ÜµÍ±¥”½¹½Éµ…±¥é…Ñ¥½¸½Ñ¡É•Í¡½±µ™¥ĞÉ•ÕÍ”°…¡•‰•¹¡µ…É¬İ½É¬°½…ÉÍ”…‘•¹”½¡½É¥é½¸)½Ù•É…”°±½…°µÉ•™¥¹•µ•¹Ğ±¥¹•…”°ÍÑ½ÀÉ•…Í½¹Ì°‰•ÍĞ•½¹½µ¥Œ½•áÁ±½É…Ñ½Éä•Ù¥‘•¹”°…¹„)‘¥…¹½ÍÑ¥Œ‰Õ‘•Ñ}•á¡…ÕÍÑ•‘€™±…œ¸…¹‘¥‘…Ñ”±¥ÍÑ¥¹Ì…¸‰”™¥±Ñ•É•‰ä)Í•±•Ñ¥½¹}ÑÉ…¬õ•½¹½µ¥ñ•áÁ±½É…Ñ½Éå€¸Á…ÍÍ•‘}Ù…±¥‘…Ñ¥½¹€µ•…¹Ì…±°Ù…±¥‘…Ñ¥½¸…Ñ•ÌÁ…ÍÍ•)‰ÕĞÑ¡”‰½Õ¹‘•¥Í½±…Ñ•µÑ•ÍĞ…±±½İ…¹”İ…Ì…±É•…‘ä½¹ÍÕµ•ì¥Ğ¥Ì¹½Ğ„É•©•Ñ¥½¸¸()I•…µ½¹±ä½µÁ½Í¥Ñ”•Ù¥‘•¹”¥Ì…Ù…¥±…‰±”…Ğè()Ñ•áĞ)P€½½ÁÑ¥µ¥é…Ñ¥½¸½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½½µÁ½Í¥Ñ”µÍ•…É )P€½½ÁÑ¥µ¥é…Ñ¥½¸½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½½µÁ½Í¥Ñ”µÍ•…É ½…¹‘¥‘…Ñ•Ì)P€½½ÁÑ¥µ¥é…Ñ¥½¸½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½½µÁ½Í¥Ñ”µÍ•…É ½…¹‘¥‘…Ñ•Ì½í…¹‘¥‘…Ñ•}¥‘ô)€()IÕ¸‰½Õ¹‘•É•Í•…É ™½È•Ù•ÉäÍ½É¥¹œµ•¹…‰±•…Ñ…±½œ¥¹ÍÑÉÕµ•¹Ğè()Í )ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹ÉÕ¹}İ••­±å}ÍÑÉ…Ñ•å}É•Í•…É )€()Q¡”‘•™…Õ±ĞÉ…¹”¥ÌÑ¡”±…Ñ•ÍĞ€ÜÌÀ½µÁ±•Ñ”‘…åÌ¸UÍ”€´µ‘ÉäµÉÕ¹€Ñ¼Ù…±¥‘…Ñ”Ñ¡”)…Ñ…±½œ°É…¹”°…¹‰Õ‘•Ğİ¥Ñ¡½ÕĞ±½…‘¥¹œÑ¡”™Õ±°‘…Ñ…Í•Ğ½ÈÁ•ÉÍ¥ÍÑ¥¹œ•Ù¥‘•¹”¸)½µÁ±•Ñ•¥‘•¹Ñ¥…°•Ù¥‘•¹”¥ÌÉ•½¹ÍÑÉÕÑ•‰•™½É”…¹‘±•Ì½ÈÑ¡”•…ÑÕÉ”µ…ÑÉ¥à…É”)±½…‘•¸Q¡”©½ˆÁÉ½•ÍÍ•Ì¥¹ÍÑÉÕµ•¹ÑÌÍ•ÅÕ•¹Ñ¥…±±ä°‘½•Ì¹½Ğ…Ñ¥Ù…Ñ”…¹‘¥‘…Ñ•Ì°…¹)‘½•Ì¹½ĞµÕÑ…Ñ”A…Á•ÈQÉ…‘¥¹œ¸I…¥±İ…äÍ•ÑÕÀ°…¹‘¥‘…Ñ”™…µ¥±¥•Ì°½ÕÑ½µ•Ì°…¹É•ÕÍ”)Í•µ…¹Ñ¥Ì…É”‘½Õµ•¹Ñ•¥¸)m‘½Ì¼ÀÄĞµİ••­±äµÍÑÉ…Ñ•äµÉ•Í•…É ¹µ‘t¡‘½Ì¼ÀÄĞµİ••­±äµÍÑÉ…Ñ•äµÉ•Í•…É ¹µ¤¸)Q¡…ĞÉÕ¹‰½½¬…±Í¼‘½Õµ•¹ÑÌÑ¡”‰½Õ¹‘•Ñ…É•Ñ•µÉ•™¥¹•µ•¹ĞÍÑ…”™½È•½¹½µ¥…±±ä)ÁÉ½µ¥Í¥¹œØĞ¸Ä¹•…Èµ¥ÍÍ•Ì…¹¥ÑÌÉ•…µ½¹±ä•Ù¥‘•¹”•¹‘Á½¥¹ÑÌ¸((ŒŒ50µ½‘•°É•™¥¹•µ•¹Ğ()½µÁ±•Ñ•µ½‘•°µÍÉ••¹¥¹œÉ•ÍÕ±ÑÌ…¸‰”É•™¥¹•Ñ¡É½Õ „Ù•ÉÍ¥½¹•)M=8½¹ÑÉ…Ğİ¥Ñ¡½ÕĞ¡…¹¥¹œ…ÁÁ±¥…Ñ¥½¸½‘”¸Q¡”•¹¥¹”ÍÕÁÁ½ÉÑÌI¥‘”°1¥¡Ñ	4°…¹…Ñ	½½ÍĞìÉ•É•ÍÍ¥½¸°‘¥É•Ñ¥½¸°…¹½ÍĞµ…‘©ÕÍÑ•…Ñ¥½¸½‰©•Ñ¥Ù•Ìì™½±µ±½…°™•…ÑÕÉ”Í•±•Ñ¥½¸ìÑÉ…¥¹¥¹œµ™É½é•¸½¹™¥‘•¹”‰Õ­•ÑÌì‘•Ñ•Éµ¥¹¥ÍÑ¥ŒÑÉ¥…°½ÉÕ¹Ñ¥µ”‰Õ‘•ÑÌì…¹Á•ÉÍ¥ÍÑ•É•ÕÍ”¸()UÍ”Ñ¡”‘½Õµ•¹Ñ•Á±…•¡½±‘•È½¹™¥ÕÉ…Ñ¥½¸…Ì„ÍÑ…ÉÑ¥¹œÁ½¥¹Ğè()‰…Í )ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹É•™¥¹•}µ±}ÍÉ••¹¥¹}É•ÍÕ±Ğ€´µ½¹™¥œ‘½Ì½•á…µÁ±•Ì½µ°µÉ•™¥¹•µ•¹Ğµ…Ñ‰½½ÍĞ´ÌÁ´¹©Í½¸€´µ‘ÉäµÉÕ¸)ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹É•™¥¹•}µ±}ÍÉ••¹¥¹}É•ÍÕ±Ğ€´µ½¹™¥œ‘½Ì½•á…µÁ±•Ì½µ°µÉ•™¥¹•µ•¹Ğµ…Ñ‰½½ÍĞ´ÌÁ´¹©Í½¸)€()I•Á±…”¥ÑÌÑİ¼Á±…•¡½±‘•ÈUU%Ìİ¥Ñ „½µÁ±•Ñ•ÍÉ••¹¥¹œÉÕ¸½É•ÍÕ±ĞÁ…¥È¸I•™¥¹•µ•¹Ğ™¥ÉÍĞÉ•½¹ÍÑÉÕÑÌÑ¡”±½Í•€ÈÀÈÀ´ÈÀÈØÍÉ••¹¥¹œÉ…¹”½¹±äÑ¼Ù•É¥™äÁ•ÉÍ¥ÍÑ•¥‘•¹Ñ¥Ñä°É•±•…Í•Ì¥ÑÌ™•…ÑÕÉ”µ…ÑÉ¥à°…¹Ñ¡•¸‰Õ¥±‘Ì„Í•Á…É…Ñ”‘•Ù•±½Áµ•¹Ğµ½¹±ä‘…Ñ…Í•Ğ•¹‘¥¹œ…Ğ€ÈÀÈĞ™½È•Ù•Éä…±Õ±…Ñ¥½¸¸5…Ñ¡¥¹œ™¥¹•ÉÁÉ¥¹Ğ¥‘•¹Ñ¥Ñä¹•Ù•È…ÕÑ¡½É¥é•Ì™•…ÑÕÉ”Í•±•Ñ¥½¸°™¥ÑÑ¥¹œ°Ñ…É•ÑÌ°µ•ÑÉ¥Ì°½È±…ÍÍ¥™¥…Ñ¥½¸Ñ¼½¹ÍÕµ”ÁÉ½Ñ•Ñ•€ÈÀÈĞ´ÈÀÈØ™¥¹…°µÑ•ÍĞ•Ù¥‘•¹”¸M•”m50µ½‘•°ÍÉ••¹¥¹œ…¹É•™¥¹•µ•¹Ñt¡‘½Ì½µ°µµ½‘•°µÍÉ••¹¥¹œ¹µ¤™½È½¹ÑÉ…ÑÌ°±…ÍÍ¥™¥…Ñ¥½¹Ì°É•ÕÍ”°…¹É•…µ½¹±äA$•¹‘Á½¥¹ÑÌ¸()ÕÑ¡•¹Ñ¥…Ñ•±¥•¹ÑÌ…¸‘¥Í½Ù•ÈÑ¡”•á…ĞÉ•ÅÕ•ÍĞ½¹ÑÉ…Ğİ¥Ñ P€½½ÁÑ¥µ¥é…Ñ¥½¸½µ°µÉ•™¥¹•µ•¹Ğµ½¹ÑÉ…Ñ€…¹•¹ÅÕ•Õ”¥Ğİ¥Ñ A=MP€½½ÁÑ¥µ¥é…Ñ¥½¸½µ°µÉ•™¥¹•µ•¹ĞµÉÕ¹Í€¸Q¡”A$½¹±äÙ…±¥‘…Ñ•Ì…¹Á•ÉÍ¥ÍÑÌÅÕ•Õ•İ½É¬ì¥Ğ¹•Ù•È±½…‘ÌÑ¡”¡¥ÍÑ½É¥…°‘…Ñ…Í•Ğ½ÈÑÉ…¥¹Ìµ½‘•±Ì¸IÕ¸Ñ¡”‘•‘¥…Ñ•I…¥±İ…äÉ•Í•…É İ½É­•È…Ì½¹”É•Á±¥„¥¹¥Ñ¥…±±äè()‰…Í )ÁåÑ¡½¸€µ´…ÁÀ¹©½‰Ì¹ÁÉ½•ÍÍ}µ±}É•™¥¹•µ•¹Ñ}ÅÕ•Õ”)€()•Á±½äµ¥É…Ñ¥½¹Ì…¹Ñ¡”A$‰•™½É”ÍÑ…ÉÑ¥¹œ½ÈÉ•‘•Á±½å¥¹œÑ¡”İ½É­•È¸EÕ•Õ”±…¥µÌ…É”‘ÕÉ…‰±”…¹½¹ÕÉÉ•¹äµÍ…™”¥¸A½ÍÑÉ•ME0ì…¹•±±…Ñ¥½¸…¹É•ÑÉäÕÍ”Ñ¡”ÉÕ¸•¹‘Á½¥¹ÑÌ‘½Õµ•¹Ñ•¥¸m50µ½‘•°ÍÉ••¹¥¹œ…¹É•™¥¹•µ•¹Ñt¡‘½Ì½µ°µµ½‘•°µÍÉ••¹¥¹œ¹µ¤¸é•É¼µÁÉ½É•ÍÌ…Ñ…Í•Ñ¥¹•ÉÁÉ¥¹Ñ¡…¹•‘€ÉÕ¸µ…ä‰”É•ÑÉ¥••áÁ±¥¥Ñ±ä…™Ñ•È„É•½¹ÍÑÉÕÑ¥½¸™¥àèÉ•ÑÉäÁÉ•Í•ÉÙ•Ì¥ÑÌÉÕ¸%…¹¥‘•¹Ñ¥Ñä¸%˜Á¥¹¹•¡¥ÍÑ½É¥…°…¹‘±•ÌÑÉÕ±ä¡…¹•°¥Ğ™…¥±Ì‰•™½É”•Ù…±Õ…Ñ¥½¸……¥¸…¹É•ÅÕ¥É•Ì„¹•ÜÍÉ••¹¥¹œÉÕ¸¸((ŒŒŒá¡…ÕÍÑ¥Ù”=!1XÍ•…É …µÁ…¥¸()Q¡”Ù•ÉÍ¥½¹••á¡…ÕÍÑ¥Ù•}½¡±Ù}ØÅ€…µÁ…¥¸½½É‘¥¹…Ñ•ÌÑ¡”•á¥ÍÑ¥¹œÉ•™¥¹•µ•¹Ğ)ÅÕ•Õ”…É½ÍÌ•Ù•Éä‘•±…É••¹¥¹”°™ÕÑÕÉ”‘ÕÉ…Ñ¥½¸°½‰©•Ñ¥Ù”°…¹ÍÕÁÁ½ÉÑ•™•…ÑÕÉ”)Ù…É¥…¹Ğ¸%ĞÕ…É…¹Ñ••ÌÍÑ…”µ½¹”‰É•…‘Ñ ‰•™½É”ÕµÕ±…Ñ¥Ù”€Ğ¼ÄØ¼ØĞµÑÉ¥…°‘••Á•¹¥¹œ°)ÕÍ•Ì½¹Í•ÉÙ…Ñ¥Ù”½¹™¥ÕÉ•½ÍÑÌ°…¹Á•ÉÍ¥ÍÑÌ…Õ‘¥Ñ…‰±”•½¹½µ¥ŒÁÉ½µ½Ñ¥½¸½È)‘¥Í…ÉÉ•…Í½¹Ì¸%Ğ¹•Ù•È½¹ÍÕµ•ÌÑ¡”ÁÉ½Ñ•Ñ•€ÈÀÈĞ´ÈÀÈØ™¥¹…°µÑ•ÍĞÁ•É¥½¸()M•”má¡…ÕÍÑ¥Ù”=!1X50M•…É …µÁ…¥¹t¡‘½Ì½µ°µÍ•…É µ…µÁ…¥¸¹µ¤™½ÈA$°1$°)I…¥±İ…ä‘•Á±½åµ•¹Ğ°µ½¹¥Ñ½É¥¹œ°…¹•±±…Ñ¥½¸°½ÕÑ½µ•Ì°…¹É½±±‰…¬¸(