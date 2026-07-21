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
- [Experiment memory and experience-guided search](docs/008-experiment-memory.md)
- [Architecture decisions](docs/decisions/)

## Current phase

Argos now includes a production-ready API foundation, persisted market-data and
Paper Trading workflows, historical simulation and refinement tooling, and
read-only decision-support APIs. It remains private and research-oriented: it
does not trade real money or guarantee profitability.

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
per-instrument results keyed by asset, provider, market type, and interval.
Missing snapshots or benchmark candles return HTTP 200 with typed unavailable
reasons; unknown Runs retain the existing HTTP 404 behavior. See
[`docs/007-paper-strategy-evaluation.md`](docs/007-paper-strategy-evaluation.md)
for the complete metric and benchmark contract.

Performance is derived only from immutable equity snapshots. Its implicit first
point is the run's initial cash at `started_at`. Equity is cash plus Position
market value, and total return compares the latest equity with that initial
capital. Current drawdown is the non-negative decline from the preceding running
peak at the latest snapshot; maximum drawdown is the worst such decline observed
in the run. These metrics are not annualized, predictive, or a guarantee of
profitability. A run without snapshots returns a neutral baseline. Assets,
including the currently configured BTC and ETH instruments, do not participate
in the formulas.

The MVP reads the complete aggregate snapshot history because it produces only
two points per hour. A database aggregation or daily summaries can be introduced
if volume later justifies them.

## Historical optimization memory API

Persisted refinement evidence is exposed through read-only endpoints:

- `GET /optimization/refinement-runs`
- `GET /optimization/refinement-runs/{run_id}`
- `GET /optimization/refinement-runs/{run_id}/evaluations`
- `GET /optimization/candidates`
- `GET /optimization/candidates/{stable_candidate_id}`
- `GET /optimization/refinement-runs/{run_id}/walk-forward`
- `GET /optimization/walk-forward-runs/{walk_forward_run_id}`
- `GET /optimization/walk-forward-runs/{walk_forward_run_id}/candidates`
- `GET /optimization/walk-forward-runs/{walk_forward_run_id}/selections`
- `GET /optimization/walk-forward-runs/{walk_forward_run_id}/candidates/{candidate_id}/windows`

Run-list filters include `asset`, `provider`, `market_type`, `interval`, `status`,
`recommendation_status`, `started_from`, and `started_to`. Candidate-list filters
include the complete instrument identity, `evidence_classification`, Score and
Strategy model names and versions, and `minimum_validation_runs`.

Run summaries use deterministic `limit`/`offset` pagination. Candidate pages, run
evaluations, and candidate evaluation history use stable keyset cursors and return
`has_more` plus `next_cursor`; every page has a maximum size of 200. Candidate filtering
scans bounded batches until the requested public page is complete, without a hidden
global candidate cap. Filters use the complete asset, provider, market type, and interval identity.
Responses preserve Decimal values and UTC evidence timestamps, expose explicit public
models rather than PostgreSQL JSONB payloads, and never execute a refinement or contact
an external provider.

Examples:

```sh
curl "http://localhost:8000/optimization/refinement-runs?asset=BTC%2FUSDT&provider=binance&market_type=spot&interval=5m&status=completed&limit=25&offset=0"
curl "http://localhost:8000/optimization/candidates?asset=BTC%2FUSDT&provider=binance&market_type=spot&interval=5m&minimum_validation_runs=2&limit=25"
curl "http://localhost:8000/optimization/refinement-runs/<run-id>/evaluations?limit=100"
curl "http://localhost:8000/optimization/candidates/<stable-candidate-id>?limit=100"
```

See [Experiment memory and experience-guided search](docs/008-experiment-memory.md)
for the evidence-compatibility and planning contracts behind these read models.

An optional persisted walk-forward pass evaluates the production baseline,
buy-and-hold, and a bounded set of refined candidates over multiple chronological
out-of-sample test windows:

```sh
python -m app.jobs.refine_historical_strategy \
  --asset BTC/USDT --provider binance --market-type spot --interval 5m \
  --from 2024-01-01T00:00:00Z --to 2026-01-01T00:00:00Z \
  --initial-cash 10000 --fee-rate 0.001 --slippage-rate 0.0005 \
  --persist --walk-forward \
  --walk-forward-training-days 180 \
  --walk-forward-validation-days 60 \
  --walk-forward-test-days 60 \
  --walk-forward-step-days 60 \
  --walk-forward-max-windows 6 \
  --walk-forward-max-candidates 10
```

The version-2 windows are UTC and half-open. Each window performs bounded candidate
generation and training screening, ranks candidates only on that window's validation
segment, and evaluates only its selected candidates on the isolated test segment.
Global refinement validation/test evidence and later windows cannot influence an
earlier selection. The most recent configured number of valid windows is retained,
with ordinals normalized from one; per-window selection ranks, reasons, metrics, and
fingerprints are persisted and available through the selections endpoint. Identical
completed configurations are reused. Walk-forward
classifications are decision support only: they do not modify production Strategy
configuration or active Paper Trading.

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

### Historical market-candle backfill

Import only persisted market candles for an aligned UTC range. Bounds are
`[from, to)`: `from` is inclusive and `to` is exclusive. This command never
calculates Features or Scores and never creates Decisions, Trades, Positions,
Snapshots, or Paper Runs.

```sh
python -m app.jobs.backfill_market_candles \
  --from 2024-01-01T00:00:00Z \
  --to 2024-04-01T00:00:00Z
```

Without identity filters, the job uses every ingestion-enabled catalog
instrument compatible with the existing Binance Spot adapter. Optional
`--asset`, `--provider`, `--market-type`, and `--interval` filters narrow the
selection. `--page-size` defaults to Binance's 1,000-candle maximum,
`--max-pages` supports controlled canaries, and `--dry-run` validates and reads
provider pages without persistence.

Run a canary first, then import quarter by quarter:

```sh
python -m app.jobs.backfill_market_candles \
  --from 2024-01-01T00:00:00Z \
  --to 2024-01-08T00:00:00Z \
  --max-pages 5
```

Each page commits independently and natural candle identity makes reruns
idempotent. After an interruption, rerun the same range: completed pages remain
stored and duplicates are ignored. Instrument summaries report expected slots,
received/inserted/duplicate candles, gaps and their first/last timestamps,
pages, retries, and processed bounds. PostgreSQL summaries report row and byte
growth plus an approximate added-byte cost per new candle; other database
dialects explicitly omit storage telemetry.

Review available PostgreSQL storage before each quarter and inspect missing
slots before continuing. A gap can reflect unavailable provider evidence and
must not be silently filled or inferred.

### Historical optimization dataset

Build and validate one deterministic, read-only dataset for future automatic
optimization:

```sh
python -m app.jobs.validate_historical_dataset \
  --from 2024-01-01T00:00:00Z \
  --to 2025-01-01T00:00:00Z \
  --asset BTC/USDT \
  --provider binance \
  --market-type spot \
  --interval 5m
```

The reader uses bounded keyset pages ordered by candle `open_time`; it never
uses `OFFSET` or writes to PostgreSQL. Bounds are aligned UTC `[from, to)`.
Every expected interval slot must have exactly one candle: any initial,
internal, or final gap invalidates the dataset rather than being filled,
interpolated, or hidden.

The command calculates the current seven-feature matrix once in memory using
the same pure indicator functions as operational ingestion. Warm-up candles
without a complete Feature set are omitted. It does not persist derived Feature
rows, calculate Scores, evaluate Strategies or profitability, or simulate
Trades. Process one instrument at a time so its dataset can be released before
the next instrument is loaded.

The reported SHA-256 fingerprint covers the complete instrument identity,
range, Feature model/version, and canonical ordered OHLCV evidence. Identical
evidence produces the same fingerprint; any candle change produces a different
one. The current roughly 268,000 five-minute candles per asset are read in
5,000-row pages. In-memory usage remains linear in that one instrument's range;
exact memory depends on Python and Decimal object overhead, so production runs
should observe container memory before increasing the selected range.

### Deterministic historical strategy simulation

Simulate `baseline_technical` version 1 and `baseline_long_only` version 1 for
one validated historical instrument without modifying operational or Paper
Trading state:

```sh
python -m app.jobs.simulate_historical_strategy \
  --from 2024-01-01T00:00:00Z \
  --to 2024-04-01T00:00:00Z \
  --asset BTC/USDT \
  --provider binance \
  --market-type spot \
  --interval 5m \
  --initial-cash 10000 \
  --fee-rate 0.001 \
  --slippage-rate 0.0005
```

The engine builds the Feature matrix once and evaluates only UTC 30-minute
boundaries. At each boundary it uses the five-minute candle that has just
closed: the source candle's `close_price` is both the observed Decision evidence
and the input to the existing Paper execution calculation. No later candle or
Feature row is visible, preventing look-ahead. BUY and SELL execution prices,
fees, slippage, sizing, and P&L reuse the same pure production functions.

Cash, the long-only Position, Trades, realized/unrealized P&L, and the boundary
equity curve remain immutable in-memory results. The command logs only an
aggregate summary and dataset fingerprint. It performs no writes, external
requests, or clock-dependent calculations inside the simulation.

Version 1 supports one instrument, one 30-minute cadence, and the existing
Strategy actions (`OPEN_POSITION`, `SELL_ALL`, and `NO_ACTION`). Although the
domain vocabulary reserves partial selling, `baseline_long_only` version 1 does
not emit it. This is a deterministic simulator, not backtesting infrastructure
for multiple Strategies and not an optimizer or profitability guarantee.

### Deterministic strategy refinement

Run the first bounded, read-only refiner for `baseline_long_only`:

```sh
python -m app.jobs.refine_historical_strategy \
  --from 2024-01-01T00:00:00Z \
  --to 2024-04-01T00:00:00Z \
  --asset BTC/USDT \
  --provider binance \
  --market-type spot \
  --interval 5m \
  --initial-cash 10000 \
  --fee-rate 0.001 \
  --slippage-rate 0.0005 \
  --validation-top-k 10 \
  --score-screening-top-k 12 \
  --max-score-configurations 200 \
  --max-joint-candidates 1000
```

The refiner constructs the historical Feature matrix once. Score screening first
tests the 35 directional combinations on a `0.25` simplex for trend, RSI,
return-1, and return-12. Each is combined
with volume-confirmation weights `0`, `0.125`, `0.25`, `0.375`, and `0.50`, and
the exact production score weights are included once. Screening uses only the
first chronological half of training and retains 12 score configurations by
default.

Joint refinement combines retained score configurations with entry thresholds
`0.55`, `0.60`, `0.65`, and `0.70`; exit thresholds `0.10` through `0.30` in
`0.05` steps; and target allocations `10`, `15`, and `20` percent. The production
score weights `0.55/0.1575/0.09/0.2025` with volume `0.25` and Strategy
parameters `0.60/0.20/15%` remain unchanged and explicitly identified.

Decision boundaries are split chronologically into 60% training, 20%
validation, and 20% untouched test evidence. Each segment starts with the same
cash and no open Position. Features may use earlier candles for warm-up, while
each segment contains only its own aligned Decision boundaries, preventing
future evidence and test results from influencing selection.

Training ranks every joint candidate by net return, drawdown, trade count, and a
stable ID derived from all score and Strategy parameters. Only the configured top
K reaches validation. A recommendation must also beat the production baseline
validation return after the supplied fees and slippage, keep drawdown no greater
than 10%, and execute at least one Trade. Test evaluates only the selected
candidate, production baseline, and a separately calculated buy-and-hold
benchmark; it confirms or rejects the fixed recommendation and never changes it.
Buy and hold enters and exits at segment boundaries with identical fees and
slippage. Its near-total entry reserves a `0.000001` cash-rate buffer solely to
absorb Decimal quantization effects, leaving a minimal cash residue; the buffer
is not an additional fee. Return over drawdown is reported as zero when maximum drawdown is zero,
avoiding an undefined ratio. Five additional controlled one-at-a-time ablations
use the production Strategy on the same screening segment. A directional signal
is set to zero and its production weight is redistributed proportionally across
the other three directional signals using Decimal, 18-place ROUND_HALF_EVEN
quantization, and a deterministic residual on the last enabled weight so their
sum remains exactly one. The volume ablation changes only its own weight. These
comparisons report historical metric differences and do not demonstrate
causality.

The default computational budget retains 12 of at most 200 generated score
configurations, evaluates at most 1,000 joint candidates, and sends 10 to
validation. Generation fails clearly instead of silently truncating when a limit
would be exceeded. Override these bounds with `--score-screening-top-k`,
`--max-score-configurations`, `--max-joint-candidates`, and
`--validation-top-k`.

All refinement is in memory after one historical read. The command emits compact
structured metrics and a deterministic explanation, but persists no experiment,
Feature, Score, Decision, Trade, Position, or Paper Run and never promotes a
configuration. Version 1 is single-instrument, long-only, sequential, uses one
fixed score model and a finite grid, and makes no claim of future profitability.
Score-model and Strategy parameters remain separate so future candidates can add
technical, sentiment, news, or ML component weights without coupling those
future components to Strategy thresholds; none of those components is introduced
here.

The command remains read-only by default but may use compatible completed experiment
history to build a deterministic search plan. Add `--persist` to atomically retain the
run, canonical candidates, compact evaluations, baseline, and buy-and-hold evidence.
An exact completed experiment is reused without simulation. Add `--force-recompute`
with `--persist` to compare a fresh deterministic result against stored evidence without
creating a duplicate. Run `python -m app.jobs.refinement_storage` to inspect experiment,
candidate, evaluation, and storage totals. See
[Experiment Memory and Experience-Guided Search](docs/008-experiment-memory.md) for
compatibility, classifications, planner allocation, neighborhood rules, rejected
candidate reconsideration, transaction behavior, and limitations.

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
