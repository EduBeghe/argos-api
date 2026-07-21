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
`equal_weight_buy_and_hold_ëÛh‘éì¶»§q«^t™X\ÛÛœÎÈ[šÛ›İÛˆ[œÈ™]Z[ˆH^\İ[™È™Z]š[Ü‹ˆÙYB–ØØÜËÌË\\\‹\İ˜]YŞKY]˜[X][Û‹›YJØÜËÌË\\\‹\İ˜]YŞKY]˜[X][Û‹›Y
B™›ÜˆHÛÛ\]HY]šXÈ[™™[˜ÚX\šÈÛÛ˜Xİ‚‚”\™›Ü›X[˜ÙH\È\š]™YÛ›Hœ›ÛH[[]]X›H\]Z]HÛ˜\ÚİËˆ]È[\XÚ]š\œİœÚ[\ÈH[‰ÜÈ[š]X[Ø\Ú]İ\YØ]ˆ\]Z]H\ÈØ\Ú\ÈÜÚ][Û‚›X\šÙ]˜[YK[™İ[™]\›ˆÛÛ\\™\ÈH]\İ\]Z]HÚ]][š]X[˜Ø\][ˆİ\œ™[˜]ÙİÛˆ\ÈH›Û‹[™YØ]]™HXÛ[™Hœ›ÛHH™XÙY[™È[›š[™ÂœXZÈ]H]\İÛ˜\ÚİÈX^[][H˜]ÙİÛˆ\ÈHÛÜœİİXÚXÛ[™HØœÙ\™Yš[ˆH[‹ˆ\ÙHY]šXÜÈ\™H›İ[›X[^™Y™YXİ]™KÜˆHİX\˜[YHÙ‚œ›Ùš]Xš[]KˆH[ˆÚ]İ]Û˜\ÚİÈ™]\›œÈH™]]˜[˜\Ù[[™Kˆ\ÜÙ]Ëš[˜ÛY[™ÈHİ\œ™[HÛÛ™šYİ\™Y•È[™U[œİ[Y[ËÈ›İ\XÚ\]Bš[ˆH›Ü›][\Ë‚‚•HU”™XYÈHÛÛ\]HYÙÜ™YØ]HÛ˜\Úİ\İÜH™XØ]\ÙH]›ÙXÙ\ÈÛ›BÛÈÚ[È\ˆİ\‹ˆH]X˜\ÙHYÙÜ™YØ][ÛˆÜˆZ[Hİ[[X\šY\ÈØ[ˆ™H[›ÙXÙYšYˆ›Û[YH]\ˆ\İYšY\È[K‚‚ˆÈÈ\İÜšXØ[Ü[Z^˜][ÛˆY[[ÜHTB‚”\œÚ\İY™Yš[™[Y[]šY[˜ÙH\È^ÜÙY›İYÚ™XY[Û›H[™Ú[Î‚‚‹HÑUÛÜ[Z^˜][Û‹Ü™Yš[™[Y[\[œØ‹HÑUÛÜ[Z^˜][Û‹Ü™Yš[™[Y[\[œËŞÜ[—ÚYX‹HÑUÛÜ[Z^˜][Û‹Ü™Yš[™[Y[\[œËŞÜ[—ÚYKÙ]˜[X][ÛœØ‹HÑUÛÜ[Z^˜][Û‹ØØ[™Y]\Ø‹HÑUÛÜ[Z^˜][Û‹ØØ[™Y]\ËŞÜİX›WØØ[™Y]WÚYX‚”[ˆİ[[X\šY\È\ÙH]\›Z[š\İXÈ[Z]ØÙ™œÙ]YÚ[˜][Û‹ˆØ[™Y]HYÙ\Ë[‚™]˜[X][ÛœË[™Ø[™Y]H]˜[X][Ûˆ\İÜH\ÙHİX›HÙ^\Ù]İ\œÛÜœÈ[™™]\›‚˜\×Û[Ü™X\È™^Øİ\œÛÜ˜È]™\HYÙH\ÈHX^[][HÚ^™HÙˆŒˆØ[™Y]Hš[\š[™ÂœØØ[œÈ›İ[™Y˜]Ú\È[[H™\]Y\İYX›XÈYÙH\ÈÛÛ\]KÚ]İ]HY[‚™ÛØ˜[Ø[™Y]HØ\ˆš[\œÈ\ÙHHÛÛ\]H\ÜÙ]›İšY\‹X\šÙ]\K[™[\˜[Y[]K‚”™\ÜÛœÙ\È™\Ù\™HXÚ[X[˜[Y\È[™UÈ]šY[˜ÙH[Y\İ[\Ë^ÜÙH^XÚ]X›XÂ›[Ù[È˜]\ˆ[ˆÜİÜ™TÔS”ÓÓˆ^[ØYË[™™]™\ˆ^Xİ]HH™Yš[™[Y[ÜˆÛÛXİ˜[ˆ^\›˜[›İšY\‹‚‚ˆÈÈ›ÙXİ[Ûˆ\Ş[Y[‚”›İšYHUPTÑWÕT“[™™]šY]ÈH›ÙXİ[Ûˆ˜[Y\È›Üˆ’SSÑWĞTÑWÕT“˜’SSÑWĞÓÓ“‘PÕÕSQSÕUÔÑPÓÓ‘Ø’SSÑWÔ‘PQÕSQSÕUÔÑPÓÓ‘Ø˜’SSÑWÒS‘ÑTÕSÓ—ÓSRUÑ×ÓU‘S[™ÓÔ”×ĞSÕÑQÓÔ’QÒS”ØˆHÓÔ”ÂœÙ][™È\ÈHÛÛ[XK\Ù\\˜]Y\İÙˆ^XİÜšYÚ[œÎÈÚ[Ø\™È[™˜Ü™Y[X[YÜšYÚ[œÈ\™H™Z™XİYˆ]™[ÜY[[Û›HÔÕÔ‘T×Ê˜˜\šXX›\È\™B››İ™YYYÚ[ˆUPTÑWÕT“\Èİ\YY‚‚”[ˆHTH›ØÙ\ÜÈÚ]‚‚˜ÚœØÜš\ËÜİ\Ø\KœÚ˜‚•HØÜš\\Y\È[[XšXÈ\Ü˜YHXY™Y›Ü™Hİ\[™ÈÛ™H]šXÛÜ›ˆÛÜšÙ\ˆÛ‚˜ŒŒŒ‰ÔÔ•‹NXˆHZYÜ˜][Ûˆ˜Z[\™H™]™[ÈHÙ\™\ˆœ›ÛHİ\[™Ë‚‘\Ş[Y[ÛÛ™šYİ\˜][Ûˆ]\İ]›ÚY[›š[™ÈHØ[YHZYÜ˜][ÛˆÛÛ˜İ\œ™[H[‚›][\HTH[œİ[˜Ù\Ë‚‚”[ˆ[™Ù\İ[Ûˆ\ÈHÙ\\˜]HÛ™K\Úİ›ØÙ\ÜÎ‚‚˜Úœ]Ûˆ[H\š›ØœËš[™Ù\İØÛÜÙYØØ[™\Â˜‚•H^\›˜[]›Ü›HÚİ[[›ÚÙH]Û˜ÙH]™\Hš]™HZ[]\ËÚÜHY\ˆB™š]™K[Z[]HUÈ›İ[™\KˆHİ™\›\Ú[™İÈ[™ÜİÜ™TÔS˜]\˜[[š\]Y[™\ÜÂ›XZÙH™\X]YÜˆİ™\›\[™È[›ØØ][ÛœÈY[\İ[È\È\È›İ[ˆ^XİK[Û˜ÙBœØÚY[[™ÈİX\˜[YKˆ\™ÛÜÈÛÛZ[œÈ›È[\›˜[ØÚY[\ˆÜˆ[™š[š]HÛÜ‚‚ˆÈÈÈ\İÜšXØ[X\šÙ]XØ[™H˜XÚÙš[‚’[\ÜÛ›H\œÚ\İYX\šÙ]Ø[™\È›Üˆ[ˆ[YÛ™YUÈ˜[™ÙKˆ›İ[™È\™B˜Ùœ›ÛKÊXˆœ›ÛX\È[˜Û\Ú]™H[™Ø\È^Û\Ú]™Kˆ\ÈÛÛ[X[™™]™\‚˜Ø[İ[]\È™X]\™\ÈÜˆØÛÜ™\È[™™]™\ˆÜ™X]\ÈXÚ\Ú[ÛœË˜Y\ËÜÚ][ÛœË”Û˜\ÚİËÜˆ\\ˆ[œË‚‚˜Úœ]Ûˆ[H\š›ØœË˜˜XÚÙš[ÛX\šÙ]ØØ[™\ÈˆKYœ›ÛHŒLKLUŒŒˆˆK]ÈŒLLUŒŒ‚˜‚•Ú]İ]Y[]Hš[\œËH›Øˆ\Ù\È]™\H[™Ù\İ[Û‹Y[˜X›YØ][ÙÂš[œİ[Y[ÛÛ\]X›HÚ]H^\İ[™Èš[˜[˜ÙHÜİY\\‹ˆÜ[Û˜[˜KX\ÜÙ]K\›İšY\˜K[X\šÙ]]\X[™KZ[\˜[š[\œÈ˜\œ›İÈBœÙ[Xİ[Û‹ˆK\YÙK\Ú^™XY˜][ÈÈš[˜[˜ÙIÜÈKXØ[™HX^[][K˜K[X^\YÙ\Øİ\ÜÈÛÛ›ÛYØ[˜\šY\Ë[™KYK\[˜˜[Y]\È[™™XYÂœ›İšY\ˆYÙ\ÈÚ]İ]\œÚ\İ[˜ÙK‚‚”[ˆHØ[˜\Hš\œİ[ˆ[\Ü]X\\ˆH]X\\‚‚˜Úœ]Ûˆ[H\š›ØœË˜˜XÚÙš[ÛX\šÙ]ØØ[™\ÈˆKYœ›ÛHŒLKLUŒŒˆˆK]ÈŒLKLŒŒˆˆK[X^\YÙ\ÈB˜‚‘XXÚYÙHÛÛ[Z]È[™\[™[H[™˜]\˜[Ø[™HY[]HXZÙ\È™\[œÂšY[\İ[ˆY\ˆ[ˆ[\œ\[Û‹™\[ˆHØ[YH˜[™ÙNˆÛÛ\]YYÙ\È™[XZ[‚œİÜ™Y[™\XØ]\È\™HYÛ›Ü™Yˆ[œİ[Y[İ[[X\šY\È™\Ü^XİYÛİËœ™XÙZ]™YÚ[œÙ\YÙ\XØ]HØ[™\ËØ\È[™Z\ˆš\œİÛ\İ[Y\İ[\ËœYÙ\Ë™]šY\Ë[™›ØÙ\ÜÙY›İ[™ËˆÜİÜ™TÔSİ[[X\šY\È™\Ü›İÈ[™]B™Ü›İİ\È[ˆ\›Ş[X]HYYX]HÛÜİ\ˆ™]ÈØ[™NÈİ\ˆ]X˜\ÙB™X[XİÈ^XÚ]HÛZ]İÜ˜YÙH[[Y]K‚‚”™]šY]È]˜Z[X›HÜİÜ™TÔSİÜ˜YÙH™Y›Ü™HXXÚ]X\\ˆ[™[œÜXİZ\ÜÚ[™ÂœÛİÈ™Y›Ü™HÛÛ[Z[™ËˆHØ\Ø[ˆ™Y›Xİ[˜]˜Z[X›H›İšY\ˆ]šY[˜ÙH[™›]\İ›İ™HÚ[[Hš[YÜˆ[™™\œ™Y‚‚ˆÈÈÈ\İÜšXØ[Ü[Z^˜][Ûˆ]\Ù]‚Z[[™˜[Y]HÛ™H]\›Z[š\İXË™XY[Û›H]\Ù]›Üˆ]\™H]]ÛX]XÂ›Ü[Z^˜][Û‚‚˜Úœ]Ûˆ[H\š›ØœË˜[Y]WÚ\İÜšXØ[Ù]\Ù]ˆKYœ›ÛHŒLKLUŒŒˆˆK]ÈŒKLKLUŒŒˆˆKX\ÜÙ]•ËÕTÑˆK\›İšY\ˆš[˜[˜ÙHˆK[X\šÙ]]\HÜİˆKZ[\˜[[B˜‚•H™XY\ˆ\Ù\È›İ[™YÙ^\Ù]YÙ\ÈÜ™\™YHØ[™HÜ[—İ[YXÈ]™]™\‚\Ù\ÈÑ‘”ÑUÜˆÜš]\ÈÈÜİÜ™TÔSˆ›İ[™È\™H[YÛ™YUÈÙœ›ÛKÊX‚‘]™\H^XİY[\˜[Ûİ]\İ]™H^XİHÛ™HØ[™Nˆ[H[š]X[š[\›˜[Üˆš[˜[Ø\[˜[Y]\ÈH]\Ù]˜]\ˆ[ˆ™Z[™Èš[Yš[\œÛ]YÜˆY[‹‚‚•HÛÛ[X[™Ø[İ[]\ÈHİ\œ™[Ù]™[‹Y™X]\™HX]š^Û˜ÙH[ˆY[[ÜH\Ú[™ÂHØ[YH\™H[™XØ]Üˆ[˜İ[ÛœÈ\ÈÜ\˜][Û˜[[™Ù\İ[Û‹ˆØ\›K]\Ø[™\ÂÚ]İ]HÛÛ\]H™X]\™HÙ]\™HÛZ]Yˆ]Ù\È›İ\œÚ\İ\š]™Y™X]\™Bœ›İÜËØ[İ[]HØÛÜ™\Ë]˜[X]Hİ˜]YÚY\ÈÜˆ›Ùš]Xš[]KÜˆÚ[][]B•˜Y\Ëˆ›ØÙ\ÜÈÛ™H[œİ[Y[]H[YHÛÈ]È]\Ù]Ø[ˆ™H™[X\ÙY™Y›Ü™BH™^[œİ[Y[\ÈØYY‚‚•H™\ÜYÒKLMˆš[™Ù\œš[Ûİ™\œÈHÛÛ\]H[œİ[Y[Y[]Kœ˜[™ÙK™X]\™H[Ù[İ™\œÚ[Û‹[™Ø[›ÛšXØ[Ü™\™YÒÕˆ]šY[˜ÙKˆY[XØ[™]šY[˜ÙH›ÙXÙ\ÈHØ[YHš[™Ù\œš[È[HØ[™HÚ[™ÙH›ÙXÙ\ÈHY™™\™[›Û™KˆHİ\œ™[›İYÚHš]™K[Z[]HØ[™\È\ˆ\ÜÙ]\™H™XY[‚K\›İÈYÙ\Ëˆ[‹[Y[[ÜH\ØYÙH™[XZ[œÈ[™X\ˆ[ˆ]Û™H[œİ[Y[	ÜÈ˜[™ÙNÂ™^XİY[[ÜH\[™ÈÛˆ]Ûˆ[™XÚ[X[Øš™Xİİ™\šXYÛÈ›ÙXİ[Ûˆ[œÂœÚİ[ØœÙ\™HÛÛZ[™\ˆY[[ÜH™Y›Ü™H[˜Ü™X\Ú[™ÈHÙ[XİY˜[™ÙK‚‚ˆÈÈÈ]\›Z[š\İXÈ\İÜšXØ[İ˜]YŞHÚ[][][Û‚‚”Ú[][]H˜\Ù[[™WİXÚšXØ[™\œÚ[ÛˆH[™˜\Ù[[™WÛÛ™×ÛÛ›X™\œÚ[ÛˆH›Ü‚›Û™H˜[Y]Y\İÜšXØ[[œİ[Y[Ú]İ][ÙYZ[™ÈÜ\˜][Û˜[Üˆ\\‚•˜Y[™Èİ]N‚‚˜Úœ]Ûˆ[H\š›ØœËœÚ[][]WÚ\İÜšXØ[Üİ˜]YŞHˆKYœ›ÛHŒLKLUŒŒˆˆK]ÈŒLLUŒŒˆˆKX\ÜÙ]•ËÕTÑˆK\›İšY\ˆš[˜[˜ÙHˆK[X\šÙ]]\HÜİˆKZ[\˜[[HˆKZ[š]X[XØ\ÚLˆKY™YK\˜]HŒHˆK\Û\YÙK\˜]HŒB˜‚•H[™Ú[™HZ[ÈH™X]\™HX]š^Û˜ÙH[™]˜[X]\ÈÛ›HUÈÌ[Z[]B˜›İ[™\šY\Ëˆ]XXÚ›İ[™\H]\Ù\ÈHš]™K[Z[]HØ[™H]\È\İ˜ÛÜÙYˆHÛİ\˜ÙHØ[™IÜÈÛÜÙWÜšXÙX\È›İHØœÙ\™YXÚ\Ú[Ûˆ]šY[˜ÙB˜[™H[œ]ÈH^\İ[™È\\ˆ^Xİ][ÛˆØ[İ[][Û‹ˆ›È]\ˆØ[™HÜ‚‘™X]\™H›İÈ\Èš\ÚX›K™]™[[™ÈÛÚËXZXYˆ•VH[™ÑS^Xİ][ÛˆšXÙ\Ë™™Y\ËÛ\YÙKÚ^š[™Ë[™	“™]\ÙHHØ[YH\™H›ÙXİ[Ûˆ[˜İ[ÛœË‚‚Ø\ÚHÛ™Ë[Û›HÜÚ][Û‹˜Y\Ë™X[^™Yİ[œ™X[^™Y	“[™H›İ[™\B™\]Z]Hİ\™H™[XZ[ˆ[[]]X›H[‹[Y[[ÜH™\İ[ËˆHÛÛ[X[™ÙÜÈÛ›H[‚˜YÙÜ™YØ]Hİ[[X\H[™]\Ù]š[™Ù\œš[ˆ]\™›Ü›\È›ÈÜš]\Ë^\›˜[œ™\]Y\İËÜˆÛØÚËY\[™[Ø[İ[][ÛœÈ[œÚYHHÚ[][][Û‹‚‚•™\œÚ[ÛˆHİ\ÜÈÛ™H[œİ[Y[Û™HÌ[Z[]HØY[˜ÙK[™H^\İ[™Â”İ˜]YŞHXİ[ÛœÈ
ÔS—ÔÔÒUSÓ˜ÑSĞS[™“×ĞPÕSÓ˜
Kˆ[İYÚB™ÛXZ[ˆ›ØØX[\H™\Ù\™\È\X[Ù[[™Ë˜\Ù[[™WÛÛ™×ÛÛ›X™\œÚ[ÛˆHÙ\Â››İ[Z]]ˆ\È\ÈH]\›Z[š\İXÈÚ[][]Ü‹›İ˜XÚİ\İ[™È[™œ˜\İXİ\™B™›Üˆ][\Hİ˜]YÚY\È[™›İ[ˆÜ[Z^™\ˆÜˆ›Ùš]Xš[]HİX\˜[YK‚‚ˆÈÈÈ]\›Z[š\İXÈİ˜]YŞH™Yš[™[Y[‚”[ˆHš\œİ›İ[™Y™XY[Û›H™Yš[™\ˆ›Üˆ˜\Ù[[™WÛÛ™×ÛÛ›X‚‚˜Úœ]Ûˆ[H\š›ØœËœ™Yš[™WÚ\İÜšXØ[Üİ˜]YŞHˆKYœ›ÛHŒLKLUŒŒˆˆK]ÈŒLLUŒŒˆˆKX\ÜÙ]•ËÕTÑˆK\›İšY\ˆš[˜[˜ÙHˆK[X\šÙ]]\HÜİˆKZ[\˜[[HˆKZ[š]X[XØ\ÚLˆKY™YK\˜]HŒHˆK\Û\YÙK\˜]HŒHˆK]˜[Y][Û‹]ÜZÈLˆK\ØÛÜ™K\ØÜ™Y[š[™Ë]ÜZÈLˆˆK[X^\ØÛÜ™KXÛÛ™šYİ\˜][ÛœÈŒˆK[X^Z›Ú[XØ[™Y]\ÈL˜‚•H™Yš[™\ˆÛÛœİXİÈH\İÜšXØ[™X]\™HX]š^Û˜ÙKˆØÛÜ™HØÜ™Y[š[™Èš\œİ\İÈHÍH\™Xİ[Û˜[ÛÛXš[˜][ÛœÈÛˆHŒXÚ[\^›Üˆ™[™”ÒKœ™]\›‹LK[™™]\›‹LL‹ˆXXÚ\ÈÛÛXš[™YÚ]›Û[YKXÛÛ™š\›X][ÛˆÙZYÚÈŒLXŒXŒÍÍX[™L[™H^Xİ›ÙXİ[ÛˆØÛÜ™HÙZYÚÈ\™H[˜ÛYYÛ˜ÙKˆØÜ™Y[š[™È\Ù\ÈÛ›HB™š\œİÚ›Û›ÛÙÚXØ[[ˆÙˆ˜Z[š[™È[™™]Z[œÈLˆØÛÜ™HÛÛ™šYİ\˜][ÛœÈB™Y˜][‚‚’›Ú[™Yš[™[Y[ÛÛXš[™\È™]Z[™YØÛÜ™HÛÛ™šYİ\˜][ÛœÈÚ][H™\ÚÛÂ˜MXŒX[™ÌÈ^]™\ÚÛÈŒL›İYÚŒÌ[‚˜ŒXİ\ÎÈ[™\™Ù][ØØ][ÛœÈLMX[™Œ\˜Ù[ˆH›ÙXİ[Û‚œØÛÜ™HÙZYÚÈMKÌŒMMÍKÌŒKÌŒŒXÚ]›Û[YHŒX[™İ˜]YŞBœ\˜[Y]\œÈŒÌŒŒÌMIX™[XZ[ˆ[˜Ú[™ÙY[™^XÚ]HY[YšYY‚‚‘XÚ\Ú[Ûˆ›İ[™\šY\È\™HÜ]Ú›Û›ÛÙÚXØ[H[ÈŒ	H˜Z[š[™ËŒ	B˜[Y][Û‹[™Œ	H[İXÚY\İ]šY[˜ÙKˆXXÚÙYÛY[İ\ÈÚ]HØ[YB˜Ø\Ú[™›ÈÜ[ˆÜÚ][Û‹ˆ™X]\™\ÈX^H\ÙHX\›Y\ˆØ[™\È›ÜˆØ\›K]\Ú[B™XXÚÙYÛY[ÛÛZ[œÈÛ›H]ÈİÛˆ[YÛ™YXÚ\Ú[Ûˆ›İ[™\šY\Ë™]™[[™Â™]\™H]šY[˜ÙH[™\İ™\İ[Èœ›ÛH[™›Y[˜Ú[™ÈÙ[Xİ[Û‹‚‚•˜Z[š[™È˜[šÜÈ]™\H›Ú[Ø[™Y]HH™]™]\›‹˜]ÙİÛ‹˜YHÛİ[[™BœİX›HQ\š]™Yœ›ÛH[ØÛÜ™H[™İ˜]YŞH\˜[Y]\œËˆÛ›HHÛÛ™šYİ\™YÜ’È™XXÚ\È˜[Y][Û‹ˆH™XÛÛ[Y[™][Ûˆ]\İ[ÛÈ™X]H›ÙXİ[Ûˆ˜\Ù[[™B˜[Y][Ûˆ™]\›ˆY\ˆHİ\YY™Y\È[™Û\YÙKÙY\˜]ÙİÛˆ›ÈÜ™X]\‚[ˆL	K[™^Xİ]H]X\İÛ™H˜YKˆ\İ]˜[X]\ÈÛ›HHÙ[XİY˜Ø[™Y]K›ÙXİ[Ûˆ˜\Ù[[™K[™HÙ\\˜][HØ[İ[]Y^KX[™ZÛ˜™[˜ÚX\šÎÈ]ÛÛ™š\›\ÈÜˆ™Z™XİÈHš^Y™XÛÛ[Y[™][Ûˆ[™™]™\ˆÚ[™Ù\È]‚^H[™Û[\œÈ[™^]È]ÙYÛY[›İ[™\šY\ÈÚ]Y[XØ[™Y\È[™œÛ\YÙKˆ]È™X\‹]İ[[H™\Ù\™\ÈHŒXØ\Ú\˜]HY™™\ˆÛÛ[HÂ˜XœÛÜ˜ˆXÚ[X[]X[^˜][ÛˆY™™XİËX]š[™ÈHZ[š[X[Ø\Ú™\ÚYYNÈHY™™\‚š\È›İ[ˆY][Û˜[™YKˆ™]\›ˆİ™\ˆ˜]ÙİÛˆ\È™\ÜY\È™\›ÈÚ[ˆX^[][H˜]ÙİÛˆ\È™\›Ë˜]›ÚY[™È[ˆ[™Yš[™Y˜][Ëˆš]™HY][Û˜[ÛÛ›ÛYÛ™KX]XK][YHX›][ÛœÂ\ÙHH›ÙXİ[Ûˆİ˜]YŞHÛˆHØ[YHØÜ™Y[š[™ÈÙYÛY[ˆH\™Xİ[Û˜[ÚYÛ˜[š\ÈÙ]È™\›È[™]È›ÙXİ[ÛˆÙZYÚ\È™Y\İšX]Y›ÜÜ[Û˜[HXÜ›ÜÜÂHİ\ˆ™YH\™Xİ[Û˜[ÚYÛ˜[È\Ú[™ÈXÚ[X[N\XÙH“ÕS‘ÒS—ÑU‘S‚œ]X[^˜][Û‹[™H]\›Z[š\İXÈ™\ÚYX[ÛˆH\İ[˜X›YÙZYÚÛÈZ\‚œİ[H™[XZ[œÈ^XİHÛ™KˆH›Û[YHX›][ÛˆÚ[™Ù\ÈÛ›H]ÈİÛˆÙZYÚˆ\ÙB˜ÛÛ\\š\ÛÛœÈ™\Ü\İÜšXØ[Y]šXÈY™™\™[˜Ù\È[™È›İ[[Ûœİ˜]B˜Ø]\Ø[]K‚‚•HY˜][ÛÛ\]][Û˜[YÙ]™]Z[œÈLˆÙˆ][ÜİŒÙ[™\˜]YØÛÜ™B˜ÛÛ™šYİ\˜][ÛœË]˜[X]\È][ÜİK›Ú[Ø[™Y]\Ë[™Ù[™ÈLÂ˜[Y][Û‹ˆÙ[™\˜][Ûˆ˜Z[ÈÛX\›H[œİXYÙˆÚ[[H[˜Ø][™ÈÚ[ˆH[Z]Ûİ[™H^ÙYYYˆİ™\œšYH\ÙH›İ[™ÈÚ]K\ØÛÜ™K\ØÜ™Y[š[™Ë]ÜZØ˜K[X^\ØÛÜ™KXÛÛ™šYİ\˜][ÛœØK[X^Z›Ú[XØ[™Y]\Ø[™˜K]˜[Y][Û‹]ÜZØ‚‚[™Yš[™[Y[\È[ˆY[[ÜHY\ˆÛ™H\İÜšXØ[™XYˆHÛÛ[X[™[Z]ÈÛÛ\XİœİXİ\™YY]šXÜÈ[™H]\›Z[š\İXÈ^[˜][Û‹]\œÚ\İÈ›È^\š[Y[‘™X]\™KØÛÜ™KXÚ\Ú[Û‹˜YKÜÚ][Û‹Üˆ\\ˆ[ˆ[™™]™\ˆ›Û[İ\ÈB˜ÛÛ™šYİ\˜][Û‹ˆ™\œÚ[ÛˆH\ÈÚ[™ÛKZ[œİ[Y[Û™Ë[Û›KÙ\]Y[X[\Ù\ÈÛ™B™š^YØÛÜ™H[Ù[[™Hš[š]HÜšY[™XZÙ\È›ÈÛZ[HÙˆ]\™H›Ùš]Xš[]K‚”ØÛÜ™K[[Ù[[™İ˜]YŞH\˜[Y]\œÈ™[XZ[ˆÙ\\˜]HÛÈ]\™HØ[™Y]\ÈØ[ˆYXÚšXØ[Ù[[Y[™]ÜËÜˆSÛÛ\Û™[ÙZYÚÈÚ]İ]Ûİ\[™ÈÜÙB™]\™HÛÛ\Û™[ÈÈİ˜]YŞH™\ÚÛÎÈ›Û™HÙˆÜÙHÛÛ\Û™[È\È[›ÙXÙYš\™K‚‚•HÛÛ[X[™™[XZ[œÈ™XY[Û›HHY˜][]X^H\ÙHÛÛ\]X›HÛÛ\]Y^\š[Y[š\İÜHÈZ[H]\›Z[š\İXÈÙX\˜Ú[‹ˆYK\\œÚ\İÈ]ÛZXØ[H™]Z[ˆBœ[‹Ø[›ÛšXØ[Ø[™Y]\ËÛÛ\Xİ]˜[X][ÛœË˜\Ù[[™K[™^KX[™ZÛ]šY[˜ÙK‚[ˆ^XİÛÛ\]Y^\š[Y[\È™]\ÙYÚ]İ]Ú[][][Û‹ˆYKY›Ü˜ÙK\™XÛÛ\]XÚ]K\\œÚ\İÈÛÛ\\™HHœ™\Ú]\›Z[š\İXÈ™\İ[YØZ[œİİÜ™Y]šY[˜ÙHÚ]İ]˜Ü™X][™ÈH\XØ]Kˆ[ˆ]Ûˆ[H\š›ØœËœ™Yš[™[Y[ÜİÜ˜YÙXÈ[œÜXİ^\š[Y[˜Ø[™Y]K]˜[X][Û‹[™İÜ˜YÙHİ[ËˆÙYB–Ñ^\š[Y[Y[[ÜH[™^\šY[˜ÙKQİZYYÙX\˜ÚJØÜËÌY^\š[Y[[Y[[ÜK›Y
H›Ü‚˜ÛÛ\]Xš[]KÛ\ÜÚYšXØ][ÛœË[›™\ˆ[ØØ][Û‹™ZYÚ›ÜšÛÙ[\Ë™Z™XİY˜Ø[™Y]H™XÛÛœÚY\˜][Û‹˜[œØXİ[Ûˆ™Z]š[Ü‹[™[Z]][ÛœË‚‚”[ˆXÚ\Ú[ÛˆÙ[™\˜][Ûˆ[™\\ˆ^Xİ][Ûˆ\ÈHÙ\\˜]HÛ™K\Úİ›ØÙ\ÜÎ‚‚˜Úœ]Ûˆ[H\š›ØœË™Ù[™\˜]WÛX\šÙ]ÙXÚ\Ú[ÛœÂ˜‚‘XXÚ[œİ[Y[\È[ˆ[™\[™[HÛÛ[Z]YŞXÛKˆÜİÜ™TÔS[š\]Y[™\ÜÂ›XZÙ\ÈÛÛ\]Y™]šY\È[™İ™\›\[™È[›ØØ][ÛœÈ›Ë[ÜÎÈ\È\È›İ[‚™^XİK[Û˜ÙHØÚY[[™ÈÛZ[Kˆ“×ĞPÕSÓ˜\ÈHÛÛ\]Y]Y]X›HŞXÛHÚ]İ]˜H˜YKˆH›ØˆÛÛXİÈ›È›İšY\ˆÜˆ^Ú[™ÙK‚‚•\ÙHÚX[›ÜˆTH]™[™\ÜÈ[™ÚX[Ù]X˜\ÙX›ÜˆÜİÜ™TÔS™XY[™\ÜË‚“™Z]\ˆ[™Ú[ÛÛXİÈš[˜[˜ÙHÜˆ^ÜÙ\ÈÛÛ›™Xİ[Ûˆ]Z[Ë‚‚ˆÈÈÈ›ÙXİ[ÛˆÛ[ÚÙH\İ‚•Ú]HTH[›š[™È[™^[\HÜİY\İY›ÜˆH\Ş[Y[‚‚˜Ú˜İ\›‹ËÛØØ[ÜİÚX[˜İ\›‹ËÛØØ[ÜİÚX[Ù]X˜\ÙBœ]Ûˆ[H\š›ØœËš[™Ù\İØÛÜÙYØØ[™\Â˜İ\›‹ËÛØØ[ÜİÛX\šÙ]Üİ]\Â˜İ\›š‹ËÛØØ[ÜİÛX\šÙ]ØØ[™\ÏØ\ÜÙ]P•ÉL‘•TÑ	š[\˜[M[H‚˜‚ˆÈÈÈØÚÙ\‚‚Y\ˆÛÛ™šYİ\š[™È™[˜Z[[™İ\HTH[™ÜİÜ™TÔS‚‚˜Ú›XZÙHØÚÙ\‚˜‚•HTHİ\È[™\[™[HÙˆÜİÜ™TÔS™XY[™\ÜËˆ]È]X˜\ÙHX[[™Ú[™]\›œÈLÈÚ[HÜİÜ™TÔS\È[˜]˜Z[X›K‚‚ˆÈÈÈ\\ˆ˜Y[™ÈYZ[š\İ˜][Û‚‚”İ\Üˆ™\Ù]HXİ]™HÚ[][]YXØÛİ[Ú]İ][][™Èš[Üˆ^\š[Y[Î‚‚˜Úœ]Ûˆ[H\š›ØœËœ™\Ù]Ü\\—ØXØÛİ[KZ[š]X[XØ\ÚLK\™X\ÛÛˆ˜˜\Ù[[™H™Y›Ü™H™X]\™H‚˜‚•HÛÛ[X[™\œÙ\È[Û™^H\ÈXÚ[X[İÜÈHXİ]™H^\š[Y[[™Ü™X]\ÈHœ™\Ú[šÙY[ˆ[ˆÛ™H˜[œØXİ[Û‹ˆİÜ[™ÈH[ˆÙ\È›İ\]ZY]H]ˆÜ[ˆÜÚ][ÛœÈ™]Z[ˆZ\ˆ]X[]H[™[Hİ]K[™›ÈŞ[]XÈ˜YH\ÈÜ™X]Yˆ\İÜšXØ[[œÈ\™H]Y\XX›H›İYÚÑUÜ\\‹Ü[œØ[™ÑUÜ\\‹Ü[œËŞÜ[—ÚYXÈÜÚ][ÛœÈ[™˜Y\ÈXØÙ\[ˆÜ[Û˜[[—ÚYˆ™\Ù]™[XZ[œÈÓK[Û›H[[]][XØ]Y]]][Ûˆ[™Ú[È^\İ[™H]\™H]][XØ]Yœ›Û[™Xİ[Ûˆ]\İ™]\ÙH\È\ÙHØ\ÙK‚