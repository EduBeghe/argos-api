# Architecture Overview

Argos is an asset-agnostic quantitative research and predictive decision-support platform. Its initial backend is a modular monolith, not a distributed microservices system. Modules are separated by responsibility so they can evolve independently and, if a demonstrated operational need appears, later be extracted into independent services.

The offline ML research boundary includes a sequential, reproducible model-screening job. It consumes the validated ML experiment dataset, fits Ridge, LightGBM, and CatBoost only on chronological development folds, and persists evidence without connecting models to live Decisions or Paper Trading. See [ML Model Screening](ml-model-screening.md).

The frontend is a responsive Angular application maintained in the separate `argos-web` repository.

## Current Deployed Execution

The repository is deployed as two one-shot backend jobs plus the continuously serving API:

- `argos-ingest` runs every five minutes, persists fully closed Binance Spot candles, then calculates only bounded live Features and Scores. It does not generate Decisions.
- `argos-decide` runs at UTC minutes 02 and 32, consumes existing persisted evidence for the latest 30-minute boundary, creates Decision Cycles and Decision Snapshots, executes Paper Trading, and records equity snapshots.
- `argos-api` exposes explicit read-only HTTP operations and never runs periodic jobs as a side effect of a request.
- PostgreSQL is the shared transactional evidence store.
- `argos-web` is a static Angular SPA from a separate repository and resolves instruments from the catalog rather than symbol-specific branches.

Candle ingestion, Feature calculation, and Score calculation commit as separate stages. A downstream failure therefore preserves earlier committed candles, and a later run retries missing recent analytics even when candle inserts are duplicates. Live Feature calculation loads at most 1,000 context candles per instrument, so historical backfill does not make live runtime proportional to the complete candle table.

Decision work uses one transaction per instrument. PostgreSQL uniqueness makes completed boundaries idempotent, while independent transactions prevent one instrument failure from undoing another instrument's successful cycle. The API does not contact Binance, execute Paper Trades, or start optimization jobs.

For the consolidated current-state reference, see [Current System Reference](010-current-system.md).

## High-Level Data Flow

```text
External data source
→ Ingestion
→ Storage
→ Feature Engine
→ Scoring
→ Decision Engine
→ Decision Snapshot
→ Paper Trading
→ Metrics
→ API
→ Web
```

The analytical pipeline runs independently of its presentation layers. The API exposes system information and actions; the Web application visualizes them and supports user interaction.

## Modules

### Ingestion

Ingestion connects to external data providers and retrieves raw market data. The MVP ingests five-minute Binance Spot candles and retains normalized price, volume, activity, temporal, interval, and provenance fields that have present or justified future analytical value. Later inputs may include trades, order book data, funding rates, open interest, news, and sentiment.

Its input is provider-specific data. Its output is validated, normalized data in Argos internal formats. Ingestion orchestration calls the Binance client and passes each raw provider response to the Binance normalizer. The client handles only HTTP transport and never constructs domain models. The normalizer knows both Binance's positional kline contract and Argos's MarketCandle contract, and is responsible for translating between them.

The ingestion pipeline and closed-candle orchestration stage normalized candles without committing. The one-shot executable job owns the database session, Binance client, engine, transaction boundaries, and cleanup. It commits the fully closed BTC/USDT and ETH/USDT candle batch atomically before starting feature calculation. The current Railway deployment owns external scheduling; Argos includes no internal job framework, and retries remain outside this orchestration.

Binance's positional tuple and all other provider-specific formats must end at this boundary. Domain, persistence, Features, Scores, Strategies, and Decisions do not access Binance directly. Ingestion does not calculate trading signals or make decisions.

### Storage

Storage persists and retrieves market history, calculated features and indicators, component scores, decisions, decision snapshots, paper trades, positions, strategy configuration, and performance metrics.

Its inputs are records produced by the application modules; its outputs are consistent, queryable historical records. Storage provides persistence and retrieval boundaries and contains no trading decision logic.

### Feature Engine

The Feature Engine transforms raw market data into measurable analytical variables. A feature is a measurable input used by rules, scoring models, or future machine-learning models.

It may calculate RSI, moving averages, MACD, ATR, volatility, momentum, volume changes, and returns over different windows. Its input is normalized market data; its output is a reusable, persisted feature set. It does not decide whether to buy or sell.

The first feature pack persists EMA 20, EMA 50, RSI 14, ATR 14, one-candle return, twelve-candle return, and twenty-candle relative quote volume. Each value is versioned, aligned to its eligible candle, and uses only current and earlier data. In production, the closed-candle job starts feature calculation only after its candle transaction commits, then commits the complete feature pack for both assets in a second atomic transaction. A feature failure cannot roll back committed candles and is recoverable on the next idempotent run. The standalone feature command remains available for explicit reconstruction and historical execution; scheduling remains external.

### Scoring

Scoring executes independent analytical models using available features and context. It produces explicit component scores such as technical, momentum, volatility, market context, news, sentiment, and machine-learning scores. Every result must be measurable and persist both its score and explanation. Scoring does not execute trades.

The first persisted model is the deterministic `baseline_technical` version 1 score. It reads complete same-candle feature sets and their exact candle close, persists explicit weighted contributions, and generates deterministic explanations at the presentation boundary. The production pipeline commits candles, features, and scores in three separate transactions so a score failure preserves committed observations and features. Missing scores are recoverable idempotently through the existing ingestion command or the standalone score reconstruction command.

Illustrative output:

```text
technical: 0.65 — price is above the selected moving average
momentum: 0.40 — recent returns are positive but weakening
volatility: 0.25 — current volatility increases downside uncertainty
```

The score ranges, formulas, and explanations above are illustrative rather than selected model definitions.

### Decision Engine

The Decision Engine combines normalized, persisted Scores with portfolio state, current exposure, open positions, and versioned Strategy configuration. A Strategy may consume compatible component Scores directly or an Argos Score when one exists and that Strategy requires it. `baseline_long_only` version 1 consumes `baseline_technical` version 1 directly and contains no provider-specific decision logic.

Its inputs are persisted evidence and account context. Its output is one of `OPEN_POSITION`, `INCREASE_POSITION`, `SELL_PARTIAL`, `SELL_ALL`, or `NO_ACTION`, with requested sizing where required. `NO_ACTION` intentionally leaves the portfolio unchanged; `HOLD` is not a Decision action. The initial `baseline_long_only` version 1 emits only `OPEN_POSITION`, `SELL_ALL`, and `NO_ACTION`. It does not interact with a real exchange in the MVP.

Decision generation is separate from the five-minute ingestion pipeline. A Railway one-shot process evaluates the completed :00 and :30 UTC boundaries through `python -m app.jobs.generate_market_decisions`, scheduled externally at minute 02 and 32. It reads persisted Paper Position state through an explicit adapter, never assumes that no position is open, and never treats an earlier Decision as proof of execution. Argos contains no internal scheduler.

### Decision Snapshot

A Decision Snapshot is an immutable historical record created whenever Argos generates a decision. The persisted baseline snapshot stores the supplied position state, source candle and market price, immutable source Score identity and value, Decision and requested sizing, applied rule and explanation, and model, Strategy, and configuration versions. An aggregated Argos Score is included only when available and used. A calibrated confidence value may be absent or null; `signal_strength` is score magnitude, not probability or confidence.

Snapshots allow the team to reconstruct why a decision was made and support debugging, model-version comparisons, audits, and later model evaluation. This module persists records; it is not an analytical engine.

### Paper Trading

Paper Trading is a virtual brokerage account that uses simulated money. It receives requested intent from the Decision Engine and determines the executed asset quantity, execution price, notional, fees, simulated slippage, and resulting cash and position changes. Requested percentages are not executed quantities. Paper Trading manages versioned experiment runs, virtual cash, positions, immutable Trades, average entry price, and realized P&L. It never sends orders to a real exchange in the MVP.

Version 1 starts with 10,000 USDT by default, charges a 0.10% fee on executed notional, and applies 0.05% adverse slippage. In plain language, the simulation assumes a slightly worse execution price than the observed market price: buys cost more and sells receive less. Calculations use Decimal values quantized to 18 fractional digits with round-half-even. These are versioned simulation hypotheses, not guaranteed Binance execution costs.

Resetting administratively stops the active run and creates a linked successor without deleting or modifying historical Positions or Trades. A STOPPED run is not liquidated: an open Position remains historically open-at-stop, and only a simulated SELL_ALL closes a Position. Reset creates no synthetic Trade. Historical runs, Positions, and Trades are queryable by run identity.

Before opening a Position, Paper Trading values every existing open Position with the latest eligible persisted closed-candle price at or before the Decision timestamp. It never substitutes entry price for current market evidence and fails closed when valuation evidence is missing. The persisted candles and immutable Decision timestamp make the valuation reproducible.

The read-only Paper portfolio API values any active or historical run from persisted candles using the complete Position instrument identity. It reports each candle timestamp independently and defines `valued_at` as the UTC response-assembly time. Market value is quantity multiplied by mark price; cost basis includes entry notional and entry fees; unrealized P&L is market value minus cost basis; equity is current cash plus Position market values; and total P&L is equity minus initial cash. Version 1 does not estimate hypothetical exit fees or slippage. Configured MVP assets are not special cases in this calculation.

After all configured instrument cycles complete successfully for a 30-minute boundary, the Decision job persists one immutable Paper equity snapshot in its own atomic transaction. The parent stores run cash, Position market value, equity, P&L, return, configuration version, `scheduled_boundary`, and `recorded_at`; relational children preserve each open Position identity, mark, value, unrealized result, and source candle time. `scheduled_boundary` is the historical market cutoff and `recorded_at` is the actual persistence time. Marks use only compatible persisted candles closed at or before the boundary, never live provider access or entry-price fallback.

Snapshots are unique by Paper Run and boundary. Completed instrument transactions remain independent: any instrument failure suppresses the aggregate snapshot, while a snapshot failure leaves completed cycles committed but fails the job for retry. Repeated or concurrent executions reuse the existing snapshot without changing it. The formulas and no-hypothetical-exit-cost policy match the read-only portfolio valuation. Every instrument in a run must currently be valuable in the run's account currency; explicit multi-currency conversion remains future work.

The read-only Paper performance API summarizes any active or stopped run exclusively from its immutable aggregate equity snapshots. The implicit initial point is `initial_cash` at `run.started_at`; total return compares the latest persisted equity with that point. Current drawdown is the non-negative decline from the preceding running peak at the latest point, while maximum drawdown preserves the worst historical decline. Metrics are not annualized, predictive, or guarantees of profitability, and a run without snapshots returns a neutral baseline. The query reads only boundary and equity for the complete run history, never Position snapshots, assets, providers, candles, or external services. SQL aggregation or daily summaries may replace the bounded in-memory calculation when snapshot volume justifies it.

The read-only Paper Strategy Evaluation uses the latest successful Equity Snapshot as its authoritative boundary. A pure domain calculation combines bounded persisted Decisions, Trades, closed Positions, snapshot Position details, and compatible candles. It reports Strategy results, complete-identity instrument results, and a versioned equal-weight buy-and-hold comparison without contacting providers or persisting synthetic evaluation state. Missing benchmark evidence makes the aggregate comparison explicitly unavailable rather than changing its historical universe or weights.

Paper execution also enforces an experiment boundary: the Decision timestamp and execution timestamp must not predate the active run, and execution must not predate the Decision. A Decision created during a STOPPED predecessor run cannot execute against its successor. Timestamp eligibility is sufficient before the modules are wired; the wiring change will evaluate an explicit execution-context identity.

The reusable reset use case is exposed only through an administrative CLI until authenticated mutation endpoints exist. A future authenticated frontend reset must call the same use case.

### Metrics

Metrics evaluates the quality of Argos decisions and simulated trading performance. It calculates total return, realized P&L, unrealized P&L, maximum drawdown, win rate, profit factor, number of trades, average trade return, exposure, fees, slippage, and comparison against buy-and-hold.

This module consumes persisted decisions and paper-trading outcomes and produces evaluation evidence. It remains separate from decision generation. A high win rate alone does not prove that a strategy is profitable because losses, costs, exposure, and trade sizes also affect results.

### API

The API exposes system information and actions to the frontend. It provides read-only access to persisted market candles, ingestion freshness, Scores, and Decision Snapshots; these endpoints never trigger ingestion or Decision generation. It may later provide access to simulated positions, simulated account state, performance metrics, and backtest execution.

Bounded historical Candle, Score, Decision Snapshot, and Paper Trade reads use the complete asset, provider, market type, and interval identity. A partial asset-plus-interval identity is accepted only when the enabled catalog resolves it unambiguously. Inclusive timezone-aware ranges, deterministic chronological ordering, and a maximum limit are shared query semantics. Repositories filter, order, and limit in PostgreSQL; routes perform no calculations, execution, provider access, or cross-endpoint HTTP calls. Existing response models remain authoritative so explainability, provenance, Decimal values, strategy intent, and simulated execution costs are preserved without frontend-specific projections.

The immutable in-code market instrument catalog is the MVP source of truth for canonical Asset identity plus provider, provider symbol, market type, interval, presentation metadata, display order, and ingestion, scoring, and Decision capabilities. The current argos-ingest process consumes only a validated Binance Spot five-minute projection and fails before I/O if it is empty, mixed, duplicated, or incompatible; a future provider requires its own projection, adapter, and job. Feature, Score, and Decision jobs select instruments through their capability projections. Existing market read endpoints resolve provider and market type from the catalog rather than assuming Binance as a fallback. The read-only `GET /market/instruments` response exposes the same definitions so the frontend can render instruments without BTC- or ETH-specific components. Paper Trading capability is derived from Decision capability in the MVP. SPY and Alpaca are not configured. A future equity source also requires market-session rules, not custom frontend logic. Persistence may replace the in-code catalog if scale or dynamic administration later warrants it without changing the read contract.

The read-only `GET /dashboard/overview` endpoint composes the initial screen from catalog-ordered instrument definitions and bounded batch queries for market status, latest Scores, latest Decisions, and open Paper Positions. Market status is queried and related by complete asset, provider, market type, and interval identity; freshness uses the configured interval's explicit closure rule and fails clearly when that rule is not implemented. Instruments without ingestion capability do not affect system freshness. The compositor reads one UTC assembly timestamp for freshness and portfolio valuation, performs no calculations or provider calls, and treats missing persisted evidence as normal partial information. An active Paper Run whose portfolio lacks valuation evidence fails explicitly with HTTP 409. Detailed endpoints remain authoritative for equity history, trades, and other historical collections. Frontend cards are driven by catalog data rather than BTC- or ETH-specific components; future instruments do not change the overview contract. SPY remains outside the MVP.

The API orchestrates application use cases and maps them to explicit module interfaces. It does not contain core analytical or trading logic.

In production, the API and one-shot ingestion job are separate processes. The API process applies migrations before Uvicorn starts and owns HTTP resource lifecycle. The ingestion job owns its database and Binance client resources and separate candle and feature transactions. The external platform owns five-minute scheduling; PostgreSQL natural uniqueness makes repeated and overlapping ingestion idempotent without distributed locks. Browser access is restricted to explicitly configured CORS origins. API-only authentication settings and middleware do not participate in jobs or migrations.

The one-shot Decision Cycle job is also separate from ingestion. At minute 02 and 32 it selects fresh compatible Score evidence no later than the corresponding :00 or :30 boundary, reads the persisted Paper Position, creates one Decision Snapshot, and simulates at most one Trade. NO_ACTION completes an auditable cycle without a Trade. It never contacts Binance.

Each configured instrument uses an independent transaction, so a BTC-specific evidence or valuation failure does not roll back a committed ETH cycle. PostgreSQL uniqueness and row locking make retries and overlapping Railway invocations safe, but do not claim exactly-once scheduling. There is no internal scheduler and no real-money execution.

### Web

The Web module lives in the separate `argos-web` repository and is implemented as a responsive Angular dashboard. It displays price evolution, scores and their explanations, decisions, paper trades, virtual portfolio state, performance metrics, buy-and-hold comparison, strategy status, and process health.

The initial experience is optimized for desktop web use while remaining responsive. The browser does not calculate authoritative trading logic.

## Module Boundaries

- Each module has one primary responsibility.
- Analytical outputs are measurable and persisted where relevant.
- Modules communicate through explicit application interfaces.
- External provider-specific formats do not leak beyond Ingestion.
- Frontend concerns do not leak into analytical modules.
- Real-money trading is outside the MVP.

## Initial MVP Simplifications

The first implementation may use:

- BTC/USDT and ETH/USDT;
- Binance Spot;
- normalized five-minute candle data with retained price, volume, activity, temporal, interval, and provenance fields;
- a small initial feature set;
- a small number of explainable scores;
- one simulated account;
- one decision cycle;
- no real-money execution; and
- no news, sentiment, order book, or machine-learning integration until the base pipeline is measurable.

## Open Decisions

The following remain unresolved:

- future Feature sets and Score models;
- future Argos Score aggregation;
- stop-loss and take-profit rules;
- future Strategy types;
- calibrated confidence;
- backtesting methodology;
- evaluation conventions and future provider, market, and asset-class support requirements.

## Architectural Principle

> Argos does not attempt to predict the future with certainty. It estimates probabilities using measurable evidence, records why every decision was made, and evaluates those decisions through simulated results.
