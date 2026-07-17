# Architecture Overview

Argos is an asset-agnostic quantitative research and predictive decision-support platform. Its initial backend is a modular monolith, not a distributed microservices system. Modules are separated by responsibility so they can evolve independently and, if a demonstrated operational need appears, later be extracted into independent services.

The frontend is a responsive Angular application maintained in the separate `argos-web` repository.

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

Decision generation is separate from the five-minute ingestion pipeline. A future Railway one-shot process will evaluate every 30 minutes through `python -m app.jobs.generate_market_decisions`. It must receive persisted Paper Position state through an explicit adapter and currently fails closed because that source does not exist. It never assumes that no position is open or treats an earlier Decision as proof of execution. Argos contains no internal scheduler.

### Decision Snapshot

A Decision Snapshot is an immutable historical record created whenever Argos generates a decision. The persisted baseline snapshot stores the supplied position state, source candle and market price, immutable source Score identity and value, Decision and requested sizing, applied rule and explanation, and model, Strategy, and configuration versions. An aggregated Argos Score is included only when available and used. A calibrated confidence value may be absent or null; `signal_strength` is score magnitude, not probability or confidence.

Snapshots allow the team to reconstruct why a decision was made and support debugging, model-version comparisons, audits, and later model evaluation. This module persists records; it is not an analytical engine.

### Paper Trading

Paper Trading is a virtual brokerage account that uses simulated money. It receives requested intent from the Decision Engine and determines the executed asset quantity, execution price, notional, fees, simulated slippage, and resulting cash and position changes. Requested percentages are not executed quantities. Paper Trading manages versioned experiment runs, virtual cash, positions, immutable Trades, average entry price, and realized P&L. It never sends orders to a real exchange in the MVP.

Version 1 starts with 10,000 USDT by default, charges a 0.10% fee on executed notional, and applies 0.05% adverse slippage. In plain language, the simulation assumes a slightly worse execution price than the observed market price: buys cost more and sells receive less. Calculations use Decimal values quantized to 18 fractional digits with round-half-even. These are versioned simulation hypotheses, not guaranteed Binance execution costs.

Resetting administratively stops the active run and creates a linked successor without deleting or modifying historical Positions or Trades. A STOPPED run is not liquidated: an open Position remains historically open-at-stop, and only a simulated SELL_ALL closes a Position. Reset creates no synthetic Trade. Historical runs, Positions, and Trades are queryable by run identity.

Before opening a Position, Paper Trading values every existing open Position with the latest eligible persisted closed-candle price at or before the Decision timestamp. It never substitutes entry price for current market evidence and fails closed when valuation evidence is missing. The persisted candles and immutable Decision timestamp make the valuation reproducible.

Paper execution also enforces an experiment boundary: the Decision timestamp and execution timestamp must not predate the active run, and execution must not predate the Decision. A Decision created during a STOPPED predecessor run cannot execute against its successor. Timestamp eligibility is sufficient before the modules are wired; the wiring change will evaluate an explicit execution-context identity.

The reusable reset use case is exposed only through an administrative CLI until authenticated mutation endpoints exist. A future authenticated frontend reset must call the same use case.

### Metrics

Metrics evaluates the quality of Argos decisions and simulated trading performance. It calculates total return, realized P&L, unrealized P&L, maximum drawdown, win rate, profit factor, number of trades, average trade return, exposure, fees, slippage, and comparison against buy-and-hold.

This module consumes persisted decisions and paper-trading outcomes and produces evaluation evidence. It remains separate from decision generation. A high win rate alone does not prove that a strategy is profitable because losses, costs, exposure, and trade sizes also affect results.

### API

The API exposes system information and actions to the frontend. It provides read-only access to persisted market candles, ingestion freshness, Scores, and Decision Snapshots; these endpoints never trigger ingestion or Decision generation. It may later provide access to simulated positions, simulated account state, performance metrics, and backtest execution.

The API orchestrates application use cases and maps them to explicit module interfaces. It does not contain core analytical or trading logic.

In production, the API and one-shot ingestion job are separate processes. The API process applies migrations before Uvicorn starts and owns HTTP resource lifecycle. The ingestion job owns its database and Binance client resources and separate candle and feature transactions. The external platform owns five-minute scheduling; PostgreSQL natural uniqueness makes repeated and overlapping ingestion idempotent without distributed locks. Browser access is restricted to explicitly configured CORS origins.

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
