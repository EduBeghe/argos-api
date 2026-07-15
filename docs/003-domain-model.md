# Argos Domain Model

## Purpose

This document defines the shared business vocabulary of Argos independently of any programming language, framework, provider, or persistence technology. Future backend, frontend, persistence, tests, and documentation must use these concepts consistently.

The MVP focuses on BTC/USDT and ETH/USDT on Binance Spot. The domain remains asset-agnostic and must not become permanently coupled to Binance, BTC, ETH, USDT, or cryptocurrency.

## Core Domain Concepts

### Asset

An Asset represents any financial instrument analyzed by Argos. Initial MVP examples are BTC/USDT and ETH/USDT. Future examples may include SPY, GLD, and YPF.

An Asset contains identity and market metadata, but not historical prices. Historical prices belong to Market Candles.

### Market Candle

A Market Candle represents one normalized OHLCV candle. It conceptually contains a timestamp, open, high, low, close, and volume.

Provider-specific market data must be normalized into this structure before entering the analytical pipeline. Provider-specific formats must not leak beyond Ingestion.

### Feature

A Feature is a measurable value extracted from normalized market data. Examples include RSI, EMA 50, EMA 200, ATR, MACD, Momentum, return over a time window, Volume Change, and Realized Volatility.

Features are reusable inputs for analytical models and Strategies. They never make trading decisions.

### Score

A Score is the output of one independent analytical model. Every Score conceptually contains:

- type;
- value;
- explanation;
- timestamp;
- asset; and
- model or rule version.

Scores must be measurable, explainable, versioned where relevant, and persistable. A Score is not a trading decision.

#### MVP Score Types

The MVP uses a small set of explainable Scores derived from market data and technical Features. Possible initial categories include Technical Score, Momentum Score, Volatility Score, and Market Context Score.

Formulas, ranges, weights, and thresholds remain open decisions.

#### Future Score Types Outside the MVP

The following are future extension points and are not part of the initial MVP:

- News Score estimates the probable impact of identifiable and verified market events.
- Sentiment Score measures collective market tone, attention, fear, optimism, or excessive enthusiasm.
- Machine Learning Score represents a probability or prediction produced by a trained model for a clearly defined future outcome.

These Scores require additional data sources, collection pipelines, deduplication, validation, model training, versioning, and evaluation. They may be produced by distinct systems; Argos does not assume that one autonomous agent generates all three. Providers, APIs, AI models, prompts, training techniques, and implementation details remain undefined.

### Argos Score

The Argos Score is an aggregated analytical summary produced from multiple component Scores. It provides a high-level view of current analytical confidence while preserving explainability.

The aggregation algorithm is intentionally undefined and is not necessarily an arithmetic average. The Argos Score preserves every component Score's contribution and explanation, does not replace component Scores, and does not execute trades. Strategies may consume it together with selected component Scores.

The Argos Score is an analytical summary, not a trading decision.

### Strategy

A Strategy is a configurable decision policy. It consumes:

- the Argos Score;
- one or more component Scores;
- account state;
- portfolio state;
- current exposure;
- configured thresholds; and
- risk rules.

A Strategy conceptually defines relevant component Scores, decision thresholds, position sizing rules, stop-loss rules, take-profit rules, and eligibility conditions. Different Strategies may use different subsets of available Scores. Strategies produce Decisions.

Whether Strategies are represented as configuration, code, or a hybrid remains an open architectural decision.

### Decision

A Decision is an action proposed by a Strategy. Its possible values are:

- `BUY`: open or increase exposure.
- `SELL`: reduce exposure according to the active Strategy.
- `CLOSE_POSITION`: explicitly exit an existing position.
- `NO_ACTION`: intentionally leave the current portfolio unchanged without opening, closing, or modifying a position.

Every Decision must be explainable and reference the evidence used to produce it. `HOLD` is not part of this domain vocabulary.

### Decision Snapshot

A Decision Snapshot is an immutable historical record created every time Argos generates a Decision. It conceptually contains:

- timestamp;
- asset;
- market price;
- market context;
- Features;
- component Scores;
- Argos Score;
- Decision;
- confidence;
- applied rules;
- Strategy version; and
- model and configuration versions.

Its purpose is traceability, reproducibility, debugging, audits, comparison between model versions, and later model and Strategy evaluation. A Decision Snapshot is a persisted historical record, not an analytical engine.

### Paper Account

A Paper Account is a virtual trading account using simulated money. It conceptually contains virtual cash, virtual holdings, open positions, closed positions, realized P&L, unrealized P&L, and Trade history.

It never interacts with real money in the MVP.

### Paper Position

A Paper Position represents one simulated position for an Asset and may be open or closed. It conceptually contains the asset, quantity, average entry price, entry time, current or exit price, exit time, realized or unrealized P&L, and related Trades.

### Trade

A Trade is one simulated execution generated by Paper Trading after a Decision. It conceptually contains:

- asset;
- side;
- quantity;
- execution price;
- fees;
- simulated slippage;
- timestamp; and
- related Decision.

A Decision is an analytical instruction. A Trade is the resulting simulated execution.

### Metrics

Metrics are measurable performance indicators used to evaluate Argos and its Strategies. Examples include Total Return, Realized P&L, Unrealized P&L, Maximum Drawdown, Win Rate, Profit Factor, Number of Trades, Average Trade Return, Exposure, Fees, Slippage, Sharpe Ratio, and comparison against Buy-and-Hold.

Metrics evaluate Decisions, Trades, Positions, Accounts, and Strategies. They never generate Decisions. A high Win Rate alone does not prove profitability.

## Domain Relationships

The conceptual domain relationships are:

- An Asset has Market Candles.
- Market Candles produce Features.
- Features feed analytical models.
- Analytical models produce component Scores.
- Component Scores contribute to the Argos Score.
- A Strategy consumes the Argos Score, selected component Scores, account context, portfolio context, and risk rules.
- A Strategy generates a Decision.
- A Decision creates a Decision Snapshot.
- Paper Trading may execute a Decision.
- Paper Trading creates Trades.
- Trades update Paper Positions and the Paper Account.
- The Paper Account, Positions, Trades, and Decisions generate Metrics.

These are conceptual domain relationships, not database relationships or implementation instructions.

## MVP Boundary

The initial MVP includes:

- BTC/USDT;
- ETH/USDT;
- Binance Spot;
- normalized OHLCV candles;
- a small technical Feature set;
- a small set of explainable technical, momentum, volatility, or market-context Scores;
- an aggregated Argos Score;
- one or more Strategies;
- Decisions;
- Decision Snapshots;
- Paper Trading;
- simulated Trades;
- simulated Positions;
- one Paper Account;
- performance Metrics;
- API exposure; and
- dashboard visualization.

The initial MVP excludes:

- News Score;
- Sentiment Score;
- Machine Learning Score;
- news ingestion;
- social-media ingestion;
- autonomous agents;
- order-book analysis;
- funding rates;
- open interest;
- futures;
- leverage;
- real-money execution;
- stocks; and
- ETFs.

These exclusions are sequencing decisions, not permanent product limitations.

## Domain Principles

- Everything in Argos must be measurable.
- Every Decision must be explainable.
- Every Decision must be reproducible.
- Every analytical output must preserve its provenance and version where relevant.
- Component Scores must remain available even when an Argos Score exists.
- Domain concepts must remain independent from external providers.
- Domain concepts must remain independent from frameworks and persistence technology.
- Business terminology must remain consistent across backend, frontend, tests, and documentation.
- Future complexity must not be introduced before the base pipeline is measurable.

## Open Decisions

The following remain unresolved:

- exact initial Feature set;
- exact initial Score types;
- score value ranges;
- score aggregation method;
- Strategy representation as configuration, code, or hybrid;
- which component Scores each Strategy consumes;
- decision thresholds;
- position sizing rules;
- stop-loss rules;
- take-profit rules;
- candle interval;
- initial simulated capital;
- fee model; and
- slippage model.

No Signal concept is introduced at this stage. It may be considered later if a clear domain need appears.
