# Argos Domain Model

## Purpose

This document defines the shared business vocabulary of Argos independently of any programming language, framework, provider, or persistence technology. Future backend, frontend, persistence, tests, and documentation must use these concepts consistently.

The MVP focuses on BTC/USDT and ETH/USDT on Binance Spot. The domain remains asset-agnostic and must not become permanently coupled to Binance, BTC, ETH, USDT, or cryptocurrency. Asset, Feature, Score, Argos Score, Strategy, Decision, Decision Snapshot, Paper Account, Paper Position, Trade, and Metrics are provider-independent domain concepts.

## Core Domain Concepts

### Asset

An Asset represents any financial instrument analyzed by Argos. Initial MVP examples are BTC/USDT and ETH/USDT. Future examples may include SPY, GLD, and YPF.

An Asset contains identity and market metadata, but not historical prices. Historical prices belong to Market Candles.

### Market Candle

A Market Candle represents normalized price, volume, activity, temporal, interval, and provenance data for one market window. It remains provider-independent at the domain level.

Provider-specific market data must be normalized into this concept before entering the analytical pipeline, and provider formats must not leak beyond Ingestion. The initial retained field set is defined by the [Binance market-data contract](004-binance-market-data-contract.md) without copying Binance's positional indexes into the domain model.

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

The first persisted Score model is `baseline_technical` version 1. It consumes the seven version-1 Features documented in the analytical feature reference and uses the documented range and interpretation boundaries. Future Score models remain open and must be versioned independently.

#### Future Score Types Outside the MVP

The following are future extension points and are not part of the initial MVP:

- News Score estimates the probable impact of identifiable and verified market events.
- Sentiment Score measures collective market tone, attention, fear, optimism, or excessive enthusiasm.
- Machine Learning Score represents a probability or prediction produced by a trained model for a clearly defined future outcome.

These Scores require additional data sources, collection pipelines, deduplication, validation, model training, versioning, and evaluation. They may be produced by distinct systems; Argos does not assume that one autonomous agent generates all three. Providers, APIs, AI models, prompts, training techniques, and implementation details remain undefined.

### Argos Score

The Argos Score is an optional aggregated analytical summary produced from multiple component Scores. It provides a high-level analytical view while preserving explainability.

The aggregation algorithm is intentionally undefined and is not necessarily an arithmetic average. The Argos Score preserves every component Score's contribution and explanation, does not replace component Scores, and does not execute trades. A Strategy may consume it when available and required, or consume compatible component Scores directly.

The Argos Score is an analytical summary, not a trading decision.

### Strategy

A Strategy is a configurable decision policy. It consumes:

- an Argos Score when available and required, or one or more compatible component Scores directly;
- account state;
- portfolio state;
- current exposure;
- configured thresholds; and
- risk rules.

A Strategy conceptually defines relevant Scores, decision thresholds, position sizing rules, stop-loss rules, take-profit rules, and eligibility conditions. Different Strategies may use different subsets of available Scores. Strategies produce Decisions. The initial `baseline_long_only` version 1 consumes `baseline_technical` version 1 directly and is deterministic code with explicit versioned configuration constants; it is not a generic rule engine or database-editable configuration.

Future Strategies may use a different representation; this decision is not permanent beyond the versioned initial Strategy.

### Decision

A Decision is an action proposed by a Strategy. Its possible values are:

- `OPEN_POSITION`: create a new position when none exists; includes the requested target portfolio percentage.
- `INCREASE_POSITION`: increase an open position; includes the requested target portfolio percentage or requested increase amount.
- `SELL_PARTIAL`: sell a specified percentage of an open position; includes the requested position percentage to sell.
- `SELL_ALL`: sell 100% of an open position and leave the asset with no open position.
- `NO_ACTION`: intentionally leave the portfolio unchanged without opening, closing, or modifying any position.

Every Decision must be explainable and reference the evidence used to produce it. `HOLD` is not part of this domain vocabulary. `INCREASE_POSITION` and `SELL_PARTIAL` are defined for future Strategy versions but are not emitted by `baseline_long_only` version 1.

### Decision Snapshot

A Decision Snapshot is an immutable historical record created every time Argos generates a Decision. It conceptually contains:

- timestamp;
- asset;
- market price;
- market context;
- Features;
- every Score used by the Strategy;
- an Argos Score only when available and used;
- Decision;
- calibrated confidence, when available;
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

A Decision records requested intent and sizing. A Trade records what Paper Trading actually executed. Requested percentages are not executed quantities.

### Metrics

Metrics are measurable performance indicators used to evaluate Argos and its Strategies. Examples include Total Return, Realized P&L, Unrealized P&L, Maximum Drawdown, Win Rate, Profit Factor, Number of Trades, Average Trade Return, Exposure, Fees, Slippage, Sharpe Ratio, and comparison against Buy-and-Hold.

Metrics evaluate Decisions, Trades, Positions, Accounts, and Strategies. They never generate Decisions. A high Win Rate alone does not prove profitability.

## Domain Relationships

The conceptual domain relationships are:

- An Asset has Market Candles.
- Market Candles produce Features.
- Features feed analytical models.
- Analytical models produce component Scores.
- Component Scores may contribute to an Argos Score.
- A Strategy consumes compatible component Scores directly or an Argos Score when required, together with account and portfolio context and risk rules.
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
- normalized Market Candles;
- a small technical Feature set;
- a small set of explainable technical, momentum, volatility, or market-context Scores;
- an optional future aggregated Argos Score;
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

- future Feature sets;
- future Score models;
- future Argos Score aggregation;
- stop-loss rules;
- take-profit rules;
- future Strategy types;
- calibrated confidence;
- backtesting methodology;
- Paper Trading execution details;
- initial Paper Account cash balance;
- fee model; and
- slippage model.

Support requirements for future providers, markets, and asset classes and their compatible calendars, Features, Scores, and versioned configuration also remain open.

No Signal concept is introduced at this stage. It may be considered later if a clear domain need appears.
