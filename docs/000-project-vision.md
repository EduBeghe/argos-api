# Project Vision

## Purpose

Argos is a private, asset-agnostic quantitative research and predictive decision-support platform. Its purpose is to turn measurable evidence into transparent probability estimates and simulated decisions that can be evaluated objectively.

Argos does not predict the future with certainty. It must communicate uncertainty and must not claim or imply guaranteed profitability.

## Principles

- Evidence over certainty: estimates must be grounded in observable data.
- Explicit metrics: every analytical component must produce persisted, explainable metrics.
- Traceable signals: every signal must preserve its component scores and the reasoning that produced it.
- Independent components: scores may cover technical analysis, momentum, volatility, market context, news, sentiment, and machine-learning outputs.
- Explainable aggregation: independent scores may later contribute to an aggregated Argos Score without erasing their individual values or rationale.
- Asset-agnostic design: the MVP compares BTC/USDT and ETH/USDT using Binance Spot observations, while core domain concepts remain provider-independent and applicable to other providers, markets, and asset classes.
- Honest evaluation: performance must account for fees and simulated slippage and must be compared with a relevant baseline.
- Safety first: the MVP will use paper trading only and will not trade real money.

## Product direction

Argos will support a repeatable path from market observation to feature and score calculation, decision generation, simulated execution, and performance evaluation. Persisted evidence and outcomes will be exposed for a web dashboard so users can inspect what the platform observed, decided, simulated, and measured.

## Open decisions

Future Feature sets, Score models, Argos Score aggregation, provider and asset-class support requirements, fee and slippage models, Paper Trading execution details, calibrated confidence, backtesting methodology, and evaluation conventions remain open. Current MVP and versioned choices are recorded in the repository documentation and ADRs without becoming permanent product restrictions.
