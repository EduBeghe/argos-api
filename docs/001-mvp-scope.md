# MVP Scope

## Objective

The MVP will demonstrate an auditable research loop for one liquid cryptocurrency without coupling the domain model permanently to cryptocurrency. It will estimate probabilities from measurable evidence, make simulated decisions, paper trade them, and evaluate performance after fees and simulated slippage.

The MVP will not trade real money and will not claim guaranteed profitability.

## Functional stages

### 1. Market observation

Ingest and persist the market data needed for the initial research subject. Observations must retain enough context to support reproducible analysis and later inspection.

### 2. Feature and score calculation

Calculate a small initial set of explicit, persisted, explainable features and component scores. Candidate score families may include technical analysis, momentum, volatility, market context, news, sentiment, and machine-learning outputs, but the initial selection remains an open decision.

An aggregated Argos Score may be introduced later. Any aggregation must preserve every contributing component score, metric, and rationale.

### 3. Decision generation

Generate simulated decisions from the available evidence. Each decision must record the observations, component metrics, scores, reasoning, requested action, and requested sizing that supported it. The initial `baseline_long_only` version-1 thresholds, position allocation, cadence, and supported actions are defined in ADR-009. They are versioned hypotheses rather than permanent rules for future Strategies.

### 4. Paper-trade execution

Translate simulated decisions into paper trades. Execution must model fees and slippage and persist orders, fills, positions, and resulting cash or portfolio state. No live-money execution is in scope.

### 5. Performance evaluation

Evaluate the simulated strategy using persisted decisions and paper-trade results. Report performance after fees and simulated slippage and compare it with buy-and-hold over the same evaluation period.

## Initial metrics

The MVP must expose at least:

- total return;
- realized and unrealized profit and loss (P&L);
- maximum drawdown;
- win rate;
- profit factor;
- number of trades;
- average trade return;
- exposure;
- fees;
- simulated slippage; and
- comparison against buy-and-hold.

Metric definitions and calculation conventions must be documented before implementation and remain open decisions for now.

## Dashboard information

The API must expose persisted market observations, component metrics and scores, generated decisions and reasoning, paper trades and positions, and performance metrics for consumption by a separate web dashboard.

## Out of scope

- Trading real money.
- Claims or assurances of profitability.
- Broad multi-asset coverage in the first implementation.
- Final formulas, thresholds, providers, exchanges, frameworks, deployment platforms, model choices, or score weights.
- Production operational concerns not required to validate the research loop.

## Open decisions

The score set, metric conventions, exact fee and slippage models, initial Paper Account cash balance, Paper Trading execution details, stop-loss and take-profit rules, future Strategy types, calibrated confidence, backtesting methodology, evaluation period, technologies, deployment approach, and any future Argos Score aggregation method remain unresolved.
