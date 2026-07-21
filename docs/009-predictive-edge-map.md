# Predictive Edge Map

The Predictive Edge Map is a reproducible historical diagnostic of whether the seven
current Features and derived `baseline_technical` version-1 components carry measurable
information about later price movement. It never creates a Strategy, places an order,
or claims causality or profitability.

Volatility measures movement size; predictability measures a repeatable relationship
between present evidence and later movement. A predictor, prediction horizon,
evaluation frequency, and holding period are separate concepts.

## Temporal and statistical contract

Windows use half-open chronological training, validation, and test ranges. Training
alone defines predictor centers, positive/negative direction mappings, five quantile
bins, and the high-volatility threshold. Validation selects candidate relationships
and freezes each quantile bin, its training-defined bounds, direction mapping, and
suggested action. Test filters observations through those exact frozen bins and can
only confirm or reject each selection; evidence from another bin cannot confirm it. An exact
future candle must remain inside its test phase; data at or after the test end is never
used. Defaults are 180/60/60 days, a 60-day step, and six recent windows.

Default horizons are 30 minutes, 1, 2, 6, 12, 24, and 48 hours. Gross forward return,
direction, absolute movement, and net movement magnitude after friction are calculated in memory. The
default neutral band is round-trip fees plus slippage. No per-candle target or
predictor observation is persisted.

EMA predictors use `(EMA / close) - 1`, not nominal price levels. Directional predictors
are explicitly zero-centered or training-centered. ATR and relative volume are
magnitude/activity predictors, while volume confirmation is confirmation-only; these
predictors expose no fabricated directional hit rate.

Correlation sign describes a mathematical relationship, not an action. Each selected
bin derives one explicit suggestion: `long_entry`, `exit_or_avoid`, or
`diagnostic_only`. The derivation combines the training-frozen mapping, the bin's
position relative to its center, validation return, long-aligned net return or avoided
downside, and friction. A negative correlation may therefore support a long entry in a
low bin and an exit/avoid signal in a high bin. No suggestion represents short selling.
Global predictor correlation and hit metrics remain diagnostics and do not enable or
disable all bins. Validation evaluates each directional bin independently and requires
its hit rate to improve materially over the phase's unconditional always-up or
always-down benchmark for the suggested action; the bin's own majority is not used as
its benchmark.

The analysis reports correlation, directional precision, movement, post-friction
outcomes, dispersion, standard error, quantile behavior, and consistency across
windows. Undefined metrics remain null with `insufficient_evidence`; zero is not used
as a substitute. Conditions are deliberately limited to all observations, EMA 20
above/below EMA 50, and training-defined high/normal volatility.

Evidence can be promising positive, promising negative, weak, unstable, not detectable,
or insufficient. Promising evidence must clear versioned sample, window-consistency,
directional/correlation, improvement over directional benchmarks, and post-friction
requirements and cannot rest on one unusual
window. Failure to detect an edge is not proof that none exists. Multiple comparisons
and false discoveries remain limitations. Correlation is not causality, and historical
evidence does not guarantee future profitability.

## Strategy hypotheses and future flow

Promising evidence creates an immutable `StrategyHypothesis` with complete instrument
identity, predictor and hypothesis contract versions, horizon, condition, frozen
direction mapping, one bin/range, one suggested action, validation evidence, confirmed
test evidence, cross-window stability, rationale, limitations, and cautious timing
suggestions. It has no entry/exit threshold or position size and is not executable.
Different bins of one predictor may create independent long-entry and exit/avoid
hypotheses when each has independently selected and confirmed evidence.

```text
Predictive Edge Map
-> Evidence-Guided Candidate Generator
-> Historical Validation
-> Manual Promotion
-> Parallel Paper Wallet
```

## Persistence and API

Only configuration, windows, aggregate metrics, per-bin validation selections and test
confirmations, training boundaries, stability, classifications, benchmark summaries,
and hypotheses are relationally persisted. The
dataset fingerprint and every material input form the reusable experiment identity.
All collections use bounded deterministic keyset pagination, UTC, and Decimal.

- `GET /optimization/predictive-edge-runs`
- `GET /optimization/predictive-edge-runs/{run_id}`
- `GET /optimization/predictive-edge-runs/{run_id}/edges`
- `GET /optimization/predictive-edge-runs/{run_id}/edges/{edge_id}/windows`
- `GET /optimization/predictive-edge-runs/{run_id}/hypotheses`

The GET endpoints are strictly read-only, do not contact providers, and never start an
analysis. Without `--persist`, analysis performs no insert, update, reuse-count change,
or commit. Insufficient history is persisted only as the explicit
`insufficient_history` conclusion, never as an empty completed run.
