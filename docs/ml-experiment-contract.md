# ML Experiment Dataset Contract v1

## Purpose and scope

This contract defines a reproducible dataset for future machine-learning experiments. A future model will predict the numeric return of one instrument over a specified future duration. A future duration is simply the elapsed time between a decision boundary and the exact later closed candle used to measure the return. This foundation neither trains a model nor claims predictive ability.

BTC/USDT and ETH/USDT are separate experiments. Their observations are never copied, joined, or used to repair one another. The design is independent of the existing combinatorial Strategy research: it builds numerical explanatory variables and future-return targets, while the existing research system evaluates explicit Strategies and remains available as a baseline.

## Frozen time contract

All ranges are half-open and UTC.

- Development: `[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`.
- Untouched final test: `[2024-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.
- Data at or after 2026-01-01 is outside version 1.

The final-test rows are represented in the manifest but are not included in development folds. This PR never calculates final-test performance. Keeping 2024–2025 untouched prevents later feature, threshold, and model choices from learning from the final evaluation period.

Potential observations occur at UTC minute `00` and `30`. At boundary `12:00`, the reference price is the close of the candle opened at `11:55`, whose persisted close timestamp is before `12:00`. A candle opened at `12:00` cannot affect that observation.

## Targets

For every eligible boundary and duration:

```text
future_return = exact_future_close / reference_close - 1
```

Ordinary return is stored as `float32`; logarithmic return is not used. The 80 durations are generated from three named ranges:

- 30 minutes through 12 hours every 30 minutes: 24;
- 13 through 48 hours every hour: 36;
- 54 hours through seven days every six hours: 20.

The target uses the fully closed five-minute candle ending at the exact future boundary. A missing exact candle, or any gap between the reference and target, excludes that target. No nearest-candle substitution is allowed.

## Input variables

`app/ml/registry.py` is the authoritative deterministic registry. Each entry supplies a stable name, English description, category, historical lookback, `float32` output type, calculation identifier and version, and directional interpretation where meaningful.

The registry derives periods from named sets and contains:

- price returns at 30m, 1h, 2h, 4h, 8h, 16h, 24h, 48h, and 7d;
- EMA distance, one-candle EMA slope, adjacent-scale EMA spread, and volatility-normalized trend at the same systematic scales;
- previous rolling highs/lows, range position, normalized distances, and breakout flags at 1h, 4h, 12h, 24h, and 7d; the current candle is excluded from previous ranges;
- return variability, normalized high-low range, volatility change, and adjacent-scale ratios at 1h, 4h, 12h, 24h, and 7d;
- relative, changed, normalized, and signed-price volume measurements at 1h, 4h, 24h, and 7d;
- normalized candle body, range, wicks, close position, and positive-candle fractions;
- sine/cosine encodings for UTC time of day and weekday.

Zero-range candle close position is deterministically `0.5`. Zero denominators in relative diagnostics produce a neutral finite `0.0`; non-finite outputs make the observation ineligible. Missing calculated values are never filled with zero.

## Missing evidence

Persisted Binance gaps are treated as genuine missing evidence. Argos never interpolates, forward-fills, synthesizes, or borrows a candle from another asset. A row is excluded for insufficient history, a missing reference, a gap in the maximum historical input window, a non-finite feature, a missing future interval candle, or a missing exact target. Exclusion reasons are counted explicitly. Diagnostics distinguish a discontinuous range (`gap_event_count`) from the number of absent five-minute slots inside all such ranges (`missing_candle_slot_count`) and report the first and last missing opening timestamps.

## Development folds and leakage protection

Six-month validation blocks begin on 2022-01-01, 2022-07-01, 2023-01-01, and 2023-07-01. Every emitted fold has two complete years before its embargo-adjusted training end. Consequently, the 2022-01 validation block is omitted. Accumulating and exact rolling-two-year folds remain for the 2022-07, 2023-01, and 2023-07 blocks; the exact rolling-three-year fold remains only for 2023-07. A mode is omitted when its full history is unavailable and is never silently shortened.

A fixed seven-day gap separates training from validation because seven days is the longest target. Therefore no training target can reach validation evidence. Development also stops before the untouched final-test partition. All matrices use integer split indices. `eligible_indices()` is the safe public split operation: for one fold and one duration it intersects temporal rows, feature eligibility, and that duration's target eligibility, so a training consumer cannot accidentally receive NaN or final-test evidence.

## Memory and reproducibility

The reader processes one catalog instrument into compact NumPy columns: timestamps are `int64`, prices and volume are numerical arrays, and no per-candle object graph or datetime lookup dictionary is retained. The builder precomputes causal full-series returns, rolling aggregates, extrema, volatility, volume, EMA, and continuity arrays, then selects the UTC 30-minute reference positions into one shared `float32` feature matrix. Calculations are reused by period rather than repeated for every boundary and variable.

`LazyTargetProvider` uses vectorized timestamp lookup and a continuity prefix to compute one one-dimensional duration target at a time in O(boundaries); it never walks every intervening candle or retains an 80-column target matrix. The paged reader does not contact Binance or write PostgreSQL, and temporary full-size arrays are released after boundary selection.

The SHA-256 dataset fingerprint covers candle values, complete instrument identity, requested range, registry and calculation versions, duration contract, folds, and missing-data policy. Identical ordered evidence and configuration produce the same fingerprint.

## Validation

Inspect the frozen contract without PostgreSQL:

```sh
python -m app.jobs.validate_ml_experiment_dataset --asset BTC/USDT --contract-only
```

Validate one stored instrument across the complete 2020–2026 contract:

```sh
python -m app.jobs.validate_ml_experiment_dataset --asset BTC/USDT
python -m app.jobs.validate_ml_experiment_dataset --asset ETH/USDT
```

The full command emits structured raw and feature-eligible boundary counts, gap events and missing slots, raw and eligible fold counts, all 80 duration eligibility/exclusion summaries, fingerprint, elapsed time, and peak resident memory when the operating system exposes it. It is diagnostic and read-only; it does not train models, choose thresholds, approve Strategies, trade, persist ML variables, or expose an API.
