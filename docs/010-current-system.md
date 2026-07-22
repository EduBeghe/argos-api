# Current System Reference

This document describes the implementation on `master` after PR #46. It distinguishes operational behavior from planned work and should be updated with material backend changes.

## Instruments and providers

An instrument is the complete identity `(asset, provider, market_type, interval)`. Display symbols are metadata, not identity. The catalog currently contains BTC/USDT and ETH/USDT on Binance Spot with five-minute candles. Both enable ingestion, scoring, decisions, and Paper Trading. The public catalog reports all four capabilities; internally Paper Trading currently derives eligibility from the decision-enabled projection rather than maintaining a separate flag. The code exposes capability-specific catalog resolution instead of symbol branches.

Binance is the only implemented provider. BTC and ETH are the current MVP configuration, not permanent architectural constants. SPY, stocks, ETFs, other providers, market calendars, and provider-specific jobs are future work. `/market/instruments` is the public catalog source.

## Market data and backfill

The live job requests Binance Spot `GET /api/v3/klines`, accepts only closed five-minute candles, normalizes the 12-position response, and persists exact Decimals. Natural uniqueness is asset, provider, market type, interval, and open time. Repeated responses are harmless duplicates.

`BINANCE_BASE_URL` is configurable. `https://api.binance.com` is the default; `https://data-api.binance.vision` is useful for public market-data access where the main endpoint returns HTTP 451 because of network-location restrictions. The URL changes transport only, not instrument identity or normalization.

Historical backfill is candle-only. It does not create historical Feature or Score rows. Existing measurements observed approximately 308–327 bytes of additional table-and-index storage per candle; this is an environment-specific estimate, not a database guarantee. Persisting raw normalized candles preserves reproducible evidence, while historical Feature matrices are constructed in bounded memory for analysis. Live Feature/Score work remains restricted to the recent ingestion target window.

The 1,000-candle live context limit covers the current EMA 20, EMA 50, RSI 14, ATR 14, return 1, return 12, and relative-volume 20 model with conservative recovery room. It prevents an historical backfill from making live runtime and memory proportional to the full table.

## Features and baseline Score

`baseline_features` version 1 calculates and persists seven same-candle technical measurements:

| Feature | Meaning |
| --- | --- |
| `ema_20` / `ema_50` | Fast and slow exponentially smoothed close levels |
| `rsi_14` | Wilder relative-strength balance over 14 close changes |
| `atr_14` | Wilder average true range, including gaps from the previous close |
| `return_1` / `return_12` | Fractional close return over one and twelve candles |
| `relative_volume_20` | Current quote volume divided by the previous 20-candle mean |

The exact formulas and alignment rules are in [Analytical Features](005-analytical-features.md).

`baseline_technical` version 1 consumes all seven Features plus close price. It derives trend, RSI, one-candle return, twelve-candle return, volume activity, and volatility ratio. Directional weights are 0.55 trend, 0.1575 RSI, 0.09 return 1, and 0.2025 return 12; they sum exactly to 1. Volume confirmation has weight 0.25 and scales the directional score by a factor from 0.75 to 1. Persisted Decimals use scale 18 and round-half-even.

`overall_score = directional_score × volume_confirmation_factor`; `signal_strength = abs(overall_score)`. Interpretations range from strongly bearish through neutral to strongly bullish. They describe measured evidence, not an automatic instruction. The Strategy alone decides requested action. See [Baseline Score](006-baseline-score-model.md).

## Baseline Strategy and action vocabulary

`baseline_long_only` version 1, configuration version 1, consumes `baseline_technical` version 1. When no position exists, an overall Score of at least 0.60 requests `open_position` with a 15% target portfolio allocation. When a position exists, a Score at or above 0.20 produces `no_action`; below 0.20 it requests `sell_all` for 100% of that position.

The enum also reserves `increase_position` and `sell_partial`, but version 1 rejects and never emits them. “Open position” is intent at the eligible boundary, not an assertion that the price is a market peak or trough. The threshold uses several normalized signals and can occur during an established trend. A Decision Snapshot preserves requested sizing, rule, evidence identity, and explanation; a Trade is the separate simulated execution record.

Limitations include fixed thresholds and weights, long-only exposure, one position per instrument, one cadence, no order-book model, and no guarantee that historical behavior persists.

## Paper Trading

Paper Trading uses versioned experiments (“runs”), not real wallets or exchange accounts. Version 1 defaults to 10,000 USDT, a 0.10% fee per execution notional, and 0.05% adverse slippage. Buys use observed price multiplied by `1 + slippage_rate`; sells use observed price multiplied by `1 - slippage_rate`. Slippage is a simplified unfavorable execution assumption, not an exchange quote or a second fee.

A run stores initial/current cash, configuration, status, timestamps, optional predecessor, Positions, immutable Trades, Decision Cycles, equity snapshots, and performance history. Valuation uses persisted compatible candles. Realized P&L comes from closed quantity; unrealized P&L marks open quantity; equity is cash plus marked position value. Performance includes return, P&L, drawdown, costs, activity, and snapshot boundaries.

Resetting through `app.jobs.reset_paper_account` stops the active run, starts a successor, links `predecessor_run_id`, and preserves all history. It does not synthesize sells or erase open-at-stop Positions. Changing initial funds therefore creates a new experiment rather than rewriting the old one.

## Historical optimization

### Reproducible dataset and simulation

A historical dataset is identified by complete instrument and half-open UTC range. Readers enforce ordering, candle continuity, closed-candle eligibility, and a deterministic fingerprint. Feature matrices are built from that dataset without persisting millions of historical Feature rows. Historical simulation reuses the production Score, Strategy, fees, slippage, capital, and cadence, and compares results with a buy-and-hold benchmark.

### Refinement and experiment memory

Refinement separates training, validation, and isolated test evidence; explores bounded Score/Strategy parameter grids; evaluates controlled one-at-a-time ablations; and can persist compact runs, candidates, evaluations, benchmarks, and walk-forward evidence. `experiment_key` fingerprints all effective inputs. An identical completed graph can be reused, while incomplete or incompatible evidence is rejected or replaced atomically. Persistence is memory, not learning or activation: it avoids repeating identical experiments and lets later planning use version-compatible history.

### Predictive Edge Map

The Predictive Edge Map examines versioned predictors across horizons, conditions, and walk-forward windows. Training freezes centers, mappings, volatility thresholds, and quantile bounds. Validation selects individual bin/action identities; isolated test confirms only those same frozen bins. Evidence may be promising, unstable, or not detectable. Temporal isolation, fingerprints, frozen mappings, and segment-specific observations limit leakage; none proves causality.

### Complete edge-guided Strategies and qualification v3

Generation remains bounded, but a `long_entry` is now an independent long-only
entry rule rather than only a filter on `baseline_long_only`. Complete candidates
freeze entry and timed/signal/baseline exit rules, cadence, holding bounds, cooldown,
allocation, costs, and source evidence. `exit_or_avoid` remains a separate defensive
overlay.

Qualification v3 assigns `rejected_validation` before isolated test,
`rejected_test` only after validation passes, and `confirmed_candidate` only when
the complete Strategy satisfies the existing strict activity, net-return, production
outperformance, risk, window, and stability evidence. Test metrics are null when
validation does not authorize test. Buy and hold remains an independent comparison.
Earlier evidence remains readable as legacy but is not eligible under v3.

Exact completed runs use a constant-size insert-only candle revision to short-circuit
before loading historical candles, building the Feature matrix, analyzing edges, or
simulating candidates. The weekly job processes scoring-enabled catalog instruments
sequentially under a PostgreSQL advisory lock and never activates candidates or mutates
Paper Trading. See [weekly Strategy research](014-weekly-strategy-research.md).

Plain-language definitions:

- Validation return is net return after fees and slippage in periods used after discovery.
- Test return is net return in later isolated periods that did not participate in selection.
- Confirmed windows count independent time windows that passed validation and then test.

## Current state and next steps

Implemented: live ingestion; bounded Features/Scores; Decisions; Paper execution; portfolio/performance and equity history; dashboard API; separate frontend; candle backfill; reproducible simulation; refinement persistence/reuse; Predictive Edge Map; complete edge-guided Strategy research; qualification v3; weekly research execution; and candidate APIs/UI.

Not implemented: candidate correction/approval or activation; parallel wallets per Strategy; stopping individual Strategies; additional providers/assets; ML; news/sentiment; authentication; or real-money trading. These are possible next steps, not production commitments.
