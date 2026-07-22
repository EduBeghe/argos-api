# Weekly Strategy Research

## Current System

Argos can run a bounded weekly research process with:

```bash
python -m app.jobs.run_weekly_strategy_research
```

The process resolves scoring-enabled instruments from the market catalog and handles
them sequentially. Each instrument has an isolated transaction and failure boundary.
The job does not activate a candidate, access an exchange, or mutate Paper Trading.
A completed run with no confirmed candidate is successful research, not an operational
failure.

The research pipeline separates discovery, economic screening, complete-Strategy
generation, training screening, validation, freezing, isolated test, qualification,
and persistence. Test evidence cannot generate, tune, or rank a candidate.

## Complete Strategy Contract

Generation contract version 3 replaces the previous action-filter-only search. That
filter could only permit an entry already requested by `baseline_long_only`, so many
valid edge hypotheses produced no Trades. A version-3 `long_entry` is an independent
long-only entry rule with immutable cadence, holding bounds, cooldown, target
allocation, costs, and source evidence.

The bounded candidate families are:

- `edge_entry_timed_exit`: edge entry with a horizon-derived timed exit;
- `edge_entry_signal_exit`: edge entry paired with one compatible negative edge;
- `edge_entry_baseline_exit`: edge entry with the production Strategy used only as an
  exit rule;
- `exit_or_avoid_overlay`: the legacy defensive overlay that closes an open Position
  or blocks a production entry.

No family creates short exposure, leverage, repeated buys into an open Position, or
exposure above its frozen target. Fees and slippage use the production Paper Trading
calculation functions.

## Budget, Ranking, and Outcomes

The default run retains at most 12 entry hypotheses and 6 exit hypotheses and evaluates
at most the configured candidate budget. Ranking is deterministic; a fingerprint is
the final tie breaker. Before simulation, a multiplicative round-trip cost screen
rejects expected movement that cannot cover fees, slippage, and the versioned margin.

Every run persists a funnel and one explicit outcome, including confirmed candidates,
validation rejection, isolated-test rejection, no predictive edge, or edges that were
not economic after costs. The explanation is derived from recorded evidence. Argos
does not claim that a historical candidate guarantees future profitability.

When validation does not pass, test is not executed. Its metrics are `null`, and no
test rejection reason is fabricated. Older zero-filled evidence remains readable as
legacy but is not eligible under the current qualification contract.

## Reuse

Before loading historical candles, Argos reads a constant-size dataset revision using
the complete instrument identity, requested range, candle count, first and last open
times, and latest ingestion timestamp. Market candles are insert-only and duplicate
inserts use `ON CONFLICT DO NOTHING`, so this revision detects additions without
materializing the dataset.

The reuse identity also includes effective window, predictor, edge, generation,
ranking, qualification, cost, capital, cadence, and candidate-budget inputs. A complete
and structurally valid graph short-circuits candle loading, Feature matrix construction,
edge analysis, and candidate simulation. Running, failed, incomplete, or legacy graphs
are not reusable.

## Railway Runbook

Create a separate Railway service named `argos-research` from the backend repository.
Use the same database and logging environment variables as the API and run:

```text
python -m app.jobs.run_weekly_strategy_research
```

Suggested UTC cron:

```text
17 3 * * 0
```

The offset avoids the usual ingest and decide minutes, while the PostgreSQL advisory
lock still prevents concurrent weekly runs. Use Railway logs for the per-instrument
funnel, outcome, work counters, and stage timings. `Run now` is safe: identical evidence
is reused. A no-candidate outcome needs interpretation, not a retry. Retry operational
failures after correcting their cause. Process instruments sequentially to bound memory;
do not configure parallel BTC/ETH instances. Temporarily disable the Railway cron to
pause research; do not change ingest or decide schedules.

Useful flags:

- repeat `--asset` to restrict catalog instruments;
- `--lookback-days 730` controls the complete historical range;
- `--maximum-candidates` controls the simulation budget;
- `--force-recompute` reruns analysis while preserving idempotent persistence;
- `--dry-run` validates instruments, range, and budget without loading or persisting
  the full analysis.

Automatic promotion, parallel wallets, real-money trading, and frontend presentation
of the new additive fields remain outside this change.
