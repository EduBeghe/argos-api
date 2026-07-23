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

Predictive Edge Discovery v4 makes the discovery stage a bounded funnel: at most 120
training relationships, 40 frozen validation configurations, and 12 isolated-test
configurations. Its diagnostic output identifies the bottleneck, strongest observed
signal, rejection counts and requirement margins rather than reducing a zero-candidate
run to “no stable predictive edge”. Stage timings and computed/reused counts make a
slow first execution auditable. Exact reuse remains keyed by data content, instrument,
costs, configuration, profile, and contract versions; v3 evidence is readable but is
not reusable as v4.

The separate v4.1 research contract uses A1/A2/A3/B/C limits of 24 directional
relationships, 8 contexts, 192 generated interactions, 60 retained interactions, 40
validation configurations, and 12 isolated tests. Processing is sequential. Context
regimes and derived predictors are frozen from training, and test cannot change bounds,
action, ranking, or fingerprints. These limits do not imply an economic pattern exists.

Confirmed v4.1 interactions become versioned, context-conditioned
`StrategyHypothesis` records and join confirmed v4 hypotheses before edge-guided
generation and walk-forward evaluation. Complete fingerprints prevent duplicates; no
candidate is activated automatically.

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

The generator considers at most 12 entry hypotheses and 6 exit hypotheses. It expands
each viable entry across compatible timed, signal, and production-baseline exits,
30-minute through six-hour cadences, and 10%, 15%, and 20% allocations. After complete
fingerprint deduplication, the economic pre-screen retains at most 60 Strategies.
Selection is deterministic and stratified across every available family, then across
hypotheses, cadences, and allocations. Missing families donate capacity to the
remaining families; no available family or single prolific hypothesis can monopolize
the 60-candidate generation budget solely because of sort order.

The staged funnel evaluates at most 60 candidates on training, retains 24 using only
training evidence, evaluates those on validation, retains at most 12 qualified
candidates using only validation evidence, and executes isolated test once for at
most the best 5. The configured candidate budget limits that final test set. Test data
never participates in generation or earlier ranking. Ranking is deterministic and a
complete candidate fingerprint is the final tie breaker.

Training qualification requires sufficient trades, positive net return, and acceptable
drawdown. Validation additionally must strictly beat the production Strategy before a
candidate can consume isolated-test budget. Buy-and-hold remains an informative
benchmark rather than an additional qualification gate.

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

Weekly runs cache immutable evidence separately for training, validation, and test.
Each record identifies the complete candidate, stage, exact segment boundaries,
fingerprint of the rows inside that segment, costs, and generation, ranking, and
qualification versions. Complete training and validation evidence is persisted even
when the candidate does not advance, while test remains absent unless it actually ran.
A later week therefore reuses unchanged stages and computes only new or changed
segments; `--force-recompute` bypasses both exact-run and stage reuse. Funnel counters
report computed and reused windows for each stage as well as their totals.

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

Structured output keeps `edge_guided_run_id` and `predictive_edge_run_id` separate and
reports `predictive_edge_v4_conclusion`, `regime_conditioned_v41_conclusion`, and
`strategy_research_outcome`. Exact reuse returns both persisted conclusions, the
original v4.1 stage counters, and current reuse-work counters.

Original counters describe the immutable work that produced the stored evidence;
current counters describe only this invocation. Exact reuse performs no dataset load,
Feature build, edge analysis, or Strategy simulation. Its `compute_avoided_estimate`
sums persisted v4 work, v4.1 A1/A2/A3/validation/test work, and complete Strategy work,
so a costly no-Strategy result still reports positive avoided computation.

Production verification:

```sh
python -m app.jobs.run_weekly_strategy_research --asset BTC/USDT --persist --force-recompute
python -m app.jobs.run_weekly_strategy_research --asset BTC/USDT --persist
curl "$API_BASE_URL/optimization/predictive-edge-runs/$RUN_ID/diagnostics"
curl "$API_BASE_URL/optimization/predictive-edge-runs/$RUN_ID/near-misses?kind=actionable"
curl "$API_BASE_URL/optimization/predictive-edge-runs/$RUN_ID/near-misses?kind=diagnostic"
```

Compare persisted run IDs, conclusions, original counters, and near-miss counts between
the forced and exact-reuse runs. Only current computed/reused/avoided work and elapsed
time should differ.

Useful flags:

- repeat `--asset` to restrict catalog instruments;
- `--lookback-days 730` controls the complete historical range;
- `--maximum-candidates` controls the simulation budget;
- `--force-recompute` reruns analysis while preserving idempotent persistence;
- `--dry-run` validates instruments, range, and budget without loading or persisting
  the full analysis.

Automatic promotion, parallel wallets, real-money trading, and frontend presentation
of the new additive fields remain outside this change.
