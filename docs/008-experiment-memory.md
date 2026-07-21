# Experiment Memory and Experience-Guided Search

Argos can persist compact historical refinement evidence and use compatible completed
experiments to plan later searches. Persistence is not learning by itself: the planner
must load version-compatible candidate profiles before selecting its bounded candidate
set. Running and failed experiments never contribute evidence.

## Compatibility and identity

`experiment_key` is a deterministic SHA-256 identity over the dataset fingerprint,
complete instrument and requested range, chronological split boundaries, score,
Strategy, simulation-engine and planner versions, initial cash, costs, decision cadence,
budget, and planner allocation. Any changed input produces another experiment. An exact
completed experiment is reused without simulation unless `--force-recompute` requests a
determinism check.

Evaluation reuse additionally requires the same candidate, dataset, phase and segment
boundaries, costs, cash, model/Strategy/engine versions, and cadence. Evidence from a
different window may prioritize work but cannot replace current-window evaluation.
`history_cutoff` is captured before planning, preventing later results from entering the
current plan.

Candidate definitions are canonical and unique by the same ordered values used for the
candidate SHA-256. Runs store compact metrics only; they do not duplicate candles,
Features, matrices, Decisions, Trades, or equity curves.

## Candidate profiles

Profile version 1 aggregates deterministic Decimal medians, win and confirmation rates,
worst drawdowns, activity, fees, evaluated instruments, evidence dates, and consecutive
recent failures. Compatible evidence must use the same score, Strategy, simulation
engine, and profile versions. The classifications are:

- `INSUFFICIENT_EVIDENCE`: fewer than two validation windows.
- `PROMISING`: at least two validation windows, win rate at least 0.50, positive median
  validation excess return, and no validation drawdown above 10%.
- `ROBUST_CANDIDATE`: at least three validation and two test windows, validation win rate
  at least 0.66, confirmation rate at least 0.50, positive median validation and test
  excess returns, and test drawdown at most 10%.
- `PAPER_TRIAL_READY`: at least four validation and three test windows, win rate at least
  0.75, confirmation rate at least 0.66, positive median test excess return, test
  drawdown at most 10%, and no two consecutive recent failures.
- `HISTORICALLY_REJECTED`: at least three validation windows with no wins or non-positive
  median excess return, or three consecutive recent failures.

These classifications summarize historical evidence. They do not prove causality,
predict future performance, or guarantee profitability.

## Search planning

Planner version 1 assigns the joint budget to known rechecks (15%), neighborhoods around
promising candidates (50%), uncertain rechecks (15%), and novel exploration (20%).
Allocations are Decimal values that sum exactly to one; integer residual slots are
distributed deterministically after reserving the first slot for the production
baseline. A budget of one therefore evaluates only the baseline. Larger budgets retain
a positive novel-exploration allocation. Candidates are de-duplicated, including when
the baseline also appears in history or the novel pool, and the total never exceeds the
configured budget.

Neighborhoods move 0.05 directional weight between two signals while preserving an
exact sum of one, vary volume confirmation by 0.125, or adjust one Strategy dimension by
0.05/0.05/5 percentage points. Invalid combinations are discarded. The planner does not
generate the full neighborhood Cartesian product.

Rejected candidates lose priority but are not permanently forbidden. A later version,
the configured reconsideration age, a neighborhood of a robust region, or novel
exploration can make them eligible for new evidence.

With no compatible history, the refiner uses its original deterministic screening and
joint-search behavior. Test evidence from the current experiment remains outside
selection.

## Operation

The default command remains read-only. It may read compatible history for planning but
writes nothing:

```sh
python -m app.jobs.refine_historical_strategy <arguments>
```

Persist one atomic completed experiment with:

```sh
python -m app.jobs.refine_historical_strategy <arguments> --persist
```

Repeating the exact command reuses the completed experiment and reports
`reused_existing_experiment=true`. Verify determinism without creating a duplicate with:

```sh
python -m app.jobs.refine_historical_strategy <arguments> --persist --force-recompute
```

A mismatch fails explicitly. Failed runs retain only sanitized failure state; partial
evaluations never become valid evidence. PostgreSQL uniqueness resolves concurrent
claims without distributed locks.

A `running` experiment is an execution lease for two hours. An identical invocation
inside that timeout fails closed as concurrent work. After the timeout, PostgreSQL
atomically reclaims the same row only when both its `running` state and previously
observed `started_at` still match. Reclamation refreshes `started_at` and clears terminal
timestamps and failure text. This compare-and-set rule permits one recovery process,
preserves the experiment identity, and prevents a killed process from blocking the
experiment indefinitely.

The claimed `started_at` is also the execution's ownership token. Before writing any
candidate, evaluation, or benchmark evidence, the owner locks the matching `running`
row in the same PostgreSQL transaction. Completion and failure transitions require the
same run identity, state, and token and treat a zero-row update as an explicit lost
lease. Consequently, a superseded process rolls back all attempted evidence and cannot
complete or fail the recovered owner's execution.

Inspect row counts and combined table/index storage with:

```sh
python -m app.jobs.refinement_storage
```

At roughly 900 compact evaluations per asset per week, storage is expected to remain in
the low megabytes per run, depending on PostgreSQL JSONB and index overhead. The
diagnostic command provides the actual deployment-specific measurement.

Version 1 is single-instrument and sequential. It adds no cron, public endpoint,
frontend, automatic promotion, new Strategy, sentiment/news/ML signal, or Railway
configuration.

## Read-only API

Persisted memory is available to frontend consumers through four bounded read-only
resources:

- `GET /optimization/refinement-runs` lists experiments newest first and supports
  complete instrument identity, lifecycle, recommendation, and UTC start filters.
- `GET /optimization/refinement-runs/{run_id}` composes the typed run summary, search
  plan, segment benchmarks, recommendation, digest, and grouped evaluation counts.
- `GET /optimization/refinement-runs/{run_id}/evaluations` pages compact evaluations
  without materializing an unbounded experiment history in the run detail.
- `GET /optimization/candidates` lists canonical parameters with profiles calculated by
  the same versioned `experience.py` rules used by planning.
- `GET /optimization/candidates/{stable_candidate_id}` returns the canonical definition,
  current profile, chronological paginated evidence, related run identities, and
  recommendation attribution.

Run summaries use `limit` and `offset`. Candidate collections and evaluation histories
use deterministic keyset cursors with `has_more` and `next_cursor`; every page has a
maximum limit of 200. Classification filters are composed over bounded candidate batches
until a complete public page or the real end of the dataset is reached, so there is no
silent global scan cap. Profile evidence uses the same centrally defined compatible
phases and versions as optimization planning. Queries are data-driven by full instrument
identity and avoid one query per candidate. Decimal values and UTC
timestamps remain exact. The contract maps stored budgets, allocations, plans, and
parameters into explicit versionable schemas; it does not expose raw JSONB, candles,
Features, matrices, or unsafe failure details. GET requests never write, start a
refinement, or contact a provider. Unknown run and candidate identities return 404.

## Walk-forward stability evaluation

The original deterministic 60/20/20 split remains the primary refinement pass.
Optional walk-forward evaluation complements it with several chronological, half-open
training, validation, and test windows. Features are calculated once in memory from
prior candles. Algorithm version 2 runs bounded candidate generation and training
screening independently inside each window, ranks candidates only with that window's
validation matrix, and evaluates only the selected candidates on its test matrix.
Global refinement validation/test evidence, test outcomes, and later windows cannot
influence an earlier selection.

Version 2 defaults to 180 training days, 60 validation days, 60 test days, a
60-day step, at most six windows, and at most ten candidates. CLI flags expose every
duration and limit, plus minimum evidence, profitable-window ratio, drawdown, and
return-dispersion thresholds. Insufficient history is persisted explicitly and never
produces a recommendation.

When more valid windows exist than the configured limit, only the most recent windows
are built and retained, with ordinals normalized to `1..N`. Each window stores UTC
bounds, decision count, a selection fingerprint over training/validation evidence and
configuration, and a full fingerprint that additionally includes isolated test
evidence. Candidate selections preserve validation rank, selection reason, validation
metrics, and whether test evaluation was authorized. Nonselection is not an evaluation
failure and contributes no stability evidence.
Candidate evaluations preserve net return, P&L, drawdown, ending equity, fees, trades,
and return over drawdown. Stability requires multiple profitable windows, bounded
drawdown and dispersion, and trading activity; one exceptional window cannot establish
stability. Production baseline and buy-and-hold reuse the exact refinement simulation,
fees, slippage, and cash mathematics.

The complete configuration and source refinement run form the experiment identity.
Repeating an identical completed evaluation reuses persisted evidence and reports
computed, reused, and avoided counts. Main metrics remain relational and indexed.

Run manually with:

```sh
python -m app.jobs.refine_historical_strategy \
  --asset BTC/USDT --provider binance --market-type spot --interval 5m \
  --from 2024-01-01T00:00:00Z --to 2026-01-01T00:00:00Z \
  --initial-cash 10000 --fee-rate 0.001 --slippage-rate 0.0005 \
  --persist --walk-forward \
  --walk-forward-training-days 180 \
  --walk-forward-validation-days 60 \
  --walk-forward-test-days 60 \
  --walk-forward-step-days 60 \
  --walk-forward-max-windows 6 \
  --walk-forward-max-candidates 10
```

Read-only consumers use:

- GET /optimization/refinement-runs/{run_id}/walk-forward
- GET /optimization/walk-forward-runs/{walk_forward_run_id}
- GET /optimization/walk-forward-runs/{walk_forward_run_id}/candidates
- GET /optimization/walk-forward-runs/{walk_forward_run_id}/selections
- GET /optimization/walk-forward-runs/{walk_forward_run_id}/candidates/{candidate_id}/windows

Candidate and window evidence use stable keyset cursors and a maximum page size of 200.
GET requests never run simulations or contact providers. Walk-forward is decision
support, not proof of causality or profitability, and never promotes a candidate or
changes active Paper Trading. Weekly execution, explicit manual promotion into an
immutable Strategy version, and an isolated parallel wallet remain future work.
