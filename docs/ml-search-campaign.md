# Exhaustive OHLCV ML Search Campaign

`exhaustive_ohlcv_v1` is a durable coordinator over the existing ML refinement
queue. It does not train in HTTP requests and does not read protected final-test
evidence. All calculations remain restricted to development data before
`2024-01-01T00:00:00Z`.

For each completed screening source, the campaign creates exactly 3,600 immutable
families: three engines, all 80 registered future durations, three objectives, and
five feature policies (`all`, and top 16, 32, 64, or 96 variables). It gives every
family one evaluation before deepening any family. Economically surviving families
then receive cumulative budgets of 4, 16, and 64 configurations. Compatible earlier
evaluations are reused.

Child refinement runs are shared by exact experiment identity. Multiple campaigns may
reference the same queued, running, or complete child run; each campaign records that
reuse independently without claiming that the shared computation was performed again.
Cancelling one campaign never cancels a child still required by another active
campaign.

Promotion follows the versioned `economic_successive_halving_v1` gates exposed by
`GET /optimization/ml-search-campaign-contract`. Predictive accuracy alone cannot
promote a family. The scientific outcomes are `coverage_incomplete`,
`search_budget_exhausted`, `no_strong_candidate`, and
`strong_candidates_found`. The last label means strong development evidence only;
it does not authorize Strategy simulation, Paper Trading, or production activation.

## Railway operation

Deploy migration `20260731_25`, redeploy the API, and keep one `argos-research`
replica running the existing worker:

```sh
python -m app.jobs.process_ml_refinement_queue
```

Preview without writing by using `--dry-run`; enqueue durable work with `--enqueue`:

```sh
python -m app.jobs.enqueue_ml_search_campaign \
  --source-screening-run-id 00000000-0000-0000-0000-000000000000 \
  --preset exhaustive_ohlcv_v1 --dry-run
```

The command may exit after enqueue. PostgreSQL retains the campaign, family state,
child linkage, budgets, and conclusions. Monitor the campaign and its bounded family
pages through `/optimization/ml-search-campaigns`. Cancellation is cooperative when a
child is running. Only operationally failed campaigns may be retried; scientific
outcomes are terminal. Retry preserves campaign progress and identifiers. When the
responsible child refinement is failed, cancelled, or structurally incomplete, retry
atomically resets that same refinement run to a clean queued lease while preserving
its immutable experiment identity and any already persisted reusable evaluation rows.
Concurrent retry requests are serialized; only one may requeue the campaign.

## Development economic re-audit

A completed `exhaustive_ohlcv_v1` campaign can be diagnosed without starting a new
search or reading protected evidence:

```sh
python -m app.jobs.audit_ml_search_near_misses \
  --campaign-id cf1fee07-143a-4759-bcc4-f662f03a7aa9 \
  --maximum-candidates 15 \
  --output development-reaudit.json
```

The command is read-only. It reconstructs the complete persisted family funnel, then
selects a deterministic prefix of discarded stage-4 direction-classification
families. Eligibility requires complete finite evidence, at least 500 observations,
five positive-net and five favorable-direction folds, four folds beating the
benchmark, positive median net return, worst net return of at least `-0.02`, and a
canonical persisted confidence bucket. The maximum candidate count is bounded from
1 through 25.

Before loading candles, the audit verifies that every selected family's immutable
identity agrees with its best evaluation, stage child, completed refinement run, and
screening source. This includes the instrument and dataset fingerprint, engine,
objective, horizon, canonical feature policy, hyperparameters, effective evaluation
fingerprint, contract versions, costs, and deterministic seed. Missing records,
incomplete runs, or any disagreement abort the whole report as structural corruption;
they are never skipped or treated as an economic failure.

Each selected candidate is refit independently for the four accumulating-history
validation blocks (`2022H1`, `2022H2`, `2023H1`, and `2023H2`). Every model, feature
selection, and score cutoff uses only that block's training rows. Validation uses UTC
30-minute decisions and the preceding closed five-minute candle. Each half-year is
simulated independently from initial cash at 1x, 1.5x, and 2x configured costs and is
force-liquidated at its block end. Returns are compounded with `Decimal`; drawdown is
the maximum individual block drawdown. The benchmark is informational only.

The loader uses the half-open development range `[2020-01-01, 2024-01-01)` and never
reads the 2024-2026 protected period. Output is canonical JSON with no timestamps or
random identifiers added at runtime, so unchanged persisted evidence produces
byte-identical reports. Without `--output`, stdout contains only the JSON document;
logs use stderr. With `--output`, the UTF-8 report is written to that path.

This can be CPU-intensive because each selected immutable configuration is fitted
four times and simulated at three cost levels. Run it as a one-shot Railway research
command with sufficient memory. A `development_proxy_pass` only means that this
development diagnostic met the current final-policy arithmetic gates. It is not
protected validation, a recommendation, a deployment authorization, or evidence of
future profitability.

For rollback, stop the research worker and API version using the new endpoints, then
downgrade one Alembic revision. The downgrade removes only campaign tables and does
not delete historical screening or refinement evidence.

