# Versioned ML Search Campaigns

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

Deploy migrations through head, redeploy the API, and keep one `argos-research`
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
its scientific plan and any already persisted reusable evaluation rows. For an
economic-v2 child persisted with the former mixed confidence contract, reset
canonicalizes only the legacy bucket identity and recomputes the experiment key while
preserving the run, campaign, family, and child identifiers. Concurrent retry requests
are serialized; only one may requeue the campaign.

## Economic policy search v2

`economic_policy_search_v2` is independently versioned and leaves every v1 request,
fingerprint, row, and lifecycle rule unchanged. Its diversified track contains 432
families per source: three engines, twelve horizons, three objectives, and four
feature policies. Optional seed families add deterministic local horizon variants.
Each seeded-local child starts from the seed evaluation's exact canonical
hyperparameters and searches only that point plus its immediate lower and upper
canonical neighbors. Invalid or incompatible seed evidence is rejected before
campaign rows are created. The complete source universe remains capped at 600
families.

Each fitted model evaluates 45 policies without refitting: five training-derived
entry-confidence fractions, three exact five-minute holding multipliers, and three
training-fitted regime policies. Economic promotion uses seven development folds and
retains objective diversity without promoting a family that failed its stage gate.
The five economic confidence fractions (`0.05`, `0.10`, `0.20`, `0.30`, and `0.40`)
belong only to that policy grid. Child refinement requests retain the narrower legacy
confidence-bucket identity (`0.05`, `0.10`, and `0.20`) used by refinement evidence
and protected validation.

The temporal protocol is immutable and half-open:

- development/search: `[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`;
- qualification: `[2024-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`;
- reserved final evidence: `[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`.

Only frozen development candidates are qualified on calendar 2024. Development rows
are purged by the authoritative seven-day split gap, and both the feature boundary
and its horizon endpoint must precede the effective development cutoff.
Qualification uses only boundaries whose complete target endpoint is before 2025.
The persisted temporal manifest records requested and effective boundaries, purge
counts, horizon, columns, and the exact consumed-data fingerprint.

Qualification is stored in `ml_economic_qualifications`; it never writes
protected-final, deployment, Decision, or Paper state. Before any market read, the
worker revalidates the complete frozen scientific identity against the source,
family, evaluation, completed refinement run, policy, costs, contract versions, and
temporal boundaries. Exact identities may be reused across campaigns; any mismatch
fails closed.

Stage 5 is a dedicated qualification lifecycle. After qualification rows are queued,
the generic refinement scheduler is no longer entered. Repeated coordination is
idempotent while work is queued or running, and the campaign terminates only after
all qualifications are terminal. `qualified_candidates_found`,
`no_qualified_candidate`, and `no_strong_candidate` remain distinct conclusions.

Qualification workers maintain an independent database heartbeat during both
dataset loading and CPU evaluation. Publication and failure updates require the same
worker lease. A stale worker that lost ownership cannot publish evidence or overwrite
the new owner; cancellation is also checked at the ownership boundary. The shared
boundary guard rejects every v2 read extending past
`2025-01-01T00:00:00Z`.

Before releasing the database read transaction, the worker materializes the selected
evaluation as an immutable `PlannedEvaluation` containing only normalized scientific
values. Background CPU calculation receives that snapshot, the fully materialized
dataset, and the frozen policy; it never receives an ORM record, session, connection,
or lazy query result. This keeps the long calculation outside a database transaction
without allowing rollback-expired ORM state to cross the thread boundary.

Discover both preset contracts at
`GET /optimization/ml-search-campaign-contract/v2`. Qualification evidence is
read-only and paginated at
`GET /optimization/ml-search-campaigns/{campaign_id}/qualifications`, with detail at
`GET /optimization/ml-search-campaigns/{campaign_id}/qualifications/{qualification_id}`.
The list accepts a `provenance` filter.

An authenticated operator may submit one through ten explicitly selected discarded
Stage 3 families to
`POST /optimization/ml-search-campaigns/{campaign_id}/diagnostic-qualifications`.
Eligibility is deliberately narrow: the completed v2 campaign must have full
coverage and outcome `no_strong_candidate`, and every selected family must have
failed Stage 3 only because it beat buy-and-hold in two folds instead of the required
three. The request is atomic and idempotent; queued, running, or completed identities
are reused, while failed or cancelled work requires the existing explicit retry
operation. These rows carry provenance `diagnostic_stage3_near_miss`; automatic
development-candidate rows retain `automatic_development_candidate`.

Diagnostic qualification consumes only the untouched half-open interval
`[2024-01-01T00:00:00Z, 2025-01-01T00:00:00Z)`. It cannot read the reserved 2025
interval, change the source campaign or family classification, or imply promotion to
strong development candidate. It uses the same qualification queue, worker, lease,
cost-stress scenarios, and gates as automatic qualification.

The first production enqueue can retain campaign
`cf1fee07-143a-4759-bcc4-f662f03a7aa9` only as provenance and use its seed family,
while the current completed source screening run IDs remain operator inputs:

```sh
python -m app.jobs.enqueue_ml_search_campaign \
  --source-screening-run-id <current-btc-screening-run-id> \
  --source-screening-run-id <current-eth-screening-run-id> \
  --preset economic_policy_search_v2 \
  --seed-family-id a43c2c2b-d04c-446e-8cef-0838707c2ed7 \
  --enqueue
```

The seed is research provenance, not final-validation or deployment authorization.

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
