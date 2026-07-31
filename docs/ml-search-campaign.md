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

For rollback, stop the research worker and API version using the new endpoints, then
downgrade one Alembic revision. The downgrade removes only campaign tables and does
not delete historical screening or refinement evidence.
