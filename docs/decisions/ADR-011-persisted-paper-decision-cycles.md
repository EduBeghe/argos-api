# ADR-011: Persisted Paper Decision Cycles

## Status

Accepted.

## Context

Decision generation and Paper execution run independently from five-minute market ingestion. Retries and overlapping external cron invocations must not reevaluate a boundary after portfolio state changes or execute the same intent twice.

## Decision

Each instrument and active Paper Trading run has one persisted Paper Decision Cycle for a scheduled UTC 30-minute boundary. Its natural identity includes run, instrument, strategy and configuration versions, and boundary. The cycle records its source Score, Decision Snapshot, optional Paper Trade, action, and completion timestamps. NO_ACTION is a completed auditable cycle without a Trade.

The external job runs at UTC minute 02 and 32 using cron `2,32 * * * *`. It selects the latest compatible Score backed by a candle closed at or before the corresponding :00 or :30 boundary. A cycle may start from its boundary through ten minutes afterward, inclusively; a later invocation fails closed instead of executing at an unrealistic old price. Evidence age is a separate rule: it measures when the analytical input's candle closed relative to the boundary, while cycle-start delay measures when Paper execution begins relative to that boundary. Evidence older than ten minutes at the boundary is stale and fails closed. These limits give the separate five-minute ingestion job time to persist its boundary evidence without allowing delayed execution.

PostgreSQL uniqueness claims each cycle. A committed cycle makes retries no-ops; a transaction failure rolls back its claim so a retry can recover. BTC/USDT and ETH/USDT use independent transactions, so one instrument failure does not undo another instrument's completed cycle. Scheduling is at-least-once and not exactly-once.

## Consequences

Cycles are reproducible and safe under overlapping Railway executions without an internal scheduler or distributed lock. The job contacts no provider or exchange and never moves real money. Future instruments can extend the injected instrument collection without changing cycle identity semantics.

