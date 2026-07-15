# ADR-004: PostgreSQL

## Status

Accepted

## Context

Argos must persist market observations, analytical metrics and scores, signal reasoning, simulated decisions, paper trades, positions, and evaluation results with reliable relationships and auditability.

## Decision

Use PostgreSQL as the primary persistent data store for the initial platform.

## Consequences

The platform gains transactional persistence, relational integrity, and flexible querying across its research trail. Schema design, time-series storage strategy, retention, scaling, hosting, and operational configuration remain open decisions.
