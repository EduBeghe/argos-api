# ADR-002: Separate Web and API Repositories

## Status

Accepted

## Context

Argos needs an analytical backend and a web dashboard. They have distinct responsibilities and may evolve independently.

## Decision

Maintain the API and web dashboard in separate repositories. The API repository owns ingestion, analysis, simulated decisions, paper-trade execution, persistence, evaluation, and the interface exposed to the web application. The web repository owns dashboard presentation and interaction.

## Consequences

Each repository can evolve and be reviewed independently. Their integration requires an explicit, versioned interface and coordinated changes when that interface evolves. Specific interface technologies remain an open decision.
