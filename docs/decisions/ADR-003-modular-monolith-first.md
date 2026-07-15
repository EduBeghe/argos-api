# ADR-003: Modular Monolith First

## Status

Accepted

## Context

The MVP must validate the complete research loop while preserving clear boundaries between observation, analysis, decision generation, paper execution, and evaluation. Premature distribution would add operational complexity before those boundaries are proven.

## Decision

Build the initial API as a modular monolith with explicit domain-module boundaries. Do not preclude later extraction of independently scalable components when evidence supports it.

## Consequences

The MVP can be developed and operated as one system while keeping responsibilities separated. Module boundaries require discipline, and future extraction may require migration work. Framework and module implementation details remain open decisions.
