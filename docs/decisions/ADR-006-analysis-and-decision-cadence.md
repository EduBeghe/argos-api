# ADR-006: Analysis and Decision Cadence

## Status

Accepted

## Context

Five-minute candles provide enough observations for experimentation without the volume and noise of very short intervals. Recalculating Features and Scores for every closed candle preserves current analytical information.

Allowing a Strategy decision only every 30 minutes reduces excessive reactions to short-term noise and limits unnecessary simulated trading and costs. Candle interval, analysis cadence, and decision cadence are separate concepts even when some values coincide.

## Decision

Use five-minute Market Candles. Run the Feature and Score analysis cycle every five minutes, after each candle is closed and persisted. Run the Strategy decision cycle every 30 minutes.

Decisions use only closed candles and persisted analytical data. Strategies must not be evaluated from partially formed candles. The 30-minute decision cycle is an initial configurable MVP value, not a permanent domain rule.

## Consequences

Each Decision may use up to six newly closed five-minute candles since the previous decision cycle. Scores may change between decision cycles without automatically producing a new Decision. The cadence must be configurable so future Strategies can be evaluated under different intervals.

The MVP does not support multiple simultaneous candle intervals or per-Strategy decision cadences. This decision does not introduce Decision lifecycle states. Decision actions are `OPEN_POSITION`, `INCREASE_POSITION`, `SELL_PARTIAL`, `SELL_ALL`, and `NO_ACTION`; ADR-009 defines which actions the initial Strategy emits.
