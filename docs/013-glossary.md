# Glossary

**Statistical signal:** A measured historical relationship that clears statistical
requirements; it may still be uneconomic.

**Economic edge:** A statistical signal whose expected movement also covers simulated
fees, slippage, and the versioned economic margin.

**Near miss:** A training/validation relationship ranked by explicit requirement
margins. It is diagnostic evidence, not a confirmed or activatable candidate.

**Strategy hypothesis:** An immutable, non-executable interpretation of confirmed
bin-level evidence with its condition, action, range, and limitations.

**Confirmed Strategy:** A frozen research Strategy that passes validation and isolated
test requirements. Confirmation is historical evidence, not a profitability promise.

- **Asset:** the economic symbol being observed, such as BTC/USDT. It is only one part of instrument identity.
- **Instrument:** asset plus provider, market type, and interval.
- **Candle:** open/high/low/close/volume observation for a fixed interval; Argos analyzes only closed candles.
- **Feature:** one versioned measurement calculated from candle history, such as EMA or RSI.
- **Score:** a versioned aggregation of Features into comparable evidence. It is not a trade instruction.
- **Strategy:** deterministic policy that converts compatible evidence and portfolio state into requested intent.
- **Threshold:** configured boundary used by a rule, not a guarantee or learned certainty.
- **Position:** simulated quantity currently or historically held for an instrument.
- **Trade:** immutable simulated execution of a Decision; it records price, quantity, fees, slippage, and cash impact.
- **P&L:** profit and loss. Realized P&L belongs to closed quantity; unrealized P&L marks open quantity.
- **Return:** P&L relative to starting capital, normally represented as a decimal fraction.
- **Drawdown:** decline from a prior equity peak to a later trough.
- **Fee:** simulated percentage cost charged on execution notional.
- **Slippage:** simulated unfavorable difference between observed and execution price.
- **Validation:** evidence used after discovery to decide whether a candidate deserves isolated testing.
- **Test:** later isolated evidence that did not participate in candidate selection.
- **Walk-forward:** repeated chronological training/validation/test windows that preserve time order.
- **Benchmark:** independent reference behavior, such as production Strategy or buy and hold.
- **Candidate:** frozen historical Strategy/Score behavior under evaluation; it is not automatically active.
- **Legacy:** evidence produced under an older contract and retained for audit, but not eligible under the current contract.
- **Fingerprint:** deterministic hash of effective inputs used to detect identical or incompatible evidence.
- **Idempotency:** repeating an operation produces no duplicate identity or repeated financial effect.
