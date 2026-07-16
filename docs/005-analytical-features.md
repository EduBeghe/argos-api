# Analytical Feature Reference

All features use Decimal arithmetic, Binance Spot five-minute candles, and version 1. A value belongs to the candle at its stated index and uses no later data. These are measurements, not trading decisions or claims of profitability.

| Feature | Formula | Inputs | First eligible candle | Intuition | Principal limitation |
| --- | --- | --- | --- | --- | --- |
| EMA 20 | SMA seed over 20 closes; then `EMA = (2 / 21) Ã— close + (19 / 21) Ã— prior EMA` | close | index 19 | Smoothed recent price level | Lagging and sensitive to the selected period |
| EMA 50 | SMA seed over 50 closes; then `EMA = (2 / 51) Ã— close + (49 / 51) Ã— prior EMA` | close | index 49 | Slower price trend context | Responds slowly to regime changes |
| RSI 14 | Wilder averages of 14 close changes; `100 - 100 / (1 + average gain / average loss)` | close | index 14 | Balance of recent gains and losses | Extreme values do not guarantee reversal |
| ATR 14 | Wilder average of true range, where true range includes intrabar range and gaps from the prior close | high, low, close | index 13 | Absolute price volatility | Not directional and scale-dependent |
| Return 1 | `close[t] / close[t-1] - 1` | close | index 1 | One-candle fractional price change | No context beyond one interval |
| Return 12 | `close[t] / close[t-12] - 1` | close | index 12 | One-hour fractional price change | Endpoint-only; ignores the path between closes |
| Relative volume 20 | `quote volume[t] / average(quote volume[t-20:t])` | quote-asset volume | index 20 | Current activity relative to the prior 20 candles | Omitted when the historical baseline is zero |

Returns are fractions: `0.01` means +1% and `-0.02` means -2%. Relative volume `1` matches its historical baseline, while `1.5` is 50% above it.
