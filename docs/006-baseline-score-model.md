# Baseline Technical Score Model

`baseline_technical`, version 1, is a deterministic rule-based analytical model. It is not AI, a probability of success, or a trading instruction.

## Inputs

A score requires the same-candle version-1 values for EMA 20, EMA 50, RSI 14, ATR 14, return 1, return 12, and relative volume 20, plus that candle's close price. Incomplete feature sets are skipped.

## Formula

- `trend = clamp((ema_20 - ema_50) / atr_14, -1, 1)`
- `rsi_signal = clamp((rsi_14 - 50) / 20, -1, 1)`
- `return_1_signal = clamp(return_1 / 0.0025, -1, 1)`
- `return_12_signal = clamp(return_12 / 0.01, -1, 1)`
- `momentum = 0.35 Ã— rsi_signal + 0.20 Ã— return_1_signal + 0.45 Ã— return_12_signal`
- `volume_activity = clamp((relative_volume_20 - 0.5) / 1.5, 0, 1)`
- `volatility_ratio = atr_14 / close_price`
- `directional = 0.55 Ã— trend + 0.45 Ã— momentum`
- `volume_confirmation = 0.75 + 0.25 Ã— volume_activity`
- `overall = directional Ã— volume_confirmation`
- `signal_strength = abs(overall)`

Persisted contributions are `0.55 Ã— trend`, `0.45 Ã— 0.35 Ã— rsi_signal`, `0.45 Ã— 0.20 Ã— return_1_signal`, and `0.45 Ã— 0.45 Ã— return_12_signal`. They sum exactly to the directional score.

## Interpretation

- `strongly_bearish`: score at or below -0.60
- `bearish`: above -0.60 through -0.20
- `neutral`: above -0.20 and below 0.20
- `bullish`: 0.20 through below 0.60
- `strongly_bullish`: 0.60 or above

Explanations use deterministic templates for trend, momentum, volume, and volatility. Volume confirms magnitude but provides no direction; volatility is measured but does not alter direction.

## Limitations

The constants are initial hypotheses, not empirically optimized. The model has not yet been backtested. Signal strength is neither probability nor statistical confidence, and profitability is not claimed.
