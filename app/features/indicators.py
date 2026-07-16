from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class CalculationPoint:
    candle_index: int
    value: Decimal


def _validate(values: Sequence[Decimal], label: str) -> None:
    if any(not value.is_finite() for value in values):
        raise ValueError(f"{label} values must be finite")


def calculate_ema(closes: Sequence[Decimal], *, period: int) -> list[CalculationPoint]:
    if period < 1:
        raise ValueError("period must be at least 1")
    _validate(closes, "close")
    if len(closes) < period:
        return []
    alpha = Decimal(2) / Decimal(period + 1)
    previous = sum(closes[:period], start=Decimal(0)) / Decimal(period)
    points = [CalculationPoint(period - 1, previous)]
    for index, close in enumerate(closes[period:], start=period):
        previous = alpha * close + (Decimal(1) - alpha) * previous
        points.append(CalculationPoint(index, previous))
    return points


def calculate_rsi(closes: Sequence[Decimal], *, period: int) -> list[CalculationPoint]:
    if period < 1:
        raise ValueError("period must be at least 1")
    _validate(closes, "close")
    if len(closes) <= period:
        return []
    changes = [current - previous for previous, current in pairwise(closes)]
    gains = [max(change, Decimal(0)) for change in changes]
    losses = [max(-change, Decimal(0)) for change in changes]
    average_gain = sum(gains[:period], start=Decimal(0)) / Decimal(period)
    average_loss = sum(losses[:period], start=Decimal(0)) / Decimal(period)

    def value() -> Decimal:
        if average_gain == 0 and average_loss == 0:
            return Decimal(50)
        if average_loss == 0:
            return Decimal(100)
        if average_gain == 0:
            return Decimal(0)
        ratio = average_gain / average_loss
        return Decimal(100) - Decimal(100) / (Decimal(1) + ratio)

    points = [CalculationPoint(period, value())]
    for index in range(period, len(changes)):
        average_gain = (average_gain * Decimal(period - 1) + gains[index]) / Decimal(
            period
        )
        average_loss = (average_loss * Decimal(period - 1) + losses[index]) / Decimal(
            period
        )
        points.append(CalculationPoint(index + 1, value()))
    return points


def calculate_atr(
    highs: Sequence[Decimal],
    lows: Sequence[Decimal],
    closes: Sequence[Decimal],
    *,
    period: int,
) -> list[CalculationPoint]:
    if period < 1:
        raise ValueError("period must be at least 1")
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("price inputs must have equal lengths")
    _validate([*highs, *lows, *closes], "price")
    if any(high < low for high, low in zip(highs, lows, strict=True)):
        raise ValueError("high must be greater than or equal to low")
    if len(closes) < period:
        return []
    ranges = [highs[0] - lows[0]]
    ranges.extend(
        max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        )
        for index in range(1, len(closes))
    )
    previous = sum(ranges[:period], start=Decimal(0)) / Decimal(period)
    points = [CalculationPoint(period - 1, previous)]
    for index in range(period, len(ranges)):
        previous = (previous * Decimal(period - 1) + ranges[index]) / Decimal(period)
        points.append(CalculationPoint(index, previous))
    return points


def calculate_return(
    closes: Sequence[Decimal], *, period: int
) -> list[CalculationPoint]:
    if period < 1:
        raise ValueError("period must be at least 1")
    _validate(closes, "close")
    points: list[CalculationPoint] = []
    for index in range(period, len(closes)):
        denominator = closes[index - period]
        if denominator == 0:
            raise ValueError("return denominator must not be zero")
        points.append(CalculationPoint(index, closes[index] / denominator - Decimal(1)))
    return points


def calculate_relative_volume(
    volumes: Sequence[Decimal], *, period: int
) -> list[CalculationPoint]:
    if period < 1:
        raise ValueError("period must be at least 1")
    _validate(volumes, "volume")
    if any(volume < 0 for volume in volumes):
        raise ValueError("volume must be non-negative")
    points: list[CalculationPoint] = []
    for index in range(period, len(volumes)):
        baseline = sum(volumes[index - period : index], start=Decimal(0)) / Decimal(
            period
        )
        if baseline != 0:
            points.append(CalculationPoint(index, volumes[index] / baseline))
    return points
