from collections.abc import Sequence
from decimal import Decimal


def calculate_ema(closes: Sequence[Decimal], *, period: int) -> list[Decimal]:
    """Calculate valid EMA values, seeded by the first period's SMA."""

    if period < 1:
        raise ValueError("period must be at least 1")
    if any(not value.is_finite() for value in closes):
        raise ValueError("close values must be finite")
    if len(closes) < period:
        return []

    alpha = Decimal(2) / Decimal(period + 1)
    previous = sum(closes[:period], start=Decimal(0)) / Decimal(period)
    values = [previous]
    for close in closes[period:]:
        previous = alpha * close + (Decimal(1) - alpha) * previous
        values.append(previous)
    return values
