from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from itertools import pairwise

import numpy as np
from numpy.typing import DTypeLike


class VariableCategory(StrEnum):
    PRICE_MOVEMENT = "price_movement"
    TREND = "trend"
    BREAKOUT_RANGE = "breakout_range"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CANDLE_SHAPE = "candle_shape"
    TIME_CONTEXT = "time_context"


@dataclass(frozen=True, slots=True)
class VariableDefinition:
    name: str
    description: str
    category: VariableCategory
    lookback: timedelta
    output_type: DTypeLike
    calculation: str
    calculation_version: int = 1
    higher_is_directional: bool | None = None


PRICE_PERIODS = (30, 60, 120, 240, 480, 960, 1440, 2880, 10080)
TREND_PERIODS = PRICE_PERIODS
RANGE_PERIODS = (60, 240, 720, 1440, 10080)
VOLATILITY_PERIODS = (60, 240, 720, 1440, 10080)
VOLUME_PERIODS = (60, 240, 1440, 10080)
POSITIVE_FRACTION_PERIODS = (60, 240, 1440)


def _slug(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 1440:
        return f"{minutes // 60}h"
    return f"{minutes // 1440}d"


def _variable(
    name: str,
    description: str,
    category: VariableCategory,
    minutes: int,
    calculation: str,
    directional: bool | None = None,
) -> VariableDefinition:
    return VariableDefinition(
        name,
        description,
        category,
        timedelta(minutes=minutes),
        np.dtype(np.float32),
        calculation,
        higher_is_directional=directional,
    )


def build_variable_registry() -> tuple[VariableDefinition, ...]:
    variables: list[VariableDefinition] = []
    for minutes in PRICE_PERIODS:
        label = _slug(minutes)
        variables.append(
            _variable(
                f"return_{label}",
                f"Ordinary close-price return over {label}.",
                VariableCategory.PRICE_MOVEMENT,
                minutes + 5,
                "return",
                True,
            )
        )
    for minutes in TREND_PERIODS:
        label = _slug(minutes)
        variables.extend(
            (
                _variable(
                    f"ema_distance_{label}",
                    f"Close distance from the {label} EMA.",
                    VariableCategory.TREND,
                    minutes,
                    "ema_distance",
                    True,
                ),
                _variable(
                    f"ema_slope_{label}",
                    f"One-candle change in the {label} EMA.",
                    VariableCategory.TREND,
                    minutes + 5,
                    "ema_slope",
                    True,
                ),
                _variable(
                    f"trend_volatility_normalized_{label}",
                    f"EMA distance normalized by {label} return volatility.",
                    VariableCategory.TREND,
                    minutes,
                    "trend_normalized",
                    True,
                ),
            )
        )
    for fast, slow in pairwise(TREND_PERIODS):
        variables.append(
            _variable(
                f"ema_spread_{_slug(fast)}_{_slug(slow)}",
                f"Difference between {_slug(fast)} and {_slug(slow)} EMAs.",
                VariableCategory.TREND,
                slow,
                f"ema_spread:{fast}",
                True,
            )
        )
    for minutes in RANGE_PERIODS:
        label = _slug(minutes)
        for suffix, description, calculation, directional in (
            ("previous_high", "Previous rolling highest high.", "previous_high", None),
            ("previous_low", "Previous rolling lowest low.", "previous_low", None),
            (
                "range_position",
                "Close position within the previous rolling range.",
                "range_position",
                True,
            ),
            (
                "distance_from_high",
                "Normalized distance below the previous rolling high.",
                "distance_from_high",
                True,
            ),
            (
                "distance_from_low",
                "Normalized distance above the previous rolling low.",
                "distance_from_low",
                True,
            ),
            (
                "breakout_high",
                "One when close exceeds the previous rolling high.",
                "breakout_high",
                True,
            ),
            (
                "breakout_low",
                "One when close is below the previous rolling low.",
                "breakout_low",
                True,
            ),
        ):
            variables.append(
                _variable(
                    f"{suffix}_{label}",
                    f"{description} Window: {label}",
                    VariableCategory.BREAKOUT_RANGE,
                    minutes + 5,
                    calculation,
                    directional,
                )
            )
    for minutes in VOLATILITY_PERIODS:
        label = _slug(minutes)
        variables.extend(
            (
                _variable(
                    f"return_volatility_{label}",
                    f"Standard deviation of 5-minute returns over {label}.",
                    VariableCategory.VOLATILITY,
                    minutes + 5,
                    "return_volatility",
                ),
                _variable(
                    f"average_normalized_range_{label}",
                    f"Average high-low range divided by close over {label}.",
                    VariableCategory.VOLATILITY,
                    minutes,
                    "normalized_range",
                ),
                _variable(
                    f"volatility_change_{label}",
                    f"Change between current and preceding {label} volatility windows.",
                    VariableCategory.VOLATILITY,
                    minutes * 2 + 5,
                    "volatility_change",
                ),
            )
        )
    for short, long in pairwise(VOLATILITY_PERIODS):
        variables.append(
            _variable(
                f"volatility_ratio_{_slug(short)}_{_slug(long)}",
                f"{_slug(short)} volatility divided by {_slug(long)} volatility.",
                VariableCategory.VOLATILITY,
                long + 5,
                f"volatility_ratio:{short}",
            )
        )
    for minutes in VOLUME_PERIODS:
        label = _slug(minutes)
        variables.extend(
            (
                _variable(
                    f"relative_volume_{label}",
                    f"Current volume divided by its previous {label} baseline.",
                    VariableCategory.VOLUME,
                    minutes + 5,
                    "relative_volume",
                ),
                _variable(
                    f"aggregated_volume_change_{label}",
                    f"Change in aggregated volume versus the preceding {label} window.",
                    VariableCategory.VOLUME,
                    minutes * 2,
                    "volume_change",
                ),
                _variable(
                    f"normalized_aggregated_volume_{label}",
                    "Aggregated volume normalized by its preceding window.",
                    VariableCategory.VOLUME,
                    minutes * 2,
                    "normalized_volume",
                ),
                _variable(
                    f"signed_price_volume_{label}",
                    f"Mean signed 5-minute return multiplied by volume over {label}.",
                    VariableCategory.VOLUME,
                    minutes + 5,
                    "signed_price_volume",
                    True,
                ),
            )
        )
    for name, description, calculation in (
        ("candle_body", "Absolute candle body normalized by open.", "candle_body"),
        ("candle_range", "Full candle range normalized by open.", "candle_range"),
        ("upper_wick", "Upper wick normalized by open.", "upper_wick"),
        ("lower_wick", "Lower wick normalized by open.", "lower_wick"),
        (
            "candle_close_position",
            "Close position inside the candle; zero-range candles use 0.5.",
            "candle_close_position",
        ),
    ):
        variables.append(
            _variable(name, description, VariableCategory.CANDLE_SHAPE, 5, calculation)
        )
    for minutes in POSITIVE_FRACTION_PERIODS:
        variables.append(
            _variable(
                f"positive_candle_fraction_{_slug(minutes)}",
                f"Fraction of positive candles over {_slug(minutes)}.",
                VariableCategory.CANDLE_SHAPE,
                minutes,
                "positive_fraction",
                True,
            )
        )
    for name, description, calculation in (
        ("utc_time_sin", "Sine encoding of UTC time of day.", "utc_time_sin"),
        ("utc_time_cos", "Cosine encoding of UTC time of day.", "utc_time_cos"),
        ("utc_weekday_sin", "Sine encoding of UTC day of week.", "utc_weekday_sin"),
        ("utc_weekday_cos", "Cosine encoding of UTC day of week.", "utc_weekday_cos"),
    ):
        variables.append(
            _variable(name, description, VariableCategory.TIME_CONTEXT, 5, calculation)
        )
    return tuple(sorted(variables, key=lambda item: item.name))


VARIABLE_REGISTRY = build_variable_registry()
MAXIMUM_HISTORICAL_LOOKBACK = max(item.lookback for item in VARIABLE_REGISTRY)
assert len({item.name for item in VARIABLE_REGISTRY}) == len(VARIABLE_REGISTRY)
