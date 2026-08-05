from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.gaps import datetime_to_ns
from app.ml.protected.simulation import (
    ProtectedPrediction,
    simulate_fixed_horizon_long_only,
)
from app.ml.protected.validation import (
    _signed_positive_scores,
    confidence_score_cutoff,
    is_decision_boundary_ns,
    is_protected_decision_boundary_ns,
)
from app.paper.config import SimulationConfiguration


@pytest.mark.parametrize(
    ("minute", "expected"),
    [(0, True), (5, False), (25, False), (30, True), (35, False)],
)
def test_decision_cadence_is_aligned_to_utc_epoch(minute: int, expected: bool) -> None:
    boundary = datetime(2024, 1, 1, 12, minute, tzinfo=UTC)
    assert is_decision_boundary_ns(datetime_to_ns(boundary)) is expected


def test_protected_cadence_range_is_start_inclusive_and_end_exclusive() -> None:
    assert is_protected_decision_boundary_ns(
        datetime_to_ns(datetime(2024, 1, 1, tzinfo=UTC))
    )
    assert not is_protected_decision_boundary_ns(
        datetime_to_ns(datetime(2026, 1, 1, tzinfo=UTC))
    )


@pytest.mark.parametrize(
    "fraction", [Decimal("0.05"), Decimal("0.10"), Decimal("0.20")]
)
def test_confidence_cutoff_uses_development_absolute_score_quantile(
    fraction: Decimal,
) -> None:
    development_scores = np.linspace(-0.93, 0.81, 41, dtype=np.float64)
    cutoff = confidence_score_cutoff(development_scores, fraction)
    expected = Decimal(
        str(np.quantile(np.abs(development_scores), 1.0 - float(fraction)))
    )
    assert cutoff == expected


def test_five_percent_bucket_is_not_literal_score_threshold_and_preserves_ties() -> (
    None
):
    development_scores = np.asarray([0.01] * 18 + [0.80, -0.80], dtype=np.float64)
    cutoff = confidence_score_cutoff(development_scores, Decimal("0.05"))
    assert cutoff > Decimal("0.05")
    assert cutoff == Decimal("0.80")
    assert sum(abs(Decimal(str(value))) >= cutoff for value in development_scores) == 2


def test_negative_high_confidence_score_never_opens_long_position() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    result = simulate_fixed_horizon_long_only(
        predictions=(
            ProtectedPrediction(start, Decimal("-0.95"), Decimal("100")),
            ProtectedPrediction(
                start + timedelta(minutes=30), Decimal("0.80"), Decimal("101")
            ),
            ProtectedPrediction(
                start + timedelta(minutes=60), Decimal("0.10"), Decimal("102")
            ),
        ),
        instrument=Instrument(
            Asset.BTC_USDT,
            Provider.BINANCE,
            MarketType.SPOT,
            Interval.FIVE_MINUTES,
        ),
        threshold=Decimal("0.80"),
        hold=timedelta(minutes=30),
        configuration=SimulationConfiguration(),
    )
    assert result.trades[0].entry_at == start + timedelta(minutes=30)


@pytest.mark.parametrize("fraction", [Decimal("0"), Decimal("0.15"), Decimal("NaN")])
def test_confidence_cutoff_rejects_unsupported_fraction(fraction: Decimal) -> None:
    with pytest.raises(ValueError, match="unsupported confidence"):
        confidence_score_cutoff(np.asarray([0.2], dtype=np.float64), fraction)


def test_confidence_cutoff_rejects_non_finite_quantile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np, "quantile", lambda *_args, **_kwargs: np.inf)
    with pytest.raises(ValueError, match="finite and between"):
        confidence_score_cutoff(np.asarray([0.2], dtype=np.float64), Decimal("0.05"))


class _IncompatibleClassifier:
    classes_ = np.asarray([-1, 0], dtype=np.int64)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.full((len(values), 2), 0.5, dtype=np.float64)


def test_signed_scores_reject_classifier_without_positive_class() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _signed_positive_scores(
            model=_IncompatibleClassifier(),
            values=np.asarray([[1.0]], dtype=np.float32),
        )


def test_protected_simulation_exits_only_on_next_cadence_boundary() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    predictions = tuple(
        ProtectedPrediction(start + timedelta(minutes=30 * index), score, price)
        for index, (score, price) in enumerate(
            (
                (Decimal("0.90"), Decimal("100")),
                (Decimal("0.10"), Decimal("101")),
                (Decimal("0.10"), Decimal("102")),
                (Decimal("0.10"), Decimal("103")),
            )
        )
    )
    result = simulate_fixed_horizon_long_only(
        predictions=predictions,
        instrument=Instrument(
            Asset.BTC_USDT,
            Provider.BINANCE,
            MarketType.SPOT,
            Interval.FIVE_MINUTES,
        ),
        threshold=Decimal("0.80"),
        hold=timedelta(minutes=45),
        configuration=SimulationConfiguration(),
    )
    assert len(result.trades) == 1
    assert result.trades[0].entry_at == start
    assert result.trades[0].exit_at == start + timedelta(minutes=60)
    assert all(point.minute in {0, 30} for point, _ in result.equity_curve)
