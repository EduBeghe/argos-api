from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np

from app.market.catalog import Instrument
from app.ml.contract import FINAL_TEST_END, FINAL_TEST_START
from app.ml.dataset import MLExperimentDataset
from app.ml.gaps import datetime_to_ns, ns_to_datetime
from app.ml.reader import MLCandleColumns
from app.ml.refinement.calculation import _selected_columns, fit_refinement_model
from app.ml.refinement.contracts import PredictionObjective
from app.ml.refinement.models import MLRefinementEvaluationRecord
from app.ml.refinement.planning import PlannedEvaluation
from app.ml.screening.config import ScreeningEngine
from app.paper.config import SimulationConfiguration

from .contracts import (
    DECISION_CADENCE_SECONDS,
    PROTECTED_END,
    PROTECTED_START,
    SUPPORTED_CONFIDENCE_BUCKET_FRACTIONS,
)
from .policy import FinalValidationPolicyEvidence, failed_final_validation_rules
from .simulation import (
    ProtectedBenchmarkResult,
    ProtectedPrediction,
    ProtectedSimulationResult,
    simulate_buy_and_hold,
    simulate_fixed_horizon_long_only,
)


@dataclass(frozen=True, slots=True)
class ProtectedValidationCalculation:
    base: ProtectedSimulationResult
    stressed_1_5x: ProtectedSimulationResult
    stressed_2x: ProtectedSimulationResult
    buy_and_hold: ProtectedBenchmarkResult
    half_year_returns: tuple[Decimal, ...]
    failed_rules: tuple[str, ...]
    selected_feature_names: tuple[str, ...]
    model: object
    candidate: PlannedEvaluation
    selected_column_indices: tuple[int, ...]
    confidence_bucket_fraction: Decimal
    confidence_score_cutoff: Decimal
    decision_cadence_seconds: int
    parity_development_row_indices: tuple[int, ...]
    development_row_count: int
    protected_row_count: int
    first_protected_boundary: datetime
    last_protected_boundary: datetime


_NANOSECONDS_PER_SECOND = 1_000_000_000
_DECISION_CADENCE_NS = DECISION_CADENCE_SECONDS * _NANOSECONDS_PER_SECOND


def is_decision_boundary_ns(boundary_ns: int) -> bool:
    """Return whether a UTC epoch nanosecond boundary is cadence-aligned."""

    return boundary_ns % _DECISION_CADENCE_NS == 0


def is_protected_decision_boundary_ns(boundary_ns: int) -> bool:
    """Apply the protected half-open range and the frozen decision cadence."""

    return datetime_to_ns(PROTECTED_START) <= boundary_ns < datetime_to_ns(
        PROTECTED_END
    ) and is_decision_boundary_ns(boundary_ns)


def confidence_score_cutoff(
    development_signed_scores: np.ndarray[tuple[int], np.dtype[np.float64]],
    confidence_bucket_fraction: Decimal,
) -> Decimal:
    """Derive the frozen absolute-score cutoff from development evidence only."""

    if (
        not confidence_bucket_fraction.is_finite()
        or confidence_bucket_fraction not in SUPPORTED_CONFIDENCE_BUCKET_FRACTIONS
    ):
        raise ValueError("unsupported confidence bucket fraction")
    if (
        development_signed_scores.ndim != 1
        or development_signed_scores.size == 0
        or not np.all(np.isfinite(development_signed_scores))
    ):
        raise ValueError("development signed scores must be finite and non-empty")
    quantile = np.quantile(
        np.abs(development_signed_scores),
        1.0 - float(confidence_bucket_fraction),
    )
    cutoff = Decimal(str(quantile))
    if not cutoff.is_finite() or not Decimal(0) <= cutoff <= Decimal(1):
        raise ValueError("confidence score cutoff must be finite and between 0 and 1")
    return cutoff


def _signed_positive_scores(
    *, model: object, values: np.ndarray[tuple[int, int], np.dtype[np.float32]]
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("protected classifier does not expose class probabilities")
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    positive = np.flatnonzero(classes == 1)
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(values), len(classes))
        or len(positive) != 1
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError("protected classifier class probabilities are incompatible")
    return 2 * probabilities[:, int(positive[0])] - 1


def _candidate(evaluation: MLRefinementEvaluationRecord) -> PlannedEvaluation:
    policy = evaluation.feature_policy
    variables = policy.get("variables")
    parameters: list[tuple[str, bool | int | float]] = []
    for key, value in evaluation.hyperparameters.items():
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"invalid frozen hyperparameter: {key}")
        parameters.append((str(key), value))
    top_k = policy.get("top_k")
    return PlannedEvaluation(
        evaluation.fingerprint,
        ScreeningEngine(evaluation.engine),
        evaluation.future_duration_seconds,
        PredictionObjective(evaluation.objective),
        tuple(sorted(parameters)),
        str(policy["strategy"]),
        tuple(str(item) for item in variables) if isinstance(variables, list) else None,
        top_k if isinstance(top_k, int) and not isinstance(top_k, bool) else None,
    )


def _half_year_returns(
    result: ProtectedSimulationResult,
) -> tuple[tuple[Decimal, ...], bool]:
    periods = ((2024, 1), (2024, 7), (2025, 1), (2025, 7))
    values: list[Decimal] = []
    complete = True
    for year, month in periods:
        start = datetime(year, month, 1, tzinfo=UTC)
        end = datetime(
            year + (1 if month == 7 else 0), 1 if month == 7 else 7, 1, tzinfo=UTC
        )
        points = tuple(
            equity
            for boundary, equity in result.equity_curve
            if start <= boundary < end
        )
        if len(points) < 2 or not points[0]:
            complete = False
            values.append(Decimal(0))
        else:
            values.append((points[-1] - points[0]) / points[0])
    return tuple(values), complete


def calculate_protected_validation(
    *,
    dataset: MLExperimentDataset,
    candles: MLCandleColumns,
    instrument: Instrument,
    evaluation: MLRefinementEvaluationRecord,
    confidence_fraction: Decimal,
    seed: int,
) -> ProtectedValidationCalculation:
    candidate = _candidate(evaluation)
    if candidate.objective is not PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION:
        raise ValueError(
            "protected validator supports future-direction classification only"
        )
    target = dataset.targets.for_duration(
        timedelta(seconds=candidate.future_duration_seconds)
    )
    development = np.flatnonzero(
        dataset.feature_eligible
        & target.eligible
        & (dataset.boundary_time_ns < datetime_to_ns(FINAL_TEST_START))
    )
    protected_candidates = np.flatnonzero(
        dataset.feature_eligible
        & target.eligible
        & (dataset.boundary_time_ns >= datetime_to_ns(FINAL_TEST_START))
        & (dataset.boundary_time_ns < datetime_to_ns(FINAL_TEST_END))
    )
    protected = np.asarray(
        [
            row
            for row in protected_candidates
            if is_protected_decision_boundary_ns(
                int(dataset.boundary_time_ns[int(row)])
            )
        ],
        dtype=np.int64,
    )
    if not len(development) or not len(protected):
        raise ValueError("protected validation has incomplete eligible evidence")
    columns = _selected_columns(
        candidate, dataset.features[development], target.values[development]
    )
    model = fit_refinement_model(
        candidate=candidate,
        training_x=dataset.features[np.ix_(development, columns)],
        labels=(target.values[development] > 0).astype(np.int64),
        seed=seed,
    )
    development_scores = _signed_positive_scores(
        model=model,
        values=dataset.features[np.ix_(development, columns)],
    )
    cutoff = confidence_score_cutoff(development_scores, confidence_fraction)
    scores = _signed_positive_scores(
        model=model,
        values=dataset.features[np.ix_(protected, columns)],
    )
    predictions: list[ProtectedPrediction] = []
    for row, score in zip(protected, scores, strict=True):
        boundary_ns = int(dataset.boundary_time_ns[row])
        reference_ns = boundary_ns - 300 * 1_000_000_000
        candle_index = int(np.searchsorted(candles.open_time_ns, reference_ns))
        if (
            candle_index >= len(candles)
            or int(candles.open_time_ns[candle_index]) != reference_ns
        ):
            continue
        predictions.append(
            ProtectedPrediction(
                ns_to_datetime(boundary_ns),
                Decimal(str(score)),
                Decimal(str(candles.close_price[candle_index])),
            )
        )
    if not predictions:
        raise ValueError("protected validation has no cadence-aligned market evidence")
    hold = timedelta(seconds=candidate.future_duration_seconds)
    results = tuple(
        simulate_fixed_horizon_long_only(
            predictions=tuple(predictions),
            instrument=instrument,
            threshold=cutoff,
            hold=hold,
            configuration=SimulationConfiguration(
                fee_rate=Decimal("0.001") * multiplier,
                slippage_rate=Decimal("0.0005") * multiplier,
            ),
        )
        for multiplier in (Decimal("1"), Decimal("1.5"), Decimal("2"))
    )
    half_years, half_years_complete = _half_year_returns(results[0])
    benchmark = simulate_buy_and_hold(
        predictions=tuple(predictions),
        instrument=instrument,
        configuration=SimulationConfiguration(
            fee_rate=Decimal("0.001"), slippage_rate=Decimal("0.0005")
        ),
    )
    failed = failed_final_validation_rules(
        FinalValidationPolicyEvidence(
            half_years_complete
            and all(
                value.is_finite()
                for value in (
                    results[0].final_equity,
                    results[0].total_return,
                    results[0].maximum_drawdown,
                    results[1].total_return,
                    results[2].total_return,
                    *half_years,
                )
            ),
            results[0].round_trips,
            results[0].total_return,
            half_years,
            results[0].maximum_drawdown,
            results[1].total_return,
            results[2].total_return,
        )
    )
    from app.ml.registry import VARIABLE_REGISTRY

    return ProtectedValidationCalculation(
        results[0],
        results[1],
        results[2],
        benchmark,
        half_years,
        failed,
        tuple(VARIABLE_REGISTRY[index].name for index in columns),
        model,
        candidate,
        tuple(int(index) for index in columns),
        confidence_fraction,
        cutoff,
        DECISION_CADENCE_SECONDS,
        tuple(int(index) for index in development[:32]),
        len(development),
        len(predictions),
        predictions[0].boundary,
        predictions[-1].boundary,
    )
