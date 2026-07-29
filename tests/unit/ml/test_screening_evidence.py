from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.ml.screening.evidence import EvidenceClassification, classify_evidence
from app.ml.screening.models import MLModelScreeningResultRecord


def result(**overrides: object) -> MLModelScreeningResultRecord:
    values: dict[str, object] = {
        "id": uuid4(),
        "run_id": uuid4(),
        "engine": "ridge",
        "future_duration_seconds": 1800,
        "folds_evaluated": 7,
        "total_validation_samples": 700,
        "median_mae_improvement_over_zero": Decimal("0.01"),
        "worst_fold_mae_improvement_over_zero": Decimal("0.001"),
        "median_directional_accuracy": Decimal("0.51"),
        "worst_fold_directional_accuracy": Decimal("0.5"),
        "median_pearson_correlation": Decimal("0.01"),
        "median_spearman_correlation": Decimal("0.01"),
        "folds_beating_zero": 7,
        "folds_with_positive_spearman": 7,
        "total_training_time_seconds": Decimal("1.0"),
        "pareto_retained": False,
        "display_rank": 1,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    values.update(overrides)
    return MLModelScreeningResultRecord(**values)


def test_all_promising_boundaries_pass() -> None:
    assessment = classify_evidence(result())
    assert assessment.classification is EvidenceClassification.PROMISING
    assert assessment.failed_rules == ()


@pytest.mark.parametrize(
    ("overrides", "rule"),
    [
        ({"folds_beating_zero": 6}, "not_all_folds_beat_zero"),
        ({"folds_with_positive_spearman": 6}, "not_all_folds_positive_spearman"),
        (
            {"median_mae_improvement_over_zero": Decimal("0")},
            "non_positive_median_mae_improvement",
        ),
        (
            {"worst_fold_mae_improvement_over_zero": Decimal("0")},
            "non_positive_worst_mae_improvement",
        ),
        (
            {"median_directional_accuracy": Decimal("0.5")},
            "median_directional_accuracy_not_above_random",
        ),
        (
            {"worst_fold_directional_accuracy": Decimal("0.499")},
            "worst_directional_accuracy_below_random",
        ),
        ({"median_spearman_correlation": Decimal("0")}, "non_positive_median_spearman"),
        ({"median_pearson_correlation": Decimal("0")}, "non_positive_median_pearson"),
    ],
)
def test_each_promising_gate_has_a_stable_failure(
    overrides: dict[str, object], rule: str
) -> None:
    assert rule in classify_evidence(result(**overrides)).failed_rules


@pytest.mark.parametrize(
    "field", ["median_pearson_correlation", "median_spearman_correlation"]
)
def test_null_correlation_is_incomplete(field: str) -> None:
    assert (
        classify_evidence(result(**{field: None})).classification
        is EvidenceClassification.INCOMPLETE
    )


def test_six_folds_is_incomplete() -> None:
    assert (
        classify_evidence(result(folds_evaluated=6)).classification
        is EvidenceClassification.INCOMPLETE
    )


def test_partial_combined_evidence_is_insufficient() -> None:
    assessment = classify_evidence(
        result(folds_beating_zero=3, folds_with_positive_spearman=3)
    )
    assert assessment.classification is EvidenceClassification.INSUFFICIENT


def test_no_combined_positive_evidence_is_discarded() -> None:
    assessment = classify_evidence(
        result(
            median_mae_improvement_over_zero=Decimal("-1"),
            median_spearman_correlation=Decimal("-1"),
            median_pearson_correlation=Decimal("-1"),
            median_directional_accuracy=Decimal("0.4"),
        )
    )
    assert assessment.classification is EvidenceClassification.DISCARDED


def test_classification_is_deterministic_and_ignores_pareto() -> None:
    first = classify_evidence(result(pareto_retained=False, folds_beating_zero=3))
    second = classify_evidence(result(pareto_retained=True, folds_beating_zero=3))
    assert first == second
