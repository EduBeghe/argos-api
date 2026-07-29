from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.ml.screening.models import MLModelScreeningResultRecord

EVIDENCE_CLASSIFICATION_VERSION = 1
EXPECTED_FOLD_COUNT = 7
RANDOM_DIRECTIONAL_ACCURACY = Decimal("0.5")


class EvidenceClassification(StrEnum):
    PROMISING = "promising"
    INSUFFICIENT = "insufficient"
    DISCARDED = "discarded"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    classification: EvidenceClassification
    explanation: str
    failed_rules: tuple[str, ...]


def classify_evidence(result: MLModelScreeningResultRecord) -> EvidenceAssessment:
    """Classify development-fold evidence using immutable version-1 rules."""
    failed: list[str] = []
    if result.folds_evaluated != EXPECTED_FOLD_COUNT:
        failed.append("incomplete_fold_count")
    if result.median_spearman_correlation is None:
        failed.append("missing_median_spearman")
    if result.median_pearson_correlation is None:
        failed.append("missing_median_pearson")
    if failed:
        return EvidenceAssessment(
            EvidenceClassification.INCOMPLETE,
            "The persisted development-fold evidence is incomplete.",
            tuple(failed),
        )

    median_spearman = result.median_spearman_correlation
    median_pearson = result.median_pearson_correlation
    assert median_spearman is not None
    assert median_pearson is not None

    checks = (
        (result.folds_beating_zero == EXPECTED_FOLD_COUNT, "not_all_folds_beat_zero"),
        (
            result.folds_with_positive_spearman == EXPECTED_FOLD_COUNT,
            "not_all_folds_positive_spearman",
        ),
        (
            result.median_mae_improvement_over_zero > 0,
            "non_positive_median_mae_improvement",
        ),
        (
            result.worst_fold_mae_improvement_over_zero > 0,
            "non_positive_worst_mae_improvement",
        ),
        (
            result.median_directional_accuracy > RANDOM_DIRECTIONAL_ACCURACY,
            "median_directional_accuracy_not_above_random",
        ),
        (
            result.worst_fold_directional_accuracy >= RANDOM_DIRECTIONAL_ACCURACY,
            "worst_directional_accuracy_below_random",
        ),
        (
            median_spearman > 0,
            "non_positive_median_spearman",
        ),
        (
            median_pearson > 0,
            "non_positive_median_pearson",
        ),
    )
    failed = [identifier for passed, identifier in checks if not passed]
    if not failed:
        return EvidenceAssessment(
            EvidenceClassification.PROMISING,
            "Evidence is positive and consistent across all seven development folds.",
            (),
        )
    mae = result.median_mae_improvement_over_zero > 0
    spearman = median_spearman > 0
    directional = result.median_directional_accuracy > RANDOM_DIRECTIONAL_ACCURACY
    if (mae and spearman) or (mae and directional) or (spearman and directional):
        return EvidenceAssessment(
            EvidenceClassification.INSUFFICIENT,
            "Partial positive evidence exists, but version-1 consistency gates failed.",
            tuple(failed),
        )
    return EvidenceAssessment(
        EvidenceClassification.DISCARDED,
        "The complete result lacks enough combined positive screening evidence.",
        tuple(failed),
    )
