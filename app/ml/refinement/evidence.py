from dataclasses import dataclass

from .contracts import RefinementClassification

MINIMUM_OBSERVATIONS = 30
MINIMUM_POSITIVE_FOLDS = 4


@dataclass(frozen=True, slots=True)
class RefinementEvidence:
    fold_count: int
    folds_beating_benchmark: int
    folds_with_positive_direction: int
    best_confidence_net_return: float | None
    best_confidence_observations: int


def classify_refinement_evidence(
    evidence: RefinementEvidence,
) -> tuple[RefinementClassification, tuple[str, ...]]:
    """Apply version-1 descriptive gates without approving a Strategy."""
    if evidence.fold_count != 7:
        return RefinementClassification.INCOMPLETE, ("incomplete_fold_count",)
    failed: list[str] = []
    if evidence.folds_beating_benchmark < MINIMUM_POSITIVE_FOLDS:
        failed.append("benchmark_not_beaten_in_majority_of_folds")
    if evidence.folds_with_positive_direction < MINIMUM_POSITIVE_FOLDS:
        failed.append("direction_not_consistent_across_folds")
    if (
        evidence.best_confidence_net_return is None
        or evidence.best_confidence_net_return <= 0
    ):
        failed.append("no_positive_cost_adjusted_confidence_bucket")
    if evidence.best_confidence_observations < MINIMUM_OBSERVATIONS:
        failed.append("insufficient_high_confidence_observations")
    if not failed:
        return RefinementClassification.PROMISING, ()
    if len(failed) <= 2:
        return RefinementClassification.INSUFFICIENT, tuple(failed)
    return RefinementClassification.DISCARDED, tuple(failed)
