from dataclasses import dataclass

from .contracts import RefinementClassification

MINIMUM_OBSERVATIONS = 30
MINIMUM_POSITIVE_FOLDS = 4


@dataclass(frozen=True, slots=True)
class RefinementEvidence:
    fold_count: int
    complete_fold_count: int
    folds_beating_benchmark: int
    selected_bucket_fraction: float | None
    selected_bucket_observations: int
    selected_bucket_positive_net_folds: int
    selected_bucket_favorable_direction_folds: int
    selected_bucket_median_net_return: float | None
    selected_bucket_worst_net_return: float | None


def classify_refinement_evidence(
    evidence: RefinementEvidence,
) -> tuple[RefinementClassification, tuple[str, ...]]:
    """Apply version-1 descriptive gates without approving a Strategy."""
    if evidence.fold_count != 7 or evidence.complete_fold_count != evidence.fold_count:
        return RefinementClassification.INCOMPLETE, ("incomplete_fold_count",)
    failed: list[str] = []
    if evidence.folds_beating_benchmark < MINIMUM_POSITIVE_FOLDS:
        failed.append("benchmark_not_beaten_in_majority_of_folds")
    if evidence.selected_bucket_favorable_direction_folds < MINIMUM_POSITIVE_FOLDS:
        failed.append("direction_not_consistent_across_folds")
    if evidence.selected_bucket_positive_net_folds < MINIMUM_POSITIVE_FOLDS:
        failed.append("net_return_not_positive_in_majority_of_folds")
    if evidence.selected_bucket_median_net_return is None or (
        evidence.selected_bucket_median_net_return <= 0
    ):
        failed.append("no_positive_cost_adjusted_confidence_bucket")
    if evidence.selected_bucket_worst_net_return is None:
        failed.append("confidence_bucket_depends_on_single_fold")
    if evidence.selected_bucket_observations < MINIMUM_OBSERVATIONS:
        failed.append("insufficient_high_confidence_observations")
    if not failed:
        return RefinementClassification.PROMISING, ()
    if len(failed) <= 2:
        return RefinementClassification.INSUFFICIENT, tuple(failed)
    return RefinementClassification.DISCARDED, tuple(failed)
