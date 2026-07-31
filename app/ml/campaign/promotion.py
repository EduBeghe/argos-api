from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class EconomicEvidence:
    family_fingerprint: str
    source_run_id: str
    engine: str
    observations: int
    favorable_direction_folds: int
    positive_net_folds: int
    folds_beating_benchmark: int
    median_net_return: Decimal | None
    worst_net_return: Decimal | None
    promising_configurations: int = 0
    complete_and_finite: bool = True


def economic_rank(item: EconomicEvidence) -> tuple[object, ...]:
    return (
        -item.positive_net_folds,
        -(item.median_net_return or Decimal("-Infinity")),
        -(item.worst_net_return or Decimal("-Infinity")),
        -item.folds_beating_benchmark,
        -item.observations,
        item.family_fingerprint,
    )


def passes_stage_gate(stage: int, item: EconomicEvidence) -> bool:
    if not item.complete_and_finite or item.median_net_return is None:
        return False
    if stage == 1:
        return (
            item.observations >= 30
            and item.favorable_direction_folds >= 4
            and (
                item.positive_net_folds >= 2
                or item.median_net_return > Decimal("-0.0015")
            )
        )
    if stage == 2:
        return (
            item.observations >= 50
            and item.favorable_direction_folds >= 4
            and item.positive_net_folds >= 3
            and item.folds_beating_benchmark >= 3
            and item.median_net_return > Decimal("-0.0005")
        )
    if stage == 3:
        return (
            item.promising_configurations >= 1
            and item.observations >= 100
            and item.positive_net_folds >= 4
            and item.favorable_direction_folds >= 4
            and item.folds_beating_benchmark >= 4
            and item.median_net_return > 0
        )
    raise ValueError("stage must be 1, 2 or 3")


def capped_promotions(
    stage: int, evidence: tuple[EconomicEvidence, ...]
) -> tuple[EconomicEvidence, ...]:
    passing = [item for item in evidence if passes_stage_gate(stage, item)]
    if stage == 3:
        return tuple(sorted(passing, key=economic_rank))
    selected: list[EconomicEvidence] = []
    groups = sorted({(item.source_run_id, item.engine) for item in evidence})
    for group in groups:
        population = [
            item for item in evidence if (item.source_run_id, item.engine) == group
        ]
        eligible = sorted(
            (item for item in passing if (item.source_run_id, item.engine) == group),
            key=economic_rank,
        )
        cap = (len(population) + 3) // 4
        selected.extend(eligible[:cap])
    return tuple(sorted(selected, key=economic_rank))


def is_strong_candidate(item: EconomicEvidence) -> bool:
    return (
        item.complete_and_finite
        and item.promising_configurations >= 3
        and item.observations >= 100
        and item.positive_net_folds >= 5
        and item.favorable_direction_folds >= 5
        and item.folds_beating_benchmark >= 5
        and item.median_net_return is not None
        and item.median_net_return >= Decimal("0.001")
        and item.worst_net_return is not None
        and item.worst_net_return >= Decimal("-0.0015")
    )
