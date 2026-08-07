from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class EconomicV2Evidence:
    family_fingerprint: str
    policy_fingerprint: str
    source_run_id: str
    engine: str
    objective: str
    observations: int
    round_trips: int
    positive_net_folds: int
    folds_beating_benchmark: int
    median_net_return: Decimal
    worst_net_return: Decimal
    maximum_drawdown: Decimal
    inactive_folds: int
    compounded_return_1_5x: Decimal
    compounded_return_2x: Decimal
    complete_and_finite: bool = True


def passes_economic_v2_stage(stage: int, item: EconomicV2Evidence) -> bool:
    if not item.complete_and_finite:
        return False
    if stage == 1:
        return (
            item.observations >= 30
            and (
                item.positive_net_folds >= 2
                or item.median_net_return > Decimal("-0.0015")
            )
            and item.worst_net_return >= Decimal("-0.10")
        )
    if stage == 2:
        return (
            item.observations >= 50
            and item.positive_net_folds >= 3
            and item.folds_beating_benchmark >= 2
            and item.median_net_return > Decimal("-0.0005")
            and item.worst_net_return >= Decimal("-0.075")
            and item.inactive_folds <= 3
        )
    if stage == 3:
        return (
            item.observations >= 100
            and item.positive_net_folds >= 4
            and item.folds_beating_benchmark >= 3
            and item.median_net_return > 0
            and item.worst_net_return >= Decimal("-0.05")
            and item.round_trips >= 10
            and item.inactive_folds <= 2
        )
    raise ValueError("stage must be 1, 2 or 3")


def is_economic_v2_development_candidate(item: EconomicV2Evidence) -> bool:
    return (
        item.complete_and_finite
        and item.observations >= 150
        and item.positive_net_folds >= 5
        and item.folds_beating_benchmark >= 4
        and item.median_net_return >= Decimal("0.001")
        and item.worst_net_return >= Decimal("-0.02")
        and item.round_trips >= 20
        and item.inactive_folds <= 1
        and item.maximum_drawdown <= Decimal("0.25")
        and item.compounded_return_1_5x > 0
        and item.compounded_return_2x > 0
    )


def economic_v2_rank(item: EconomicV2Evidence) -> tuple[object, ...]:
    return (
        -item.positive_net_folds,
        item.inactive_folds,
        -item.compounded_return_2x,
        -item.worst_net_return,
        -item.median_net_return,
        -item.round_trips,
        item.maximum_drawdown,
        item.family_fingerprint,
        item.policy_fingerprint,
    )


def capped_economic_v2_promotions(
    stage: int, evidence: tuple[EconomicV2Evidence, ...]
) -> tuple[EconomicV2Evidence, ...]:
    passing = tuple(item for item in evidence if passes_economic_v2_stage(stage, item))
    if stage == 3:
        return tuple(sorted(passing, key=economic_v2_rank))
    groups = sorted({(item.source_run_id, item.engine) for item in evidence})
    capped: list[EconomicV2Evidence] = []
    for group in groups:
        population = tuple(
            item for item in evidence if (item.source_run_id, item.engine) == group
        )
        eligible = sorted(
            (item for item in passing if (item.source_run_id, item.engine) == group),
            key=economic_v2_rank,
        )
        capacity = (len(population) + 3) // 4
        reserved: list[EconomicV2Evidence] = []
        for objective in sorted({item.objective for item in eligible}):
            if len(reserved) == capacity:
                break
            reserved.append(
                min(
                    (item for item in eligible if item.objective == objective),
                    key=economic_v2_rank,
                )
            )
        reserved_ids = {
            (item.family_fingerprint, item.policy_fingerprint) for item in reserved
        }
        reserved.extend(
            item
            for item in eligible
            if (item.family_fingerprint, item.policy_fingerprint) not in reserved_ids
        )
        capped.extend(reserved[:capacity])
    selected: list[EconomicV2Evidence] = []
    by_source = sorted({item.source_run_id for item in capped})
    for source in by_source:
        source_items = [item for item in capped if item.source_run_id == source]
        for objective in sorted({item.objective for item in source_items}):
            selected.append(
                min(
                    (item for item in source_items if item.objective == objective),
                    key=economic_v2_rank,
                )
            )
    selected_ids = {
        (item.family_fingerprint, item.policy_fingerprint) for item in selected
    }
    selected.extend(
        item
        for item in sorted(capped, key=economic_v2_rank)
        if (item.family_fingerprint, item.policy_fingerprint) not in selected_ids
    )
    return tuple(selected)

