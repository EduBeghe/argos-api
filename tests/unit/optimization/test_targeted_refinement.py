from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from app.optimization.predictive_edge_engine_v41 import (
    ContextRegime,
    FrozenRegime,
    PhaseEvidence,
    PredictiveEdgeV41Result,
    RegimeConditionedRelationship,
    V41Candidate,
    V41Conclusion,
    V41StageCounters,
)
from app.optimization.predictive_edges import DirectionMapping, SuggestedAction
from app.optimization.targeted_refinement import (
    MAX_TARGETED_CONFIGURATIONS,
    generate_targeted_configurations,
    select_targeted_seeds,
)


def _candidate(
    *, observations: int = 96, windows: int = 6, fingerprint: str = "a" * 64
) -> V41Candidate:
    evidence = PhaseEvidence(
        observations,
        Decimal("0.005"),
        Decimal("0.005"),
        Decimal("0.00385"),
        Decimal("0.60"),
        Decimal("0.50"),
        Decimal("0.10"),
        Decimal("0.01"),
        False,
        True,
        False,
        windows,
        2,
        4,
        0,
    )
    regime = FrozenRegime(
        "trend_regime", 2, ContextRegime.BULLISH, Decimal("0"), None, 200, "r" * 64
    )
    relationship = RegimeConditionedRelationship(
        "ema_50_slope_48",
        2,
        "trend_regime",
        2,
        ContextRegime.BULLISH,
        timedelta(hours=6),
        timedelta(hours=24),
        regime,
        2,
        Decimal("-0.01"),
        Decimal("0.01"),
        DirectionMapping.POSITIVE,
        SuggestedAction.LONG_ENTRY,
        "balanced_v41",
        evidence,
        evidence,
        Decimal("0.001"),
        fingerprint,
    )
    return V41Candidate(relationship, evidence, None, False, False, "unstable")


def _result(*candidates: V41Candidate) -> PredictiveEdgeV41Result:
    zero = V41StageCounters(0, 0, 0, 0)
    return PredictiveEdgeV41Result(
        (),
        (),
        tuple(item.relationship for item in candidates),
        tuple(candidates),
        tuple(candidates),
        (),
        (),
        (),
        V41Conclusion.ECONOMIC_INTERACTION_BUT_UNSTABLE,
        zero,
        zero,
        zero,
        zero,
        zero,
        {},
    )


def test_selects_economic_multi_window_seed_and_excludes_tiny_sample() -> None:
    good = _candidate()
    tiny = _candidate(observations=1, windows=1, fingerprint="b" * 64)

    selected = select_targeted_seeds(_result(tiny, good))

    assert [item.candidate.relationship.directional_predictor for item in selected] == [
        "ema_50_slope_48"
    ]
    assert selected[0].candidate.validation.observations == 96
    assert selected[0].candidate.validation.usable_window_count == 6


def test_neighborhood_is_deterministic_deduplicated_and_bounded() -> None:
    seeds = select_targeted_seeds(
        _result(*(_candidate(fingerprint=f"{value:064x}") for value in range(8)))
    )

    first, first_duplicates = generate_targeted_configurations(seeds)
    second, second_duplicates = generate_targeted_configurations(tuple(reversed(seeds)))

    assert len(first) == MAX_TARGETED_CONFIGURATIONS
    assert len({item.fingerprint for item in first}) == len(first)
    assert [item.fingerprint for item in first] == [item.fingerprint for item in second]
    assert first_duplicates == second_duplicates
    assert {item.cadence for item in first}.issuperset(
        {timedelta(hours=4), timedelta(hours=6)}
    )
    assert {item.horizon for item in first}.issuperset(
        {timedelta(hours=12), timedelta(hours=24), timedelta(hours=48)}
    )


def test_seed_ranking_is_distance_then_economics_then_observations() -> None:
    weaker = _candidate(fingerprint="f" * 64)
    stronger_evidence = replace(
        _candidate(fingerprint="0" * 64),
        validation=replace(weaker.validation, net_movement=Decimal("0.01")),
    )

    selected = select_targeted_seeds(_result(weaker, stronger_evidence))

    assert selected[0].candidate.relationship.fingerprint == "0" * 64
