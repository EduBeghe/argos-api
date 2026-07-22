from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from app.optimization.predictive_edge_v4 import (
    CHEAP_RELATIONSHIP_LIMIT,
    DiscoveryConfiguration,
    EvidenceProfile,
    NeutralBandPolicy,
    RejectionReason,
    diversified_relationships,
    neutral_threshold_for_training,
    rank_near_misses,
    requirement_margin,
    threshold_sensitivity,
)
from app.optimization.predictive_edges import (
    PREDICTORS,
    Condition,
    EvidenceThresholds,
)


def test_neutral_policies_are_frozen_from_training() -> None:
    values = (
        Decimal("0.04"),
        Decimal("0.01"),
        Decimal("0.03"),
        Decimal("0.02"),
    )
    assert neutral_threshold_for_training(
        policy=NeutralBandPolicy.ZERO_DIRECTION,
        training_absolute_returns=values,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    ) == Decimal(0)
    assert neutral_threshold_for_training(
        policy=NeutralBandPolicy.TRAINING_NOISE,
        training_absolute_returns=values,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    ) == Decimal("0.01")
    assert neutral_threshold_for_training(
        policy=NeutralBandPolicy.ECONOMIC_FRICTION,
        training_absolute_returns=values,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    ) == Decimal("0.0030")


def test_configuration_fingerprint_includes_all_discovery_dimensions() -> None:
    base = DiscoveryConfiguration(
        timedelta(minutes=30),
        timedelta(hours=1),
        5,
        Condition.ALL,
        NeutralBandPolicy.TRAINING_NOISE,
        EvidenceProfile.BALANCED_V4,
        Decimal("0.001"),
        EvidenceThresholds(),
    )
    alternatives = (
        replace(base, cadence=timedelta(hours=1)),
        replace(base, horizon=timedelta(hours=2)),
        replace(base, bins=7),
        replace(base, profile=EvidenceProfile.EXPLORATORY_V4),
    )
    assert len({base.fingerprint, *(item.fingerprint for item in alternatives)}) == 5


def test_requirement_margins_preserve_signed_and_relative_distance() -> None:
    failed = requirement_margin(
        RejectionReason.HIT_RATE_BELOW_THRESHOLD,
        Decimal("0.54"),
        Decimal("0.55"),
    )
    assert failed.absolute_margin == Decimal("-0.01")
    assert failed.relative_margin == Decimal("-0.01") / Decimal("0.55")
    assert failed.passed is False


def test_redundancy_groups_cannot_monopolize_top_k() -> None:
    ranked = (
        *(
            (PREDICTORS[index % 2], "derived_baseline", Decimal(100 - index))
            for index in range(CHEAP_RELATIONSHIP_LIMIT + 10)
        ),
        (PREDICTORS[4], "raw_return", Decimal("0.01")),
    )
    selected = diversified_relationships(ranked)
    assert len(selected) == CHEAP_RELATIONSHIP_LIMIT
    assert {item[1] for item in selected} == {"derived_baseline", "raw_return"}


def test_strict_balanced_and_exploratory_profiles_are_distinct() -> None:
    assert {item.value for item in EvidenceProfile} == {
        "strict_v3_compatible",
        "balanced_v4",
        "exploratory_v4",
    }


def test_sensitivity_changes_one_threshold_only() -> None:
    # The public helper is typed around ranked diagnostics; an empty ranking is a
    # deterministic no-op and cannot consult isolated test evidence.
    assert threshold_sensitivity(rank_near_misses(())) == ()
