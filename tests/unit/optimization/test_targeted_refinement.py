from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from typing import cast
from unittest.mock import Mock

import pytest

import app.optimization.targeted_refinement as targeted_module
from app.optimization.edge_guided import (
    EdgeGuidedCandidateEvaluation,
    EdgeGuidedCandidateStatus,
    EdgeGuidedEvaluationResult,
    EdgeGuidedSegmentEvaluation,
)
from app.optimization.feature_matrix import HistoricalFeatureMatrix
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
from app.optimization.predictive_edges import (
    DirectionMapping,
    EvidenceThresholds,
    SuggestedAction,
)
from app.optimization.refinement import RefinementSegmentName
from app.optimization.targeted_refinement import (
    MAX_TARGETED_CONFIGURATIONS,
    TargetedCandidateStatus,
    TargetedConfiguration,
    _normalized_distance,
    _stage_requirements,
    _status,
    evaluate_targeted_refinement,
    generate_targeted_configurations,
    select_targeted_seeds,
)
from app.paper.config import SimulationConfiguration


def _candidate(
    *,
    observations: int = 96,
    windows: int = 6,
    fingerprint: str = "a" * 64,
    selected_bin: int = 2,
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
        selected_bin,
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


def test_adjacent_bins_stay_inside_frozen_three_bin_contract() -> None:
    for selected, expected in ((1, {1, 2}), (2, {1, 2, 3}), (3, {2, 3})):
        seeds = select_targeted_seeds(
            _result(_candidate(selected_bin=selected, fingerprint=f"{selected:064x}"))
        )

        configurations, _ = generate_targeted_configurations(seeds)

        assert {item.bin_ordinal for item in configurations} == expected
        assert all(1 <= item.bin_ordinal <= 3 for item in configurations)


def _metrics(
    segment: RefinementSegmentName,
    *,
    total_return: str = "0.01",
    trades: int = 2,
    drawdown: str = "0.01",
    excess: str = "0.005",
) -> EdgeGuidedSegmentEvaluation:
    return EdgeGuidedSegmentEvaluation(
        segment,
        Decimal(total_return),
        Decimal(drawdown),
        trades,
        Decimal("1"),
        Decimal(excess),
        Decimal("0"),
    )


def test_requirements_only_approve_executed_stages_and_use_real_deficits() -> None:
    training = _metrics(RefinementSegmentName.TRAINING)
    validation = _metrics(
        RefinementSegmentName.VALIDATION, total_return="-0.01", trades=0
    )
    margins = _stage_requirements(
        training=training,
        validation=validation,
        test=None,
        validation_window_count=1,
        confirmed_window_count=0,
        confirmation_ratio=Decimal(0),
        thresholds=EvidenceThresholds(),
    )

    passed = {item.requirement for item in margins if item.passed}
    failed = {item.requirement for item in margins if not item.passed}
    assert all(not item.startswith("isolated_test") for item in passed | failed)
    assert "validation.sufficient_trades" in failed
    assert "validation.positive_net_return" in failed
    assert "validation.minimum_windows" in failed
    assert _normalized_distance(margins) > 0


def test_four_targeted_statuses_are_derived_from_actual_stage_boundary() -> None:
    assert _status(None, False) is TargetedCandidateStatus.REJECTED_TRAINING
    assert _status(None, True) is TargetedCandidateStatus.REJECTED_VALIDATION

    class Evaluation:
        def __init__(self, status: EdgeGuidedCandidateStatus) -> None:
            self.status = status

    rejected = cast(
        EdgeGuidedCandidateEvaluation,
        Evaluation(EdgeGuidedCandidateStatus.REJECTED_TEST),
    )
    confirmed = cast(
        EdgeGuidedCandidateEvaluation,
        Evaluation(EdgeGuidedCandidateStatus.CONFIRMED),
    )
    assert _status(rejected, True) is (TargetedCandidateStatus.REJECTED_TEST)
    assert _status(confirmed, True) is (TargetedCandidateStatus.CONFIRMED)


def test_each_cadence_is_evaluated_with_its_own_sampling_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = select_targeted_seeds(_result(_candidate()))[0]
    configurations = (
        TargetedConfiguration(
            "4" * 64,
            seed.candidate.relationship.fingerprint,
            timedelta(hours=4),
            timedelta(hours=24),
            2,
            Decimal("-0.01"),
            Decimal("0.01"),
        ),
        TargetedConfiguration(
            "6" * 64,
            seed.candidate.relationship.fingerprint,
            timedelta(hours=6),
            timedelta(hours=24),
            2,
            Decimal("-0.01"),
            Decimal("0.01"),
        ),
        TargetedConfiguration(
            "8" * 64,
            seed.candidate.relationship.fingerprint,
            timedelta(hours=8),
            timedelta(hours=24),
            2,
            Decimal("-0.01"),
            Decimal("0.01"),
        ),
        TargetedConfiguration(
            "c" * 64,
            seed.candidate.relationship.fingerprint,
            timedelta(hours=12),
            timedelta(hours=24),
            2,
            Decimal("-0.01"),
            Decimal("0.01"),
        ),
    )
    monkeypatch.setattr(
        targeted_module,
        "generate_targeted_configurations",
        Mock(return_value=(configurations, 0)),
    )
    observed: list[tuple[timedelta, tuple[timedelta, ...]]] = []

    def evaluate(**kwargs: object) -> EdgeGuidedEvaluationResult:
        interval = kwargs["decision_interval"]
        candidates = kwargs["frozen_candidates"]
        assert isinstance(interval, timedelta)
        assert isinstance(candidates, tuple)
        observed.append(
            (interval, tuple(item.decision_interval for item in candidates))
        )
        return EdgeGuidedEvaluationResult(
            "dataset",
            len(candidates),
            0,
            len(candidates),
            0,
            (),
            funnel={
                "training_evaluated_count": len(candidates),
                "training_retained_count": 0,
                "validation_evaluated_count": 0,
                "validation_retained_count": 0,
                "test_evaluated_count": 0,
            },
        )

    monkeypatch.setattr(targeted_module, "evaluate_edge_guided_candidates", evaluate)
    evaluate_targeted_refinement(
        regime_result=_result(seed.candidate),
        dataset=Mock(),
        feature_matrix=HistoricalFeatureMatrix("dataset", ()),
        windows=(),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        thresholds=EvidenceThresholds(),
    )

    assert observed == [
        (timedelta(hours=4), (timedelta(hours=4),)),
        (timedelta(hours=6), (timedelta(hours=6),)),
        (timedelta(hours=8), (timedelta(hours=8),)),
        (timedelta(hours=12), (timedelta(hours=12),)),
    ]
