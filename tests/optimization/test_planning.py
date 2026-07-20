from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.optimization.experience import EvidenceClassification
from app.optimization.planning import (
    CandidateConfiguration,
    HistoricalCandidate,
    PlannerAllocation,
    _neighbors,
    build_search_plan,
)
from app.optimization.refinement import candidate_for
from app.scores.calculator import BASELINE_TECHNICAL_V1_PARAMETERS
from app.strategies.baseline_long_only import BASELINE_LONG_ONLY_V1_PARAMETERS
from tests.optimization.test_experience import profile


def configuration(index: int = 0) -> CandidateConfiguration:
    strategy = replace(
        BASELINE_LONG_ONLY_V1_PARAMETERS,
        entry_threshold=Decimal("0.60") + Decimal(index) / Decimal(1000),
    )
    candidate = candidate_for(BASELINE_TECHNICAL_V1_PARAMETERS, strategy)
    return CandidateConfiguration(
        candidate.stable_candidate_id, candidate.score_parameters, strategy
    )


def test_planner_allocation_requires_exact_decimal_sum_and_novel_budget() -> None:
    assert (
        sum(
            (
                PlannerAllocation().known_rechecks,
                PlannerAllocation().promising_neighborhoods,
                PlannerAllocation().uncertain_rechecks,
                PlannerAllocation().novel_exploration,
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match="sum to one"):
        PlannerAllocation(novel_exploration=Decimal("0.19"))
    with pytest.raises(ValueError, match="novel"):
        PlannerAllocation(
            known_rechecks=Decimal("0.20"),
            promising_neighborhoods=Decimal("0.60"),
            uncertain_rechecks=Decimal("0.20"),
            novel_exploration=Decimal(0),
        )


def test_no_history_preserves_legacy_fallback() -> None:
    assert (
        build_search_plan(
            historical=(),
            novel_pool=tuple(configuration(index) for index in range(20)),
            max_candidates=10,
            production_baseline=configuration(),
        )
        is None
    )


def test_same_history_produces_bounded_unique_plan_with_all_categories() -> None:
    baseline = configuration()
    promising = HistoricalCandidate(
        baseline,
        profile(evidence_classification=EvidenceClassification.PROMISING),
        False,
    )
    uncertain_configuration = configuration(1)
    uncertain = HistoricalCandidate(
        uncertain_configuration,
        profile(
            stable_candidate_id=uncertain_configuration.stable_candidate_id,
            evidence_classification=EvidenceClassification.INSUFFICIENT_EVIDENCE,
            last_evaluated_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
        False,
    )
    novel = tuple(configuration(index) for index in range(2, 80))
    first = build_search_plan(
        historical=(promising, uncertain),
        novel_pool=(baseline, uncertain_configuration, *novel),
        max_candidates=40,
        production_baseline=baseline,
    )
    second = build_search_plan(
        historical=(promising, uncertain),
        novel_pool=(baseline, uncertain_configuration, *novel),
        max_candidates=40,
        production_baseline=baseline,
    )
    assert first == second
    assert first is not None
    assert baseline in first.candidates
    assert len(first.novel_exploration) > 0
    assert len(first.candidates) <= 40
    assert len({item.stable_candidate_id for item in first.candidates}) == len(
        first.candidates
    )
    assert first.promising_neighborhoods


def test_neighbors_are_valid_and_directional_sum_is_exact() -> None:
    values = _neighbors(configuration())
    assert values
    for value in values:
        score = value.score_parameters
        assert (
            score.trend_weight
            + score.rsi_weight
            + score.return_1_weight
            + score.return_12_weight
            == 1
        )
        assert Decimal(0) <= score.volume_confirmation_weight <= Decimal("0.50")
        assert (
            value.strategy_parameters.exit_threshold
            < value.strategy_parameters.entry_threshold
        )
