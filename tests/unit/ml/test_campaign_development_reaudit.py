import argparse
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.jobs.audit_ml_search_near_misses import maximum_candidates
from app.ml.campaign.development_reaudit import (
    ACCUMULATING_FOLD_NAMES,
    CandidateEvidence,
    DevelopmentReauditError,
    aggregate_blocks,
    build_development_reaudit_report,
    candidate_evidence,
    candidate_rank,
    funnel_diagnosis,
    is_eligible,
    report_json,
    validate_candidate_scientific_identity,
    validate_maximum_candidates,
)
from app.ml.campaign.models import MLSearchCampaignFamilyRecord
from app.ml.protected.simulation import ProtectedSimulationResult
from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)
from app.ml.refinement.persistence import effective_evaluation_fingerprint
from app.ml.refinement.planning import build_refinement_plan

NOW = datetime(2040, 1, 1, tzinfo=UTC)


def family(
    *,
    identifier: int = 1,
    ordinal: int = 1,
    objective: str = "future_direction_classification",
    state: str = "discarded",
    stage: int = 4,
) -> MLSearchCampaignFamilyRecord:
    return MLSearchCampaignFamilyRecord(
        id=UUID(int=identifier),
        campaign_id=UUID(int=100),
        source_screening_run_id=UUID(int=200),
        fingerprint=f"{identifier:064x}",
        dataset_fingerprint="d" * 64,
        asset="BTC/USDT",
        provider="binance",
        market_type="spot",
        interval="5m",
        engine="ridge",
        future_duration_seconds=1800,
        objective=objective,
        feature_variant="all",
        state=state,
        current_stage=stage,
        evaluations_planned=64,
        evaluations_completed=64,
        evaluations_reused=0,
        best_evaluation_id=UUID(int=identifier + 1000),
        best_metrics={
            "complete_and_finite": True,
            "observations": 500,
            "positive_net_folds": 5,
            "favorable_direction_folds": 5,
            "folds_beating_benchmark": 4,
            "median_net_return": "0.0001",
            "worst_net_return": "-0.02",
        },
        decision_reason="failed_stage_4_economic_gate",
        deterministic_ordinal=ordinal,
        created_at=NOW,
    )


def evaluation(identifier: int = 1) -> MLRefinementEvaluationRecord:
    return MLRefinementEvaluationRecord(
        id=UUID(int=identifier + 1000),
        run_id=UUID(int=identifier + 2000),
        reused_from_evaluation_id=None,
        fingerprint=f"{identifier + 1:064x}",
        engine="ridge",
        future_duration_seconds=1800,
        objective="future_direction_classification",
        hyperparameters={"alpha": 1, "fit_intercept": True},
        feature_policy={"strategy": "all", "variables": None, "top_k": None},
        confidence_buckets=["0.05", "0.10", "0.20"],
        fold_evidence=[],
        aggregate_metrics={
            "selected_confidence_bucket": {"fraction": "0.050000000000000000"}
        },
        classification="promising_for_strategy_simulation",
        classification_version=1,
        failed_rules=[],
        elapsed_seconds=Decimal(1),
        created_at=NOW,
    )


def scientific_graph() -> tuple[
    MLSearchCampaignFamilyRecord,
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
]:
    source_id = UUID(int=200)
    result_id = UUID(int=300)
    request = MLRefinementRequest.model_validate(
        {
            "source_screening_run_id": str(source_id),
            "source_screening_result_id": str(result_id),
            "engines": ["ridge"],
            "future_durations_seconds": [1800],
            "search_space": [
                {
                    "engine": "ridge",
                    "alpha": {"values": [1]},
                    "fit_intercept": {"values": [True]},
                }
            ],
            "feature_selection": {"strategies": [{"strategy": "all"}]},
            "prediction_objectives": ["future_direction_classification"],
            "confidence_buckets": ["0.05", "0.10", "0.20"],
            "fee_rate": "0.001",
            "slippage_rate": "0.0005",
            "maximum_trials": 64,
            "maximum_runtime_seconds": 3600,
            "deterministic_seed": 1729,
        }
    )
    planned = build_refinement_plan(request).evaluations[0]
    run_id = UUID(int=2001)
    versions = {"scikit-learn": "1.0"}
    fingerprint = effective_evaluation_fingerprint(
        dataset_fingerprint="d" * 64,
        candidate_fingerprint=planned.fingerprint,
        engine=planned.engine.value,
        versions=versions,
        deterministic_seed=request.deterministic_seed,
    )
    family_record = family()
    evaluation_record = evaluation()
    evaluation_record.run_id = run_id
    evaluation_record.fingerprint = fingerprint
    evaluation_record.classification_version = 2
    evaluation_record.hyperparameters = dict(planned.hyperparameters)
    run_record = MLRefinementRunRecord(
        id=run_id,
        experiment_key="e" * 64,
        source_screening_run_id=source_id,
        source_screening_result_id=result_id,
        status="completed",
        contract_version=2,
        classification_version=2,
        dataset_fingerprint="d" * 64,
        variable_registry_version=1,
        missing_data_policy_version=1,
        library_versions=versions,
        normalized_configuration=request.normalized_payload(),
        notes=None,
        maximum_trials=64,
        maximum_runtime_seconds=3600,
        combinations_planned=1,
        evaluations_computed=1,
        evaluations_reused=0,
        evaluations_omitted=0,
        evaluations_selected=1,
        budget_exhausted=False,
        termination_reason="completed",
        started_at=NOW,
        completed_at=NOW,
        failed_at=None,
        failure_type=None,
        failure_message=None,
        worker_id=None,
        claimed_at=None,
        heartbeat_at=None,
        cancellation_requested_at=None,
        elapsed_seconds=Decimal(1),
        created_at=NOW,
    )
    return family_record, evaluation_record, run_record


@pytest.mark.parametrize(
    ("field", "failing", "passing"),
    [
        ("observations", 499, 500),
        ("positive_net_folds", 4, 5),
        ("favorable_direction_folds", 4, 5),
        ("folds_beating_benchmark", 3, 4),
        ("median_net_return", Decimal(0), Decimal("0.0000001")),
        ("worst_net_return", Decimal("-0.0200001"), Decimal("-0.02")),
    ],
)
def test_exact_eligibility_boundaries(
    field: str, failing: int | Decimal, passing: int | Decimal
) -> None:
    source = candidate_evidence(family(), evaluation())

    def changed(value: int | Decimal) -> CandidateEvidence:
        if field == "observations":
            return replace(source, observations=int(value))
        if field == "positive_net_folds":
            return replace(source, positive_net_folds=int(value))
        if field == "favorable_direction_folds":
            return replace(source, favorable_direction_folds=int(value))
        if field == "folds_beating_benchmark":
            return replace(source, folds_beating_benchmark=int(value))
        if field == "median_net_return":
            return replace(source, median_net_return=Decimal(value))
        return replace(source, worst_net_return=Decimal(value))

    rejected = changed(failing)
    assert not is_eligible(rejected)
    accepted = changed(passing)
    assert is_eligible(accepted)


def test_candidate_ranking_uses_ordinal_then_uuid_for_exact_ties() -> None:
    first = candidate_evidence(family(identifier=2, ordinal=1), evaluation(2))
    second = candidate_evidence(family(identifier=1, ordinal=1), evaluation(1))
    third = candidate_evidence(family(identifier=3, ordinal=2), evaluation(3))
    assert [
        item.family.id for item in sorted((third, first, second), key=candidate_rank)
    ] == [
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
    ]


def test_scientific_identity_accepts_the_canonical_family_evaluation_and_run() -> None:
    family_record, evaluation_record, run_record = scientific_graph()
    validate_candidate_scientific_identity(
        family=family_record, evaluation=evaluation_record, run=run_record
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("engine", "engine"),
        ("objective", "objective"),
        ("horizon", "future_duration_seconds"),
        ("feature_variant", "feature_policy"),
        ("dataset", "dataset_fingerprint"),
        ("ownership", "evaluation.run_id"),
        ("run_status", "run.status"),
    ],
)
def test_scientific_identity_mismatches_fail_with_stable_field(
    field: str, expected: str
) -> None:
    family_record, evaluation_record, run_record = scientific_graph()
    if field == "engine":
        evaluation_record.engine = "lightgbm"
    elif field == "objective":
        evaluation_record.objective = "future_return_regression"
    elif field == "horizon":
        evaluation_record.future_duration_seconds = 3600
    elif field == "feature_variant":
        family_record.feature_variant = "top_k_16"
    elif field == "dataset":
        run_record.dataset_fingerprint = "x" * 64
    elif field == "ownership":
        evaluation_record.run_id = UUID(int=999)
    else:
        run_record.status = "failed"
        run_record.completed_at = None
        run_record.failed_at = NOW
    with pytest.raises(DevelopmentReauditError) as raised:
        validate_candidate_scientific_identity(
            family=family_record, evaluation=evaluation_record, run=run_record
        )
    assert raised.value.code == "candidate_identity_mismatch"
    assert str(family_record.id) in str(raised.value)
    assert expected in str(raised.value)


def test_scientific_identity_rejects_noncanonical_evaluation_policy() -> None:
    family_record, evaluation_record, run_record = scientific_graph()
    evaluation_record.feature_policy = {
        "strategy": "top_k",
        "variables": None,
        "top_k": 16,
    }
    with pytest.raises(DevelopmentReauditError, match="feature_policy"):
        validate_candidate_scientific_identity(
            family=family_record, evaluation=evaluation_record, run=run_record
        )


def test_funnel_reconciles_all_dimensions_and_near_miss_stages() -> None:
    rows = (
        family(identifier=1, objective="future_return_regression", stage=1),
        family(identifier=2, objective="cost_adjusted_action_classification", stage=2),
        family(identifier=3, stage=3),
        family(identifier=4, stage=4),
    )
    result = funnel_diagnosis(rows, expected_total=4)
    assert result["families_total"] == 4
    assert result["near_misses"] == {"stage_3": 1, "stage_4": 1}
    terminal = result["terminal_stage_state"]
    objectives = result["by_objective"]
    assert isinstance(terminal, Mapping)
    assert isinstance(objectives, Mapping)
    assert sum(int(value) for value in terminal.values()) == 4
    assert len(objectives) == 4


def test_funnel_rejects_unknown_state_and_count_mismatch() -> None:
    with pytest.raises(DevelopmentReauditError, match="count"):
        funnel_diagnosis((family(),), expected_total=2)
    invalid = family()
    invalid.state = "unknown"
    with pytest.raises(DevelopmentReauditError, match="state"):
        funnel_diagnosis((invalid,), expected_total=1)


def simulation(
    total_return: str,
    *,
    drawdown: str = "0.1",
    round_trips: int = 5,
) -> ProtectedSimulationResult:
    value = Decimal(total_return)
    return ProtectedSimulationResult(
        Decimal("10000"),
        Decimal("10000") * (1 + value),
        Decimal("10000") * value,
        value,
        Decimal("1"),
        Decimal("2"),
        round_trips,
        3,
        2,
        Decimal("0.01"),
        Decimal("-0.01"),
        Decimal("0.2"),
        Decimal(drawdown),
        (),
        (),
    )


def test_block_aggregation_compounds_decimal_and_uses_maximum_block_drawdown() -> None:
    base = (
        simulation("0.03", drawdown="0.1"),
        simulation("0.04", drawdown="0.2"),
        simulation("0.02", drawdown="0.15"),
        simulation("0.05", drawdown="0.12"),
    )
    aggregate, failed = aggregate_blocks(base, base, base)
    assert aggregate["total_return"] == Decimal("0.147255200000000000")
    assert aggregate["round_trips"] == 20
    assert aggregate["maximum_drawdown"] == Decimal("0.2")
    assert aggregate["positive_half_years"] == 4
    assert aggregate["wins"] == 12
    assert aggregate["losses"] == 8
    assert aggregate["fees"] == Decimal("4.000000000000000000")
    assert aggregate["slippage"] == Decimal("8.000000000000000000")
    assert failed == ()


def test_report_json_is_canonical_and_order_independent() -> None:
    left = {"decimal": Decimal("0.050"), "time": NOW, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "time": NOW, "decimal": Decimal("0.050")}
    assert report_json(left) == report_json(right)
    assert '"decimal":"0.050"' in report_json(left)
    assert '"time":"2040-01-01T00:00:00+00:00"' in report_json(left)


def test_exactly_four_accumulating_folds_are_frozen_chronologically() -> None:
    assert ACCUMULATING_FOLD_NAMES == (
        "202201_accumulating",
        "202207_accumulating",
        "202301_accumulating",
        "202307_accumulating",
    )


@pytest.mark.parametrize("value", [1, 15, 25])
def test_domain_candidate_cap_accepts_only_inclusive_integer_boundaries(
    value: int,
) -> None:
    assert validate_maximum_candidates(value) == value


@pytest.mark.parametrize("value", [True, False, 1.5, Decimal("1"), "1", 0, -1, 26])
def test_domain_candidate_cap_rejects_non_integer_and_out_of_range_values(
    value: object,
) -> None:
    with pytest.raises(DevelopmentReauditError) as raised:
        validate_maximum_candidates(value)
    assert raised.value.code == "invalid_maximum_candidates"


@pytest.mark.asyncio
async def test_domain_candidate_cap_fails_before_repository_access() -> None:
    with pytest.raises(DevelopmentReauditError) as raised:
        await build_development_reaudit_report(
            session=cast(AsyncSession, object()),
            campaign_id=UUID(int=1),
            maximum_candidates=cast(int, 1.5),
        )
    assert raised.value.code == "invalid_maximum_candidates"


@pytest.mark.parametrize("value", ["0", "-1", "26", "1.5", "true", "01"])
def test_cli_rejects_invalid_candidate_caps(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        maximum_candidates(value)


@pytest.mark.parametrize("value", ["1", "15", "25"])
def test_cli_accepts_candidate_cap_boundaries(value: str) -> None:
    assert maximum_candidates(value) == int(value)
