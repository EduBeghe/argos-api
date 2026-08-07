from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ml.campaign.contracts import (
    ECONOMIC_V2_CONFIDENCE_FRACTIONS,
    ECONOMIC_V2_HOLDING_MULTIPLIERS,
    ECONOMIC_V2_REGIME_POLICIES,
    ECONOMIC_V2_STRATEGY_POLICY_VERSION,
)
from app.ml.campaign.economic_v2_policy import strategy_policy_grid
from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.economic_policy_contract import (
    economic_policy_evidence_compatibility,
    economic_policy_search_contract,
)
from app.ml.refinement.models import MLRefinementEvaluationRecord
from app.ml.refinement.persistence import (
    effective_evaluation_fingerprint,
    existing_run_evaluation,
    reusable_evaluation,
)
from app.ml.refinement.planning import build_refinement_plan
from tests.unit.ml.test_refinement_contracts import request_payload


def fingerprint(
    *, candidate: str = "c" * 64, version: str = "1.9.0", seed: int = 1729
) -> str:
    return effective_evaluation_fingerprint(
        dataset_fingerprint="d" * 64,
        candidate_fingerprint=candidate,
        engine="ridge",
        versions={"scikit-learn": version, "lightgbm": "4", "catboost": "1"},
        deterministic_seed=seed,
    )


def test_effective_identity_is_stable_for_compatible_evidence() -> None:
    assert fingerprint() == fingerprint()


def test_effective_identity_changes_for_versions_seed_or_candidate_contract() -> None:
    baseline = fingerprint()
    assert fingerprint(version="2.0.0") != baseline
    assert fingerprint(seed=1730) != baseline
    # Costs, objective, variables and hyperparameters are all frozen into the
    # candidate fingerprint by the planner.
    assert fingerprint(candidate="x" * 64) != baseline


def test_candidate_identity_changes_for_cost_objective_features_or_parameters() -> None:
    baseline_payload = request_payload()
    baseline_payload["future_durations_seconds"] = [1800]
    baseline_payload["prediction_objectives"] = ["future_return_regression"]
    baseline_payload["feature_selection"] = {"strategies": [{"strategy": "all"}]}
    baseline_payload["search_space"] = [
        {
            "engine": "ridge",
            "alpha": {"values": [1.0]},
            "fit_intercept": {"values": [True]},
        }
    ]
    baseline_payload["maximum_trials"] = 1
    baseline = (
        build_refinement_plan(MLRefinementRequest.model_validate(baseline_payload))
        .evaluations[0]
        .fingerprint
    )
    mutations: list[tuple[str, object]] = [
        ("fee_rate", "0.002"),
        ("prediction_objectives", ["future_direction_classification"]),
        ("feature_selection", {"strategies": [{"strategy": "top_k", "values": [10]}]}),
        (
            "search_space",
            [
                {
                    "engine": "ridge",
                    "alpha": {"values": [2.0]},
                    "fit_intercept": {"values": [True]},
                }
            ],
        ),
    ]
    for field, value in mutations:
        payload = dict(baseline_payload)
        payload[field] = value
        changed = (
            build_refinement_plan(MLRefinementRequest.model_validate(payload))
            .evaluations[0]
            .fingerprint
        )
        assert changed != baseline


def _request_with_policy(**changes: object) -> MLRefinementRequest:
    payload = request_payload()
    payload["future_durations_seconds"] = [3600]
    payload["prediction_objectives"] = ["future_return_regression"]
    payload["feature_selection"] = {"strategies": [{"strategy": "all"}]}
    payload["search_space"] = [
        {
            "engine": "ridge",
            "alpha": {"values": [1.0]},
            "fit_intercept": {"values": [True]},
        }
    ]
    payload["maximum_trials"] = 1
    policy: dict[str, object] = {
        "confidence_fractions": list(ECONOMIC_V2_CONFIDENCE_FRACTIONS),
        "holding_multipliers": list(ECONOMIC_V2_HOLDING_MULTIPLIERS),
        "regime_policies": list(ECONOMIC_V2_REGIME_POLICIES),
        "version": ECONOMIC_V2_STRATEGY_POLICY_VERSION,
    }
    policy.update(changes)
    payload["economic_policy_search"] = policy
    return MLRefinementRequest.model_validate(payload)


def _candidate_fingerprint(request: MLRefinementRequest) -> str:
    return build_refinement_plan(request).evaluations[0].fingerprint


def test_economic_policy_identity_is_order_independent_and_separate_from_legacy() -> (
    None
):
    request = _request_with_policy()
    reordered = _request_with_policy(
        confidence_fractions=tuple(reversed(ECONOMIC_V2_CONFIDENCE_FRACTIONS)),
        holding_multipliers=tuple(reversed(ECONOMIC_V2_HOLDING_MULTIPLIERS)),
        regime_policies=tuple(reversed(ECONOMIC_V2_REGIME_POLICIES)),
    )
    legacy = request.model_copy(update={"economic_policy_search": None})

    assert _candidate_fingerprint(request) == _candidate_fingerprint(reordered)
    assert _candidate_fingerprint(request) != _candidate_fingerprint(legacy)


def test_each_economic_policy_contract_dimension_changes_candidate_identity() -> None:
    baseline = _candidate_fingerprint(_request_with_policy())
    variants = (
        _request_with_policy(version=ECONOMIC_V2_STRATEGY_POLICY_VERSION + 1),
        _request_with_policy(confidence_fractions=(Decimal("0.25"),)),
        _request_with_policy(holding_multipliers=(Decimal("3"),)),
        _request_with_policy(regime_policies=(ECONOMIC_V2_REGIME_POLICIES[0],)),
    )
    assert all(_candidate_fingerprint(item) != baseline for item in variants)


def _selected_policy(
    request: MLRefinementRequest, **changes: object
) -> dict[str, object]:
    assert request.economic_policy_search is not None
    policy = strategy_policy_grid(3600, request.economic_policy_search)[0]
    selected: dict[str, object] = {
        "policy_fingerprint": policy.fingerprint,
        "strategy_policy_version": policy.version,
        "confidence_fraction": str(policy.confidence_fraction),
        "holding_multiplier": str(policy.holding_multiplier),
        "holding_candles": policy.holding_candles,
        "regime_policy": policy.regime_policy,
    }
    selected.update(changes)
    return selected


def _valid_metrics(request: MLRefinementRequest) -> dict[str, object]:
    assert request.economic_policy_search is not None
    aggregates = [
        {
            "policy_fingerprint": policy.fingerprint,
            "strategy_policy_version": policy.version,
            "confidence_fraction": str(policy.confidence_fraction),
            "holding_multiplier": str(policy.holding_multiplier),
            "holding_candles": policy.holding_candles,
            "regime_policy": policy.regime_policy,
        }
        for policy in strategy_policy_grid(3600, request.economic_policy_search)
    ]
    return {
        "economic_policy_search_contract": economic_policy_search_contract(
            request.economic_policy_search
        ),
        "economic_policy_v2_aggregates": aggregates,
        "selected_economic_policy_v2": dict(aggregates[0]),
    }


def test_semantic_reuse_guard_rejects_missing_malformed_and_out_of_grid_policy() -> (
    None
):
    request = _request_with_policy()
    missing = _valid_metrics(request)
    missing.pop("economic_policy_search_contract")
    malformed_selected = _valid_metrics(request)
    malformed_selected["selected_economic_policy_v2"] = {}
    wrong_version = _valid_metrics(request)
    aggregates = wrong_version["economic_policy_v2_aggregates"]
    assert isinstance(aggregates, list)
    aggregates[0] = {
        **aggregates[0],
        "strategy_policy_version": 99,
    }
    cases: tuple[tuple[dict[str, object], str], ...] = (
        (missing, "missing_economic_policy_contract"),
        (malformed_selected, "invalid_selected_economic_policy"),
        (wrong_version, "invalid_economic_policy_coverage"),
    )
    for metrics, reason in cases:
        compatibility = economic_policy_evidence_compatibility(
            aggregate_metrics=metrics,
            request=request,
            future_duration_seconds=3600,
        )
        assert not compatibility.compatible
        assert compatibility.reason_code == reason

    assert economic_policy_evidence_compatibility(
        aggregate_metrics=_valid_metrics(request),
        request=request,
        future_duration_seconds=3600,
    ).compatible


def test_legacy_reuse_guard_rejects_v2_evidence() -> None:
    legacy = _request_with_policy().model_copy(update={"economic_policy_search": None})
    compatibility = economic_policy_evidence_compatibility(
        aggregate_metrics={
            "selected_economic_policy_v2": _selected_policy(_request_with_policy())
        },
        request=legacy,
    )
    assert compatibility.reason_code == "legacy_request_has_v2_evidence"


@pytest.mark.asyncio
async def test_existing_and_cross_run_reuse_apply_the_same_semantic_guard() -> None:
    request = _request_with_policy()
    row = cast(
        MLRefinementEvaluationRecord,
        SimpleNamespace(
            id=uuid4(),
            run_id=uuid4(),
            aggregate_metrics={},
            future_duration_seconds=3600,
            created_at=None,
        ),
    )
    session = AsyncMock()
    session.scalar.return_value = row

    assert await existing_run_evaluation(session, row.run_id, "f" * 64, request) is None

    session.scalars.return_value = SimpleNamespace(all=lambda: [row])
    assert await reusable_evaluation(session, "f" * 64, request) is None

    row.aggregate_metrics = _valid_metrics(request)
    session.scalar.return_value = row
    assert await existing_run_evaluation(session, row.run_id, "f" * 64, request) is row
    assert await reusable_evaluation(session, "f" * 64, request) is row


def test_reduced_policy_grid_is_exact_and_order_independent() -> None:
    request = _request_with_policy(
        confidence_fractions=(Decimal("0.20"), Decimal("0.05")),
        holding_multipliers=(Decimal("1.5"), Decimal("0.5")),
        regime_policies=("trend_aligned", "all_regimes"),
    )
    assert request.economic_policy_search is not None
    policies = strategy_policy_grid(3600, request.economic_policy_search)
    assert len(policies) == 8
    assert len({item.fingerprint for item in policies}) == 8
    reordered = _request_with_policy(
        confidence_fractions=tuple(
            reversed(request.economic_policy_search.confidence_fractions)
        ),
        holding_multipliers=tuple(
            reversed(request.economic_policy_search.holding_multipliers)
        ),
        regime_policies=tuple(reversed(request.economic_policy_search.regime_policies)),
    )
    assert reordered.economic_policy_search is not None
    assert policies == strategy_policy_grid(3600, reordered.economic_policy_search)
    assert economic_policy_search_contract(request.economic_policy_search) == (
        economic_policy_search_contract(reordered.economic_policy_search)
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"confidence_fractions": ()},
        {"confidence_fractions": (Decimal("0.1"), Decimal("0.1"))},
        {"confidence_fractions": (Decimal("0"),)},
        {"holding_multipliers": (Decimal("0"),)},
        {"regime_policies": ("unsupported",)},
    ),
)
def test_invalid_policy_grid_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _request_with_policy(**changes)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "extra", "corrupt_fingerprint", "selected_missing"),
)
def test_policy_coverage_must_exactly_match_requested_grid(mutation: str) -> None:
    request = _request_with_policy(
        confidence_fractions=(Decimal("0.10"),),
        holding_multipliers=(Decimal("1"),),
        regime_policies=("all_regimes",),
    )
    metrics = _valid_metrics(request)
    aggregates = metrics["economic_policy_v2_aggregates"]
    assert isinstance(aggregates, list)
    if mutation == "missing":
        aggregates.clear()
    elif mutation in {"duplicate", "extra"}:
        aggregates.append(dict(aggregates[0]))
    elif mutation == "corrupt_fingerprint":
        aggregates[0]["policy_fingerprint"] = "x" * 64
    else:
        selected = dict(aggregates[0])
        selected["holding_candles"] = 99
        metrics["selected_economic_policy_v2"] = selected

    compatibility = economic_policy_evidence_compatibility(
        aggregate_metrics=metrics,
        request=request,
        future_duration_seconds=3600,
    )
    assert not compatibility.compatible
    assert compatibility.reason_code in {
        "invalid_economic_policy_coverage",
        "invalid_selected_economic_policy",
    }


def test_contract_hash_and_complete_grid_must_match_request() -> None:
    request = _request_with_policy()
    metrics = _valid_metrics(request)
    provenance = metrics["economic_policy_search_contract"]
    assert isinstance(provenance, dict)
    provenance["fingerprint"] = "0" * 64
    compatibility = economic_policy_evidence_compatibility(
        aggregate_metrics=metrics,
        request=request,
        future_duration_seconds=3600,
    )
    assert compatibility.reason_code == "economic_policy_contract_mismatch"

    other_request = _request_with_policy(confidence_fractions=(Decimal("0.10"),))
    compatibility = economic_policy_evidence_compatibility(
        aggregate_metrics=_valid_metrics(request),
        request=other_request,
        future_duration_seconds=3600,
    )
    assert compatibility.reason_code == "economic_policy_contract_mismatch"
