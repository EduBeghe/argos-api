from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.ml.campaign.contracts import (
    ECONOMIC_V2_CONFIDENCE_FRACTIONS,
    ECONOMIC_V2_HOLDING_MULTIPLIERS,
    ECONOMIC_V2_REGIME_POLICIES,
    ECONOMIC_V2_STRATEGY_POLICY_VERSION,
)
from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.economic_policy_contract import (
    economic_policy_evidence_compatibility,
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
        _request_with_policy(regime_policies=("different_regime",)),
    )
    assert all(_candidate_fingerprint(item) != baseline for item in variants)


def _selected_policy(**changes: object) -> dict[str, object]:
    selected: dict[str, object] = {
        "policy_fingerprint": "p" * 64,
        "strategy_policy_version": ECONOMIC_V2_STRATEGY_POLICY_VERSION,
        "confidence_fraction": str(ECONOMIC_V2_CONFIDENCE_FRACTIONS[0]),
        "holding_multiplier": str(ECONOMIC_V2_HOLDING_MULTIPLIERS[0]),
        "regime_policy": ECONOMIC_V2_REGIME_POLICIES[0],
    }
    selected.update(changes)
    return selected


def test_semantic_reuse_guard_rejects_missing_malformed_and_out_of_grid_policy() -> (
    None
):
    request = _request_with_policy()
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({}, "missing_selected_economic_policy"),
        ({"selected_economic_policy_v2": {}}, "missing_selected_economic_policy"),
        (
            {
                "selected_economic_policy_v2": _selected_policy(
                    strategy_policy_version=99
                )
            },
            "economic_policy_version_mismatch",
        ),
        (
            {
                "selected_economic_policy_v2": _selected_policy(
                    confidence_fraction="0.99"
                )
            },
            "selected_economic_policy_outside_grid",
        ),
    )
    for metrics, reason in cases:
        compatibility = economic_policy_evidence_compatibility(
            aggregate_metrics=metrics,
            request=request,
        )
        assert not compatibility.compatible
        assert compatibility.reason_code == reason

    assert economic_policy_evidence_compatibility(
        aggregate_metrics={"selected_economic_policy_v2": _selected_policy()},
        request=request,
    ).compatible


def test_legacy_reuse_guard_rejects_v2_evidence() -> None:
    legacy = _request_with_policy().model_copy(update={"economic_policy_search": None})
    compatibility = economic_policy_evidence_compatibility(
        aggregate_metrics={"selected_economic_policy_v2": _selected_policy()},
        request=legacy,
    )
    assert compatibility.reason_code == "legacy_request_has_v2_evidence"


@pytest.mark.asyncio
async def test_existing_and_cross_run_reuse_apply_the_same_semantic_guard() -> None:
    request = _request_with_policy()
    row = cast(
        MLRefinementEvaluationRecord,
        SimpleNamespace(
            id=uuid4(), run_id=uuid4(), aggregate_metrics={}, created_at=None
        ),
    )
    session = AsyncMock()
    session.scalar.return_value = row

    assert await existing_run_evaluation(session, row.run_id, "f" * 64, request) is None

    session.scalars.return_value = SimpleNamespace(all=lambda: [row])
    assert await reusable_evaluation(session, "f" * 64, request) is None

    row.aggregate_metrics = {"selected_economic_policy_v2": _selected_policy()}
    session.scalar.return_value = row
    assert await existing_run_evaluation(session, row.run_id, "f" * 64, request) is row
    assert await reusable_evaluation(session, "f" * 64, request) is row
