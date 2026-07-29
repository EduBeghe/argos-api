from decimal import Decimal
from typing import cast
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.planning import build_refinement_plan


def request_payload() -> dict[str, object]:
    return {
        "source_screening_run_id": str(uuid4()),
        "source_screening_result_id": str(uuid4()),
        "engines": ["ridge"],
        "future_durations_seconds": [1800, 3600],
        "search_space": [
            {
                "engine": "ridge",
                "alpha": {"values": [0.1, 1.0, 10.0]},
                "fit_intercept": {"values": [True, False]},
            }
        ],
        "feature_selection": {
            "strategies": [
                {"strategy": "all"},
                {"strategy": "top_k", "values": [10, 20]},
            ]
        },
        "prediction_objectives": [
            "future_return_regression",
            "future_direction_classification",
            "cost_adjusted_action_classification",
        ],
        "confidence_buckets": ["0.05", "0.10", "0.20"],
        "fee_rate": "0.001",
        "slippage_rate": "0.0005",
        "maximum_trials": 12,
        "maximum_runtime_seconds": 3600,
        "deterministic_seed": 1729,
        "notes": "unit test",
    }


def test_configuration_normalization_and_fingerprint_are_deterministic() -> None:
    payload = request_payload()
    first = MLRefinementRequest.model_validate(payload)
    payload["engines"] = list(reversed(cast(list[str], payload["engines"])))
    payload["future_durations_seconds"] = [3600, 1800]
    payload["confidence_buckets"] = ["0.20", "0.05", "0.10"]
    second = MLRefinementRequest.model_validate(payload)
    assert first.fingerprint() == second.fingerprint()
    assert first.normalized_payload() == second.normalized_payload()


def test_material_change_changes_fingerprint() -> None:
    first = MLRefinementRequest.model_validate(request_payload())
    payload = request_payload()
    payload["fee_rate"] = "0.002"
    second = MLRefinementRequest.model_validate(payload)
    assert first.fingerprint() != second.fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("engines", []),
        ("future_durations_seconds", []),
        ("future_durations_seconds", [1]),
        ("prediction_objectives", []),
        ("confidence_buckets", []),
        ("confidence_buckets", ["0"]),
        ("fee_rate", "-0.1"),
        ("maximum_trials", 0),
        ("maximum_runtime_seconds", 0),
    ],
)
def test_invalid_contract_values_fail_before_execution(
    field: str, value: object
) -> None:
    payload = request_payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        MLRefinementRequest.model_validate(payload)


def test_unknown_hyperparameter_and_invalid_range_are_rejected() -> None:
    payload = request_payload()
    payload["search_space"][0]["unknown"] = {"values": [1]}  # type: ignore[index]
    with pytest.raises(ValidationError):
        MLRefinementRequest.model_validate(payload)
    payload = request_payload()
    payload["search_space"][0]["alpha"] = {  # type: ignore[index]
        "minimum": 2,
        "maximum": 1,
        "step": 1,
    }
    with pytest.raises(ValidationError):
        MLRefinementRequest.model_validate(payload)


def test_plan_is_bounded_distributed_and_deterministic() -> None:
    request = MLRefinementRequest.model_validate(request_payload())
    first = build_refinement_plan(request)
    second = build_refinement_plan(request)
    assert first == second
    assert first.total_combinations == 108
    assert len(first.evaluations) == 12
    assert first.omitted_by_trial_budget == 96
    assert len({item.fingerprint for item in first.evaluations}) == 12
    assert {item.future_duration_seconds for item in first.evaluations} == {1800, 3600}
    assert len({item.objective for item in first.evaluations}) > 1


def test_all_three_engine_search_contracts_are_supported() -> None:
    payload = request_payload()
    payload["engines"] = ["ridge", "lightgbm", "catboost"]
    payload["future_durations_seconds"] = [1800]
    payload["prediction_objectives"] = ["future_return_regression"]
    payload["feature_selection"] = {"strategies": [{"strategy": "all"}]}
    payload["maximum_trials"] = 3
    payload["search_space"] = [
        payload["search_space"][0],  # type: ignore[index]
        {
            "engine": "lightgbm",
            "learning_rate": {"values": [0.05]},
            "n_estimators": {"values": [100]},
            "max_depth": {"values": [5]},
            "num_leaves": {"values": [24]},
            "min_child_samples": {"values": [20]},
            "subsample": {"values": [0.9]},
            "colsample_bytree": {"values": [0.9]},
            "reg_alpha": {"values": [0]},
            "reg_lambda": {"values": [1]},
        },
        {
            "engine": "catboost",
            "learning_rate": {"values": [0.05]},
            "iterations": {"values": [100]},
            "depth": {"values": [5]},
            "l2_leaf_reg": {"values": [3]},
            "random_strength": {"values": [1]},
            "bagging_temperature": {"values": [1]},
        },
    ]
    request = MLRefinementRequest.model_validate(payload)
    plan = build_refinement_plan(request)
    assert {item.engine.value for item in plan.evaluations} == {
        "ridge",
        "lightgbm",
        "catboost",
    }
    assert request.fee_rate == Decimal("0.001")


def test_large_cartesian_space_is_planned_with_bounded_memory() -> None:
    payload = request_payload()
    payload["engines"] = ["lightgbm"]
    payload["future_durations_seconds"] = [1800]
    payload["prediction_objectives"] = ["future_return_regression"]
    payload["feature_selection"] = {"strategies": [{"strategy": "all"}]}
    payload["maximum_trials"] = 7
    payload["search_space"] = [
        {
            "engine": "lightgbm",
            "learning_rate": {"minimum": 0.01, "maximum": 1.0, "step": 0.01},
            "n_estimators": {"minimum": 1, "maximum": 100, "step": 1},
            "max_depth": {"minimum": 1, "maximum": 100, "step": 1},
            "num_leaves": {"minimum": 2, "maximum": 101, "step": 1},
            "min_child_samples": {"minimum": 1, "maximum": 100, "step": 1},
            "subsample": {"values": [0.8, 0.9, 1.0]},
            "colsample_bytree": {"values": [0.8, 0.9, 1.0]},
            "reg_alpha": {"values": [0, 0.1, 1]},
            "reg_lambda": {"values": [0, 0.1, 1]},
        }
    ]
    request = MLRefinementRequest.model_validate(payload)

    plan = build_refinement_plan(request)

    assert plan.total_combinations == 810_000_000_000
    assert len(plan.evaluations) == 7
    assert plan.omitted_by_trial_budget == plan.total_combinations - 7
    assert plan == build_refinement_plan(request)
