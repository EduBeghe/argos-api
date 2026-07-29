from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.persistence import effective_evaluation_fingerprint
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
