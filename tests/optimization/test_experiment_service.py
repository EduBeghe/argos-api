from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

from app.optimization.experiment_service import build_experiment_key
from app.optimization.planning import PlannerAllocation
from app.optimization.refinement import RefinementBudget, split_feature_matrix
from tests.optimization.test_refinement import dataset, matrix


def key(
    *,
    fingerprint: str = "refinement-fingerprint",
    fee_rate: Decimal = Decimal("0.001"),
) -> str:
    source = replace(dataset(), fingerprint=fingerprint)
    _, segments = split_feature_matrix(matrix())
    return build_experiment_key(
        dataset=source,
        segments=segments,
        initial_cash=Decimal("10000"),
        fee_rate=fee_rate,
        slippage_rate=Decimal("0.0005"),
        budget=RefinementBudget(1, 1, 200, 60),
        decision_interval=timedelta(minutes=30),
        planner_allocation=PlannerAllocation(),
    )


def test_experiment_key_is_exact_stable_and_changes_with_dataset_or_costs() -> None:
    assert key() == key()
    assert key() != key(fingerprint="different")
    assert key() != key(fee_rate=Decimal("0.002"))


def test_experiment_key_changes_with_explicit_engine_or_planner_version() -> None:
    original = key()
    with patch("app.optimization.experiment_service.SIMULATION_ENGINE_VERSION", 2):
        assert key() != original
    with patch("app.optimization.experiment_service.REFINEMENT_PLANNER_VERSION", 2):
        assert key() != original
