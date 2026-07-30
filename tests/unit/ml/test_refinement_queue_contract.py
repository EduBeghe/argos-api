from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.jobs.process_ml_refinement_queue import (
    build_parser,
    heartbeat_interval_seconds,
)
from app.ml.contract import FUTURE_DURATIONS
from app.ml.refinement.contract_service import refinement_contract
from app.ml.refinement.models import MLRefinementRunRecord
from app.ml.refinement.schemas import RefinementRunResponse
from app.ml.registry import VARIABLE_REGISTRY


def test_contract_is_derived_from_authoritative_registry() -> None:
    contract = refinement_contract()

    assert [item.name for item in contract.variables] == [
        item.name for item in VARIABLE_REGISTRY
    ]
    assert contract.allowed_future_durations_seconds == [
        int(item.total_seconds()) for item in FUTURE_DURATIONS
    ]
    assert contract.ui_defaults.fee_rate == Decimal("0.001")
    assert contract.ui_defaults.slippage_rate == Decimal("0.0005")
    assert contract.ui_defaults.deterministic_seed == 1729


@pytest.mark.parametrize("value", ["1", "300"])
def test_poll_seconds_accepts_inclusive_boundaries(value: str) -> None:
    assert build_parser().parse_args(["--poll-seconds", value]).poll_seconds == int(
        value
    )


@pytest.mark.parametrize("value", ["0", "301"])
def test_poll_seconds_rejects_outside_boundaries(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--poll-seconds", value])


@pytest.mark.parametrize("value", ["300", "86400"])
def test_stale_seconds_accepts_inclusive_boundaries(value: str) -> None:
    assert build_parser().parse_args(
        ["--stale-after-seconds", value]
    ).stale_after_seconds == int(value)


@pytest.mark.parametrize("value", ["299", "86401"])
def test_stale_seconds_rejects_outside_boundaries(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--stale-after-seconds", value])


@pytest.mark.parametrize(
    ("stale_after", "expected"), [(300, 60.0), (7200, 60.0), (3, 1.0)]
)
def test_heartbeat_interval_is_bounded_below_stale_threshold(
    stale_after: int, expected: float
) -> None:
    assert heartbeat_interval_seconds(stale_after) == expected


def test_run_response_exposes_bounded_progress_and_cancellation() -> None:
    timestamp = datetime(2026, 7, 29, tzinfo=UTC)
    record = MLRefinementRunRecord(
        id=uuid4(),
        experiment_key="a" * 64,
        source_screening_run_id=uuid4(),
        source_screening_result_id=uuid4(),
        status="running",
        contract_version=2,
        classification_version=2,
        dataset_fingerprint="b" * 64,
        variable_registry_version=1,
        missing_data_policy_version=1,
        library_versions={},
        normalized_configuration={},
        notes=None,
        maximum_trials=10,
        maximum_runtime_seconds=60,
        combinations_planned=10,
        evaluations_selected=4,
        evaluations_computed=3,
        evaluations_reused=1,
        evaluations_omitted=6,
        budget_exhausted=True,
        termination_reason=None,
        started_at=timestamp,
        completed_at=None,
        failed_at=None,
        failure_type=None,
        failure_message=None,
        worker_id="worker",
        claimed_at=timestamp,
        heartbeat_at=timestamp,
        cancellation_requested_at=timestamp,
        elapsed_seconds=None,
        created_at=timestamp,
    )

    response = RefinementRunResponse.model_validate(record)

    assert response.work_completed == 4
    assert response.progress_fraction == Decimal("1")
    assert response.cancellation_requested is True
