from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.jobs import process_ml_refinement_queue
from app.ml.protected.contracts import (
    DECISION_CADENCE_SECONDS,
    FINAL_VALIDATION_POLICY_VERSION,
    FINAL_VALIDATION_SIMULATION_VERSION,
    FinalValidationContractError,
    validate_final_validation_contract,
)
from app.ml.protected.models import MLFinalValidationRecord
from app.ml.protected.queue import ProtectedWorkError, _sha, retry_final_validation
from app.ml.protected.worker import process_final_validation


def compatible_inputs() -> dict[str, object]:
    return {
        "simulation_version": FINAL_VALIDATION_SIMULATION_VERSION,
        "policy_version": FINAL_VALIDATION_POLICY_VERSION,
        "decision_cadence_seconds": DECISION_CADENCE_SECONDS,
    }


def test_exact_current_final_validation_contract_is_compatible() -> None:
    validate_final_validation_contract(compatible_inputs())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("simulation_version", None),
        ("simulation_version", "2"),
        ("simulation_version", True),
        ("simulation_version", 2.0),
        ("simulation_version", FINAL_VALIDATION_SIMULATION_VERSION - 1),
        ("simulation_version", FINAL_VALIDATION_SIMULATION_VERSION + 1),
        ("policy_version", None),
        ("policy_version", "1"),
        ("policy_version", True),
        ("policy_version", 1.0),
        ("policy_version", FINAL_VALIDATION_POLICY_VERSION - 1),
        ("policy_version", FINAL_VALIDATION_POLICY_VERSION + 1),
        ("decision_cadence_seconds", None),
        ("decision_cadence_seconds", "1800"),
        ("decision_cadence_seconds", True),
        ("decision_cadence_seconds", 1800.0),
        ("decision_cadence_seconds", DECISION_CADENCE_SECONDS - 1),
        ("decision_cadence_seconds", DECISION_CADENCE_SECONDS + 1),
    ],
)
def test_final_validation_contract_rejects_incompatible_exact_types_and_values(
    field: str, value: object
) -> None:
    inputs = compatible_inputs()
    inputs[field] = value
    with pytest.raises(FinalValidationContractError) as captured:
        validate_final_validation_contract(inputs)
    assert captured.value.code == "incompatible_final_validation_contract"
    assert field in str(captured.value)


@pytest.mark.parametrize(
    "missing_field",
    ["simulation_version", "policy_version", "decision_cadence_seconds"],
)
def test_final_validation_contract_rejects_missing_fields(
    missing_field: str,
) -> None:
    inputs = compatible_inputs()
    del inputs[missing_field]
    with pytest.raises(FinalValidationContractError, match=missing_field):
        validate_final_validation_contract(inputs)


def test_final_validation_contract_reports_first_failure_deterministically() -> None:
    with pytest.raises(
        FinalValidationContractError, match="simulation_version"
    ) as captured:
        validate_final_validation_contract({})
    assert captured.value.code == "incompatible_final_validation_contract"


def test_current_contract_identity_is_deterministic_and_distinct_from_legacy() -> None:
    current = compatible_inputs()
    legacy = {
        **current,
        "simulation_version": FINAL_VALIDATION_SIMULATION_VERSION - 1,
    }
    assert _sha(current) == _sha(dict(reversed(tuple(current.items()))))
    assert _sha(current) != _sha(legacy)


@pytest.mark.asyncio
async def test_worker_rejects_legacy_contract_before_any_lookup_or_protected_read() -> (
    None
):
    record = cast(
        MLFinalValidationRecord,
        SimpleNamespace(
            status="running",
            worker_id="worker-a",
            frozen_inputs={
                **compatible_inputs(),
                "simulation_version": FINAL_VALIDATION_SIMULATION_VERSION - 1,
            },
            protected_data_read_at=None,
        ),
    )
    session = AsyncMock(spec=AsyncSession)
    with pytest.raises(FinalValidationContractError) as captured:
        await process_final_validation(session, record=record, worker_id="worker-a")
    assert captured.value.code == "incompatible_final_validation_contract"
    assert record.protected_data_read_at is None
    session.get.assert_not_awaited()
    session.refresh.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled", "completed"])
async def test_retry_rejects_incompatible_contract_without_mutation(
    status: str,
) -> None:
    record = SimpleNamespace(
        id=uuid4(),
        status=status,
        frozen_inputs={
            **compatible_inputs(),
            "simulation_version": FINAL_VALIDATION_SIMULATION_VERSION - 1,
        },
        protected_data_read_at=None,
        worker_id="legacy-worker",
        claimed_at="claimed",
        heartbeat_at="heartbeat",
        failure_code="legacy_failure",
        failure_message="legacy message",
        completed_at="completed",
        cancellation_requested_at="cancelled",
    )
    before = vars(record).copy()
    session = AsyncMock(spec=AsyncSession)
    session.get.return_value = record
    with pytest.raises(ProtectedWorkError) as captured:
        await retry_final_validation(session, record.id)
    assert captured.value.code == "incompatible_final_validation_contract"
    assert vars(record) == before
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_compatible_failed_pre_read_validation_remains_retryable() -> None:
    record = cast(
        MLFinalValidationRecord,
        SimpleNamespace(
            id=uuid4(),
            status="failed",
            frozen_inputs=compatible_inputs(),
            protected_data_read_at=None,
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
            failure_code="transient",
            failure_message="transient failure",
            completed_at=None,
        ),
    )
    session = AsyncMock(spec=AsyncSession)
    session.get.return_value = record
    result = await retry_final_validation(session, record.id)
    assert result is record
    assert record.status == "queued"
    assert record.failure_code is None
    assert record.failure_message is None
    session.commit.assert_awaited_once_with()


class _SessionContext:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def __aenter__(self) -> AsyncSession:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_background_worker_persists_stable_contract_failure_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = cast(
        MLFinalValidationRecord,
        SimpleNamespace(
            id=uuid4(),
            status="running",
            worker_id="worker-a",
            claimed_at="claimed",
            heartbeat_at="heartbeat",
            frozen_inputs={
                **compatible_inputs(),
                "simulation_version": FINAL_VALIDATION_SIMULATION_VERSION - 1,
            },
            protected_data_read_at=None,
            failure_code=None,
            failure_message=None,
        ),
    )
    session = AsyncMock(spec=AsyncSession)
    session.get.return_value = record
    factory = MagicMock(return_value=_SessionContext(session))
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "fail_stale_protected_work",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "claim_protected_work",
        AsyncMock(side_effect=[None, record]),
    )
    processed = await process_ml_refinement_queue.process_once(
        factory,
        worker_id="worker-a",
        stale_after=timedelta(hours=2),
    )
    assert processed is True
    assert record.status == "failed"
    assert record.failure_code == "incompatible_final_validation_contract"
    assert record.worker_id is None
    assert record.claimed_at is None
    assert record.heartbeat_at is None
    assert record.protected_data_read_at is None
    session.rollback.assert_awaited_once_with()
    session.commit.assert_awaited_once_with()
    session.add.assert_not_called()
