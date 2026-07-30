from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.jobs import refine_ml_screening_result as job
from app.ml.contract import FINAL_TEST_END
from app.ml.refinement.contracts import MLRefinementRequest
from tests.unit.ml.test_refinement_contracts import request_payload


@pytest.mark.asyncio
async def test_dry_and_real_job_use_the_canonical_screening_dataset_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = MLRefinementRequest.model_validate(request_payload())
    requested_instruments: list[object] = []

    async def source(*_args: object) -> tuple[object, object]:
        return (
            SimpleNamespace(
                asset="BTC/USDT",
                provider="binance",
                market_type="spot",
                interval="5m",
            ),
            object(),
        )

    async def loader(**kwargs: object) -> object:
        requested_instruments.append(kwargs["instrument"])
        return SimpleNamespace(metadata=SimpleNamespace(requested_end=FINAL_TEST_END))

    monkeypatch.setattr(job, "resolve_source", source)
    monkeypatch.setattr(job, "load_canonical_ml_experiment_dataset", loader)
    session = cast(AsyncSession, object())

    dry_dataset = await job.load_refinement_dataset(session, request)
    real_dataset = await job.load_refinement_dataset(session, request)

    assert len(requested_instruments) == 2
    assert requested_instruments[0] == requested_instruments[1]
    assert dry_dataset.metadata.requested_end == FINAL_TEST_END
    assert real_dataset.metadata.requested_end == FINAL_TEST_END


def test_refinement_contract_has_no_final_test_override() -> None:
    payload = request_payload()
    payload["requested_end"] = "2026-01-01T00:00:00Z"
    with pytest.raises(ValidationError):
        MLRefinementRequest.model_validate(payload)
