from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.jobs import refine_ml_screening_result as job
from app.ml.contract import DEVELOPMENT_END, DEVELOPMENT_START
from app.ml.reader import candle_columns
from app.ml.refinement.contracts import MLRefinementRequest
from tests.unit.ml.test_refinement_contracts import request_payload


@pytest.mark.asyncio
async def test_dry_and_real_job_dataset_loading_stop_before_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = MLRefinementRequest.model_validate(request_payload())
    requested_ranges: list[tuple[object, object]] = []

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

    async def reader(**kwargs: object) -> object:
        requested_ranges.append((kwargs["start"], kwargs["end"]))
        return SimpleNamespace(
            columns=candle_columns(
                open_times=[],
                close_times=[],
                open_prices=[],
                high_prices=[],
                low_prices=[],
                close_prices=[],
                volumes=[],
            )
        )

    monkeypatch.setattr(job, "resolve_source", source)
    monkeypatch.setattr(job, "read_ml_experiment_candles", reader)
    session = cast(AsyncSession, object())

    dry_dataset = await job.load_refinement_dataset(session, request)
    real_dataset = await job.load_refinement_dataset(session, request)

    assert requested_ranges == [
        (DEVELOPMENT_START, DEVELOPMENT_END),
        (DEVELOPMENT_START, DEVELOPMENT_END),
    ]
    assert dry_dataset.metadata.requested_end == DEVELOPMENT_END
    assert real_dataset.metadata.requested_end == DEVELOPMENT_END


def test_refinement_contract_has_no_final_test_override() -> None:
    payload = request_payload()
    payload["requested_end"] = "2026-01-01T00:00:00Z"
    with pytest.raises(ValidationError):
        MLRefinementRequest.model_validate(payload)
