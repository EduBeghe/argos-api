from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.paper.evaluation_service import evaluate_paper_run
from app.paper.persistence import PaperTradingRunRecord


@pytest.mark.asyncio
async def test_run_without_snapshot_is_typed_neutral_and_skips_evidence_queries() -> (
    None
):
    run = cast(
        PaperTradingRunRecord,
        SimpleNamespace(
            id=uuid4(),
            status="active",
            account_currency="USDT",
            configuration_version=1,
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            started_at=datetime(2026, 7, 19, tzinfo=UTC),
        ),
    )
    session = cast(AsyncSession, object())
    with (
        patch(
            "app.paper.evaluation_service.latest_snapshot_and_count",
            AsyncMock(return_value=(None, 0)),
        ),
        patch(
            "app.paper.evaluation_service.evaluation_decisions", AsyncMock()
        ) as decisions,
    ):
        result = await evaluate_paper_run(session=session, run=run)

    assert result["run"]["latest_snapshot_id"] is None  # type: ignore[index]
    assert result["strategy"]["unavailable_reason"] == "no_equity_snapshot"  # type: ignore[index]
    assert result["benchmark"]["status"] == "unavailable"  # type: ignore[index]
    assert result["comparison"]["status"] == "unavailable"  # type: ignore[index]
    decisions.assert_not_awaited()
