from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.paper.evaluation_service import evaluate_paper_run
from app.paper.models import PaperRunStatus, PaperTradingRun
from app.paper.performance import EquityPoint
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


@pytest.mark.asyncio
async def test_later_performance_point_cannot_cross_snapshot_boundary() -> None:
    boundary = datetime(2026, 7, 19, 12, tzinfo=UTC)
    run_id = uuid4()
    run = cast(
        PaperTradingRunRecord,
        SimpleNamespace(
            id=run_id,
            status="active",
            account_currency="USDT",
            configuration_version=1,
            initial_cash=Decimal("100"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            started_at=boundary - timedelta(hours=1),
        ),
    )
    domain_run = PaperTradingRun(
        id=run_id,
        status=PaperRunStatus.ACTIVE,
        account_currency="USDT",
        initial_cash=Decimal("100"),
        current_cash=Decimal("102"),
        configuration_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=run.started_at,
        stopped_at=None,
        stop_reason=None,
        predecessor_run_id=None,
    )
    snapshot = SimpleNamespace(id=uuid4(), scheduled_boundary=boundary, positions=[])
    points = (
        EquityPoint(boundary, Decimal("102")),
        EquityPoint(boundary + timedelta(minutes=30), Decimal("999")),
    )
    session = cast(AsyncSession, object())
    performance_query = AsyncMock(return_value=points)
    with (
        patch(
            "app.paper.evaluation_service.latest_snapshot_and_count",
            AsyncMock(return_value=(snapshot, 1)),
        ),
        patch(
            "app.paper.evaluation_service.evaluation_decisions",
            AsyncMock(return_value=[]),
        ),
        patch(
            "app.paper.evaluation_service.evaluation_trades",
            AsyncMock(return_value=[]),
        ),
        patch(
            "app.paper.evaluation_service.evaluation_closed_positions",
            AsyncMock(return_value=[]),
        ),
        patch("app.paper.evaluation_service.get_performance_points", performance_query),
        patch("app.paper.evaluation_service.run_domain", return_value=domain_run),
    ):
        result = await evaluate_paper_run(session=session, run=run)

    performance_query.assert_awaited_once_with(
        session, run_id=run_id, evaluated_until=boundary
    )
    assert result["run"]["evaluated_until"] == boundary  # type: ignore[index]
    assert result["strategy"]["ending_equity"] == Decimal(  # type: ignore[index]
        "102.000000000000000000"
    )
    assert result["strategy"]["total_return"] == Decimal(  # type: ignore[index]
        "0.020000000000000000"
    )
    assert result["strategy"]["current_drawdown"] == Decimal(  # type: ignore[index]
        "0.000000000000000000"
    )
