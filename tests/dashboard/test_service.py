from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.dashboard.service import build_dashboard_overview
from app.market.catalog import MARKET_INSTRUMENTS
from app.market.schemas import MarketStatusResponse

pytestmark = pytest.mark.asyncio
NOW = datetime(2026, 7, 18, 20, tzinfo=UTC)


async def test_overview_without_run_keeps_order_and_partial_data() -> None:
    session = AsyncMock()
    with (
        patch(
            "app.dashboard.service.get_market_status_aggregates",
            AsyncMock(return_value=[]),
        ),
        patch(
            "app.dashboard.service.get_latest_scores_batch",
            AsyncMock(return_value=()),
        ),
        patch(
            "app.dashboard.service.get_latest_decisions_batch",
            AsyncMock(return_value=()),
        ),
        patch(
            "app.dashboard.service.get_active_run",
            AsyncMock(return_value=None),
        ),
        patch(
            "app.dashboard.service.get_open_positions_batch",
            AsyncMock(),
        ) as positions,
    ):
        result = await build_dashboard_overview(session=session, generated_at=NOW)

    assert result.generated_at == NOW
    assert result.paper.run is None
    assert result.paper.portfolio is None
    assert result.paper.performance is None
    assert [item.instrument.asset for item in result.instruments] == [
        definition.asset for definition in MARKET_INSTRUMENTS
    ]
    assert all(item.market_status is not None for item in result.instruments)
    assert all(item.latest_score is None for item in result.instruments)
    assert all(item.latest_decision is None for item in result.instruments)
    assert all(item.open_position is None for item in result.instruments)
    assert result.system.instruments_total == 2
    assert result.system.instruments_fresh == 0
    assert result.system.market_data_fresh is False
    positions.assert_not_awaited()


async def test_visible_non_ingested_catalog_card_does_not_affect_freshness() -> None:
    session = AsyncMock()
    extra = replace(
        MARKET_INSTRUMENTS[0],
        display_symbol="FUTURE/USDT",
        display_name="Future configured instrument",
        provider_symbol="FUTUREUSDT",
        ingestion_enabled=False,
        scoring_enabled=False,
        decision_enabled=False,
        display_order=3,
    )
    statuses = [
        MarketStatusResponse(
            asset=definition.asset,
            interval=definition.interval,
            candle_count=1,
            latest_open_time=NOW,
            latest_close_time=NOW,
            latest_ingested_at=NOW,
            age_seconds=0,
            is_fresh=True,
        )
        for definition in MARKET_INSTRUMENTS
    ]
    with (
        patch(
            "app.dashboard.service.market_instruments",
            return_value=(*MARKET_INSTRUMENTS, extra),
        ),
        patch(
            "app.dashboard.service.get_market_status_aggregates",
            AsyncMock(return_value=[]),
        ),
        patch(
            "app.dashboard.service.build_market_status",
            return_value=statuses,
        ),
        patch(
            "app.dashboard.service.get_latest_scores_batch",
            AsyncMock(return_value=()),
        ),
        patch(
            "app.dashboard.service.get_latest_decisions_batch",
            AsyncMock(return_value=()),
        ),
        patch(
            "app.dashboard.service.get_active_run",
            AsyncMock(return_value=None),
        ),
    ):
        result = await build_dashboard_overview(session=session, generated_at=NOW)

    assert [item.instrument.display_order for item in result.instruments] == [1, 2, 3]
    assert result.instruments[-1].instrument.display_symbol == "FUTURE/USDT"
    assert result.instruments[-1].market_status is None
    assert result.system.instruments_total == 3
    assert result.system.instruments_fresh == 2
    assert result.system.market_data_fresh is True
