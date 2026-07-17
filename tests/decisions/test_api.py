from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.database.session import get_db_session
from app.decisions.models import DecisionSnapshot
from app.market.enums import Asset, Interval, MarketType, Provider
from app.strategies.baseline_long_only import StrategyAction
from main import app
from tests.decisions.factories import make_score


async def fake_session() -> object:
    yield object()


def snapshot() -> DecisionSnapshot:
    score = make_score()
    return DecisionSnapshot(
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        decision_timestamp=datetime(2026, 7, 15, 12, 30, tzinfo=UTC),
        source_candle_open_time=score.candle_open_time,
        source_market_price=Decimal("65000.123456789012345678"),
        source_score_id=score.id,
        source_score_model_name=score.model_name,
        source_score_model_version=score.model_version,
        source_overall_score=score.overall_score,
        strategy_name="baseline_long_only",
        strategy_version=1,
        configuration_version=1,
        action=StrategyAction.OPEN_POSITION,
        requested_target_portfolio_percentage=Decimal("15.00"),
        requested_position_percentage=None,
        applied_rule="entry_threshold_met",
        explanation="Entry threshold met; request a new long position.",
        position_is_open=False,
    )


def test_latest_decision_returns_persisted_intent(client: TestClient) -> None:
    query = AsyncMock(return_value=snapshot())
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.market.api.get_latest_decision_snapshot", query):
            response = client.get(
                "/market/decisions/latest",
                params={"asset": "BTC/USDT", "interval": "5m"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["action"] == "open_position"
    assert response.json()["requested_target_portfolio_percentage"] == "15.00"
    assert response.json()["source_overall_score"] == "0.60"
    query.assert_awaited_once()


def test_latest_decision_returns_404_without_snapshot(client: TestClient) -> None:
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch(
            "app.market.api.get_latest_decision_snapshot",
            AsyncMock(return_value=None),
        ):
            response = client.get(
                "/market/decisions/latest",
                params={"asset": "ETH/USDT", "interval": "5m"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
