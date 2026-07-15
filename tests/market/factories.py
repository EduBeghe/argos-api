from datetime import UTC, datetime
from decimal import Decimal

from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.models import MarketCandle


def candle_data(**overrides: object) -> dict[str, object]:
    """Build valid market-candle input with optional field overrides."""

    data: dict[str, object] = {
        "asset": Asset.BTC_USDT,
        "provider": Provider.BINANCE,
        "market_type": MarketType.SPOT,
        "interval": Interval.FIVE_MINUTES,
        "open_time": datetime(2026, 7, 15, 12, 0, tzinfo=UTC),
        "close_time": datetime(2026, 7, 15, 12, 4, 59, 999000, tzinfo=UTC),
        "open_price": Decimal("65000.10000000"),
        "high_price": Decimal("65100.20000000"),
        "low_price": Decimal("64900.30000000"),
        "close_price": Decimal("65050.40000000"),
        "base_asset_volume": Decimal("12.34567890"),
        "quote_asset_volume": Decimal("802345.67890000"),
        "trade_count": 1000,
        "taker_buy_base_asset_volume": Decimal("6.12345678"),
        "taker_buy_quote_asset_volume": Decimal("401234.56780000"),
        "ingested_at": datetime(2026, 7, 15, 12, 5, 1, tzinfo=UTC),
    }
    data.update(overrides)
    return data


def make_candle(**overrides: object) -> MarketCandle:
    """Create a valid domain candle with optional field overrides."""

    return MarketCandle.model_validate(candle_data(**overrides))
