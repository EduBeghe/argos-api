from enum import StrEnum


class Asset(StrEnum):
    """Assets supported by the initial market scope."""

    BTC_USDT = "BTC/USDT"
    ETH_USDT = "ETH/USDT"


class Provider(StrEnum):
    """External market-data providers."""

    BINANCE = "binance"


class MarketType(StrEnum):
    """Market structures represented by Argos."""

    SPOT = "spot"


class Interval(StrEnum):
    """Supported market-candle intervals."""

    FIVE_MINUTES = "5m"
