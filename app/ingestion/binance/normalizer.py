from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

from app.ingestion.binance.normalization_errors import (
    BinanceKlineDomainError,
    BinanceKlineFieldError,
    BinanceKlineStructureError,
)
from app.ingestion.binance.types import BinanceKline
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.models import MarketCandle

_KLINE_LENGTH = 12
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def normalize_kline(
    *,
    raw_kline: BinanceKline,
    asset: Asset,
    interval: Interval,
    ingested_at: datetime,
) -> MarketCandle:
    """Transform one raw Binance Spot kline into a validated MarketCandle."""

    if not isinstance(raw_kline, list) or len(raw_kline) != _KLINE_LENGTH:
        raise BinanceKlineStructureError(
            "Binance kline must be a list containing exactly 12 positions"
        )

    open_time = _parse_timestamp(raw_kline[0], index=0, field="open time")
    open_price = _parse_decimal(raw_kline[1], index=1, field="open price")
    high_price = _parse_decimal(raw_kline[2], index=2, field="high price")
    low_price = _parse_decimal(raw_kline[3], index=3, field="low price")
    close_price = _parse_decimal(raw_kline[4], index=4, field="close price")
    base_asset_volume = _parse_decimal(raw_kline[5], index=5, field="base asset volume")
    close_time = _parse_timestamp(raw_kline[6], index=6, field="close time")
    quote_asset_volume = _parse_decimal(
        raw_kline[7], index=7, field="quote asset volume"
    )
    trade_count = _parse_integer(raw_kline[8], index=8, field="trade count")
    taker_buy_base_asset_volume = _parse_decimal(
        raw_kline[9], index=9, field="taker buy base asset volume"
    )
    taker_buy_quote_asset_volume = _parse_decimal(
        raw_kline[10], index=10, field="taker buy quote asset volume"
    )
    _require_string(raw_kline[11], index=11, field="ignore")

    try:
        return MarketCandle(
            asset=asset,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=interval,
            open_time=open_time,
            close_time=close_time,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            base_asset_volume=base_asset_volume,
            quote_asset_volume=quote_asset_volume,
            trade_count=trade_count,
            taker_buy_base_asset_volume=taker_buy_base_asset_volume,
            taker_buy_quote_asset_volume=taker_buy_quote_asset_volume,
            ingested_at=ingested_at,
        )
    except ValueError as exc:
        raise BinanceKlineDomainError(
            "Normalized Binance kline violates the MarketCandle contract"
        ) from exc


def _parse_timestamp(value: object, *, index: int, field: str) -> datetime:
    milliseconds = _parse_integer(value, index=index, field=field)
    if milliseconds < 0:
        raise BinanceKlineFieldError(
            index=index, field=field, message="must be non-negative Unix milliseconds"
        )
    try:
        return _UNIX_EPOCH + timedelta(milliseconds=milliseconds)
    except OverflowError as exc:
        raise BinanceKlineFieldError(
            index=index, field=field, message="is outside the supported datetime range"
        ) from exc


def _parse_integer(value: object, *, index: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BinanceKlineFieldError(
            index=index, field=field, message="must be an integer"
        )
    return value


def _parse_decimal(value: object, *, index: int, field: str) -> Decimal:
    raw_value = _require_string(value, index=index, field=field)
    try:
        parsed = Decimal(raw_value)
    except InvalidOperation as exc:
        raise BinanceKlineFieldError(
            index=index, field=field, message="must be a decimal string"
        ) from exc
    if not parsed.is_finite():
        raise BinanceKlineFieldError(
            index=index, field=field, message="must be a finite decimal string"
        )
    return parsed


def _require_string(value: object, *, index: int, field: str) -> str:
    if not isinstance(value, str):
        raise BinanceKlineFieldError(
            index=index, field=field, message="must be a string"
        )
    return value
