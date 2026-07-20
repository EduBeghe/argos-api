import argparse
from datetime import UTC, datetime

import pytest

from app.jobs.validate_historical_dataset import build_parser, parse_utc
from app.market.enums import Asset, Interval, MarketType, Provider


def test_validation_command_parses_complete_catalog_identity() -> None:
    args = build_parser().parse_args(
        [
            "--from",
            "2024-01-01T00:00:00Z",
            "--to",
            "2025-01-01T00:00:00Z",
            "--asset",
            "BTC/USDT",
            "--provider",
            "binance",
            "--market-type",
            "spot",
            "--interval",
            "5m",
        ]
    )
    assert (args.asset, args.provider, args.market_type, args.interval) == (
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )


def test_validation_command_rejects_non_utc_timestamp() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="UTC"):
        parse_utc("2024-01-01T00:00:00")
    assert parse_utc("2024-01-01T00:00:00Z") == datetime(2024, 1, 1, tzinfo=UTC)
