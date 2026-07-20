import argparse
from datetime import UTC, datetime

import pytest

from app.jobs.backfill_market_candles import build_parser, parse_utc, select_instruments
from app.market.enums import Asset, Interval, MarketType, Provider


def test_parse_utc_accepts_z_and_rejects_naive_or_non_utc() -> None:
    assert parse_utc("2024-01-01T00:00:00Z") == datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(argparse.ArgumentTypeError, match="UTC"):
        parse_utc("2024-01-01T00:00:00")
    with pytest.raises(argparse.ArgumentTypeError, match="UTC"):
        parse_utc("2024-01-01T01:00:00+01:00")


def test_parser_requires_bounds_and_parses_control_options() -> None:
    args = build_parser().parse_args(
        [
            "--from",
            "2024-01-01T00:00:00Z",
            "--to",
            "2024-04-01T00:00:00Z",
            "--asset",
            "BTC/USDT",
            "--provider",
            "binance",
            "--market-type",
            "spot",
            "--interval",
            "5m",
            "--page-size",
            "999",
            "--max-pages",
            "5",
            "--dry-run",
        ]
    )
    assert args.asset is Asset.BTC_USDT
    assert args.provider is Provider.BINANCE
    assert args.market_type is MarketType.SPOT
    assert args.interval is Interval.FIVE_MINUTES
    assert (args.page_size, args.max_pages, args.dry_run) == (999, 5, True)


def test_catalog_selection_is_data_driven_and_filterable() -> None:
    args = argparse.Namespace(
        asset=None, provider=None, market_type=None, interval=None
    )
    assert [item.asset for item in select_instruments(args)] == [
        Asset.BTC_USDT,
        Asset.ETH_USDT,
    ]
    args.asset = Asset.ETH_USDT
    assert [item.asset for item in select_instruments(args)] == [Asset.ETH_USDT]
