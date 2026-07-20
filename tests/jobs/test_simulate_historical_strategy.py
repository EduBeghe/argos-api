import argparse
from decimal import Decimal

import pytest

from app.jobs.simulate_historical_strategy import build_parser, parse_decimal
from app.market.enums import Asset, Interval, MarketType, Provider


def test_simulation_command_parses_identity_and_decimal_configuration() -> None:
    args = build_parser().parse_args(
        [
            "--from",
            "2024-01-01T00:00:00Z",
            "--to",
            "2024-04-01T00:00:00Z",
            "--asset",
            "ETH/USDT",
            "--provider",
            "binance",
            "--market-type",
            "spot",
            "--interval",
            "5m",
            "--initial-cash",
            "10000",
            "--fee-rate",
            "0.001",
            "--slippage-rate",
            "0.0005",
        ]
    )
    assert (args.asset, args.provider, args.market_type, args.interval) == (
        Asset.ETH_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    assert (args.initial_cash, args.fee_rate, args.slippage_rate) == (
        Decimal("10000"),
        Decimal("0.001"),
        Decimal("0.0005"),
    )


def test_simulation_command_rejects_non_finite_decimal() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        parse_decimal("NaN")
