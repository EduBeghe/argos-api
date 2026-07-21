import argparse
from decimal import Decimal

import pytest

from app.jobs.refine_historical_strategy import build_parser, positive_integer
from app.market.enums import Asset, Interval, MarketType, Provider


def test_refinement_command_parses_identity_costs_and_top_k() -> None:
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
            "--validation-top-k",
            "7",
            "--score-screening-top-k",
            "9",
            "--max-score-configurations",
            "190",
            "--max-joint-candidates",
            "900",
            "--persist",
            "--force-recompute",
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
    assert args.validation_top_k == 7
    assert args.score_screening_top_k == 9
    assert args.max_score_configurations == 190
    assert args.max_joint_candidates == 900
    assert args.persist is True
    assert args.force_recompute is True


def test_refinement_memory_flags_default_to_read_only() -> None:
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
            "--initial-cash",
            "10000",
            "--fee-rate",
            "0.001",
            "--slippage-rate",
            "0.0005",
        ]
    )
    assert args.persist is False
    assert args.force_recompute is False
    assert args.walk_forward is False


def test_refinement_command_rejects_non_positive_top_k() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        positive_integer("0")


def test_refinement_command_parses_walk_forward_configuration() -> None:
    args = build_parser().parse_args(
        [
            "--from",
            "2024-01-01T00:00:00Z",
            "--to",
            "2026-01-01T00:00:00Z",
            "--asset",
            "BTC/USDT",
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
            "--persist",
            "--walk-forward",
            "--walk-forward-training-days",
            "180",
            "--walk-forward-validation-days",
            "60",
            "--walk-forward-test-days",
            "60",
            "--walk-forward-step-days",
            "60",
            "--walk-forward-max-windows",
            "6",
            "--walk-forward-max-candidates",
            "10",
        ]
    )

    assert args.walk_forward is True
    assert args.walk_forward_training_days == 180
    assert args.walk_forward_validation_days == 60
    assert args.walk_forward_test_days == 60
    assert args.walk_forward_step_days == 60
    assert args.walk_forward_max_windows == 6
    assert args.walk_forward_max_candidates == 10

