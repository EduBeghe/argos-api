import argparse
from datetime import UTC, datetime

import pytest

from app.jobs.validate_historical_dataset import (
    build_parser,
    parse_utc,
    resolve_historical_instrument,
)
from app.market.catalog import MarketCatalogError, MarketInstrumentDefinition
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


def definition(asset: Asset, *, scoring_enabled: bool) -> MarketInstrumentDefinition:
    symbol = asset.value.replace("/", "")
    return MarketInstrumentDefinition(
        asset=asset,
        display_symbol=asset.value,
        display_name=asset.value,
        asset_class="crypto",
        base_currency=asset.value.split("/")[0],
        quote_currency="USDT",
        provider=Provider.BINANCE,
        provider_symbol=symbol,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        ingestion_enabled=True,
        scoring_enabled=scoring_enabled,
        decision_enabled=False,
        display_order=1,
    )


def test_historical_command_requires_scoring_capability_without_asset_branches() -> (
    None
):
    scoring_asset = Asset.ETH_USDT
    ingestion_only_asset = Asset.BTC_USDT
    definitions = (
        definition(scoring_asset, scoring_enabled=True),
        definition(ingestion_only_asset, scoring_enabled=False),
    )

    selected = resolve_historical_instrument(
        asset=scoring_asset,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        definitions=definitions,
    )
    assert selected.asset is scoring_asset

    with pytest.raises(MarketCatalogError, match="not configured for scoring"):
        resolve_historical_instrument(
            asset=ingestion_only_asset,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            definitions=definitions,
        )
