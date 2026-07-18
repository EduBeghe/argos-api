from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest

from app.market.catalog import (
    MARKET_INSTRUMENTS,
    MarketCatalogError,
    MarketInstrumentDefinition,
    binance_spot_five_minute_ingestion_instruments,
    enabled_decision_instruments,
    enabled_scoring_instruments,
    ingestion_enabled_instruments,
    market_instruments,
    provider_symbol,
    validate_catalog,
)
from app.market.enums import Asset, Interval, MarketType, Provider


def test_catalog_contains_current_instruments_in_stable_order() -> None:
    definitions = market_instruments()
    assert tuple(item.asset for item in definitions) == (
        Asset.BTC_USDT,
        Asset.ETH_USDT,
    )
    assert tuple(item.display_order for item in definitions) == (1, 2)
    assert binance_spot_five_minute_ingestion_instruments() == tuple(
        item.instrument for item in definitions
    )
    assert (
        enabled_scoring_instruments()
        == binance_spot_five_minute_ingestion_instruments()
    )
    assert (
        ingestion_enabled_instruments()
        == binance_spot_five_minute_ingestion_instruments()
    )
    assert (
        enabled_decision_instruments()
        == binance_spot_five_minute_ingestion_instruments()
    )
    assert provider_symbol(definitions[0].instrument) == "BTCUSDT"
    assert provider_symbol(definitions[1].instrument) == "ETHUSDT"


def test_catalog_and_definitions_are_immutable() -> None:
    assert isinstance(MARKET_INSTRUMENTS, tuple)
    with pytest.raises(FrozenInstanceError):
        MARKET_INSTRUMENTS[0].display_name = "changed"  # type: ignore[misc]


def test_capability_projections_exclude_disabled_instruments() -> None:
    display_only = replace(
        MARKET_INSTRUMENTS[1],
        ingestion_enabled=False,
        scoring_enabled=False,
        decision_enabled=False,
    )
    definitions = (MARKET_INSTRUMENTS[0], display_only)
    assert binance_spot_five_minute_ingestion_instruments(definitions) == (
        MARKET_INSTRUMENTS[0].instrument,
    )
    assert enabled_scoring_instruments(definitions) == (
        MARKET_INSTRUMENTS[0].instrument,
    )
    assert enabled_decision_instruments(definitions) == (
        MARKET_INSTRUMENTS[0].instrument,
    )
    assert ingestion_enabled_instruments(definitions) == (
        MARKET_INSTRUMENTS[0].instrument,
    )


def test_binance_ingestion_projection_rejects_empty_configuration() -> None:
    disabled = tuple(
        replace(
            item, ingestion_enabled=False, scoring_enabled=False, decision_enabled=False
        )
        for item in MARKET_INSTRUMENTS
    )
    with pytest.raises(MarketCatalogError, match="at least one"):
        binance_spot_five_minute_ingestion_instruments(disabled)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"provider": cast(Provider, "other")}, "provider"),
        ({"market_type": cast(MarketType, "other")}, "market type"),
        ({"interval": cast(Interval, "1h")}, "interval"),
    ],
)
def test_binance_ingestion_projection_rejects_incompatible_collection(
    change: dict[str, object], message: str
) -> None:
    incompatible = replace(MARKET_INSTRUMENTS[1], **change)  # type: ignore[arg-type]
    with pytest.raises(MarketCatalogError, match=message):
        binance_spot_five_minute_ingestion_instruments(
            (MARKET_INSTRUMENTS[0], incompatible)
        )


def test_binance_ingestion_projection_rejects_duplicate_identity() -> None:
    duplicate = replace(MARKET_INSTRUMENTS[0], display_order=99)
    with pytest.raises(MarketCatalogError, match="duplicate"):
        binance_spot_five_minute_ingestion_instruments(
            (MARKET_INSTRUMENTS[0], duplicate)
        )


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        (
            replace(
                MARKET_INSTRUMENTS[0],
                decision_enabled=True,
                scoring_enabled=False,
            ),
            "enable scoring",
        ),
        (
            replace(
                MARKET_INSTRUMENTS[0],
                decision_enabled=False,
                scoring_enabled=True,
                ingestion_enabled=False,
            ),
            "enable ingestion",
        ),
        (replace(MARKET_INSTRUMENTS[0], display_symbol=" "), "display_symbol"),
        (replace(MARKET_INSTRUMENTS[0], display_name=""), "display_name"),
        (
            replace(MARKET_INSTRUMENTS[0], base_currency="USDT"),
            "currencies must differ",
        ),
    ],
)
def test_invalid_capabilities_and_fields_fail(
    definition: MarketInstrumentDefinition, message: str
) -> None:
    with pytest.raises(MarketCatalogError, match=message):
        validate_catalog((definition,))


def test_duplicate_complete_identity_fails() -> None:
    duplicate = replace(MARKET_INSTRUMENTS[0], display_order=99)
    with pytest.raises(MarketCatalogError, match="complete instrument identity"):
        validate_catalog((MARKET_INSTRUMENTS[0], duplicate))


def test_duplicate_provider_symbol_fails() -> None:
    duplicate = replace(
        MARKET_INSTRUMENTS[1],
        provider_symbol=MARKET_INSTRUMENTS[0].provider_symbol,
    )
    with pytest.raises(MarketCatalogError, match="provider symbol"):
        validate_catalog((MARKET_INSTRUMENTS[0], duplicate))


def test_duplicate_display_order_fails() -> None:
    duplicate = replace(
        MARKET_INSTRUMENTS[1],
        display_order=MARKET_INSTRUMENTS[0].display_order,
    )
    with pytest.raises(MarketCatalogError, match="display order"):
        validate_catalog((MARKET_INSTRUMENTS[0], duplicate))
