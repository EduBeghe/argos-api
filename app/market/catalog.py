from dataclasses import dataclass
from enum import StrEnum

from app.market.enums import Asset, Interval, MarketType, Provider


class MarketCatalogError(ValueError):
    """Raised when the configured market instrument catalog is invalid."""


class InstrumentCapability(StrEnum):
    INGESTION = "ingestion"
    SCORING = "scoring"
    DECISION = "decision"


@dataclass(frozen=True, slots=True)
class Instrument:
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval


@dataclass(frozen=True, slots=True)
class MarketInstrumentDefinition:
    asset: Asset
    display_symbol: str
    display_name: str
    asset_class: str
    base_currency: str
    quote_currency: str
    provider: Provider
    provider_symbol: str
    market_type: MarketType
    interval: Interval
    ingestion_enabled: bool
    scoring_enabled: bool
    decision_enabled: bool
    display_order: int

    @property
    def instrument(self) -> Instrument:
        return Instrument(self.asset, self.provider, self.market_type, self.interval)


MARKET_INSTRUMENTS = (
    MarketInstrumentDefinition(
        asset=Asset.BTC_USDT,
        display_symbol="BTC/USDT",
        display_name="Bitcoin / Tether",
        asset_class="crypto",
        base_currency="BTC",
        quote_currency="USDT",
        provider=Provider.BINANCE,
        provider_symbol="BTCUSDT",
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        ingestion_enabled=True,
        scoring_enabled=True,
        decision_enabled=True,
        display_order=1,
    ),
    MarketInstrumentDefinition(
        asset=Asset.ETH_USDT,
        display_symbol="ETH/USDT",
        display_name="Ethereum / Tether",
        asset_class="crypto",
        base_currency="ETH",
        quote_currency="USDT",
        provider=Provider.BINANCE,
        provider_symbol="ETHUSDT",
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        ingestion_enabled=True,
        scoring_enabled=True,
        decision_enabled=True,
        display_order=2,
    ),
)


def validate_catalog(
    definitions: tuple[MarketInstrumentDefinition, ...],
) -> None:
    identities: set[Instrument] = set()
    provider_symbols: set[tuple[Provider, MarketType, str]] = set()
    display_orders: set[int] = set()
    for definition in definitions:
        if definition.instrument in identities:
            raise MarketCatalogError("duplicate complete instrument identity")
        identities.add(definition.instrument)
        symbol_identity = (
            definition.provider,
            definition.market_type,
            definition.provider_symbol,
        )
        if symbol_identity in provider_symbols:
            raise MarketCatalogError(
                "duplicate provider symbol within provider and market type"
            )
        provider_symbols.add(symbol_identity)
        if definition.display_order in display_orders:
            raise MarketCatalogError("duplicate instrument display order")
        display_orders.add(definition.display_order)
        for name, value in (
            ("display_symbol", definition.display_symbol),
            ("display_name", definition.display_name),
            ("asset_class", definition.asset_class),
            ("base_currency", definition.base_currency),
            ("quote_currency", definition.quote_currency),
            ("provider_symbol", definition.provider_symbol),
        ):
            if not value.strip():
                raise MarketCatalogError(f"{name} must not be empty")
        if definition.base_currency == definition.quote_currency:
            raise MarketCatalogError("base and quote currencies must differ")
        if definition.decision_enabled and not definition.scoring_enabled:
            raise MarketCatalogError("decision-enabled instruments must enable scoring")
        if definition.scoring_enabled and not definition.ingestion_enabled:
            raise MarketCatalogError(
                "scoring-enabled instruments must enable ingestion"
            )
        if definition.decision_enabled and (
            definition.provider is not Provider.BINANCE
            or definition.market_type is not MarketType.SPOT
            or definition.interval is not Interval.FIVE_MINUTES
        ):
            raise MarketCatalogError(
                "decision-enabled instrument configuration is unsupported"
            )


def market_instruments(
    definitions: tuple[MarketInstrumentDefinition, ...] = MARKET_INSTRUMENTS,
) -> tuple[MarketInstrumentDefinition, ...]:
    return tuple(sorted(definitions, key=lambda item: item.display_order))


def binance_spot_five_minute_ingestion_instruments(
    definitions: tuple[MarketInstrumentDefinition, ...] = MARKET_INSTRUMENTS,
) -> tuple[Instrument, ...]:
    enabled = tuple(
        item for item in market_instruments(definitions) if item.ingestion_enabled
    )
    if not enabled:
        raise MarketCatalogError(
            "Binance Spot 5m ingestion requires at least one enabled instrument"
        )
    identities: set[Instrument] = set()
    for definition in enabled:
        if definition.provider is not Provider.BINANCE:
            raise MarketCatalogError(
                "Binance Spot 5m ingestion received an incompatible provider"
            )
        if definition.market_type is not MarketType.SPOT:
            raise MarketCatalogError(
                "Binance Spot 5m ingestion received an incompatible market type"
            )
        if definition.interval is not Interval.FIVE_MINUTES:
            raise MarketCatalogError(
                "Binance Spot 5m ingestion received an incompatible interval"
            )
        if definition.instrument in identities:
            raise MarketCatalogError(
                "Binance Spot 5m ingestion contains a duplicate instrument identity"
            )
        identities.add(definition.instrument)
    return tuple(item.instrument for item in enabled)


def enabled_scoring_instruments(
    definitions: tuple[MarketInstrumentDefinition, ...] = MARKET_INSTRUMENTS,
) -> tuple[Instrument, ...]:
    return tuple(
        item.instrument
        for item in market_instruments(definitions)
        if item.scoring_enabled
    )


def enabled_decision_instruments(
    definitions: tuple[MarketInstrumentDefinition, ...] = MARKET_INSTRUMENTS,
) -> tuple[Instrument, ...]:
    return tuple(
        item.instrument
        for item in market_instruments(definitions)
        if item.decision_enabled
    )


def provider_symbol(instrument: Instrument) -> str:
    for definition in MARKET_INSTRUMENTS:
        if definition.instrument == instrument:
            return definition.provider_symbol
    raise MarketCatalogError("instrument is not configured in the market catalog")


def resolve_instrument(
    *,
    asset: Asset,
    interval: Interval,
    capability: InstrumentCapability,
    definitions: tuple[MarketInstrumentDefinition, ...] = MARKET_INSTRUMENTS,
) -> MarketInstrumentDefinition:
    for definition in definitions:
        if definition.asset is not asset or definition.interval is not interval:
            continue
        enabled = {
            InstrumentCapability.INGESTION: definition.ingestion_enabled,
            InstrumentCapability.SCORING: definition.scoring_enabled,
            InstrumentCapability.DECISION: definition.decision_enabled,
        }[capability]
        if enabled:
            return definition
    raise MarketCatalogError(
        f"instrument is not configured for {capability.value}: "
        f"{asset.value} {interval.value}"
    )


validate_catalog(MARKET_INSTRUMENTS)
