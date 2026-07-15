from typing import cast

from sqlalchemy import CheckConstraint, Table, UniqueConstraint

from app.market.persistence import MarketCandleRecord, to_domain, to_record
from tests.market.factories import make_candle


def test_market_candle_metadata_has_named_natural_identity() -> None:
    table = cast(Table, MarketCandleRecord.__table__)
    unique = next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    )

    assert unique.name == "uq_market_candles_natural_identity"
    assert [column.name for column in unique.columns] == [
        "asset",
        "provider",
        "market_type",
        "interval",
        "open_time",
    ]
    assert not table.indexes


def test_market_candle_metadata_has_database_validation_constraints() -> None:
    table = cast(Table, MarketCandleRecord.__table__)
    names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert "ck_market_candles_time_order" in names
    assert "ck_market_candles_high_ge_open" in names
    assert "ck_market_candles_base_volume_nonnegative" in names
    assert "ck_market_candles_provider" in names


def test_domain_record_mapping_round_trip() -> None:
    candle = make_candle()

    restored = to_domain(to_record(candle))

    assert restored == candle
