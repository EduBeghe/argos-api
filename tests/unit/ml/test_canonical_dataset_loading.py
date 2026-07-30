from datetime import timedelta
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml import loading
from app.ml.contract import DEVELOPMENT_START, FINAL_TEST_END, FINAL_TEST_START
from app.ml.dataset import build_ml_experiment_dataset
from app.ml.gaps import datetime_to_ns
from app.ml.reader import candle_columns

INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


@pytest.mark.asyncio
async def test_canonical_loader_uses_one_explicit_closed_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[tuple[object, object]] = []
    builds: list[object] = []
    columns = candle_columns(
        open_times=[],
        close_times=[],
        open_prices=[],
        high_prices=[],
        low_prices=[],
        close_prices=[],
        volumes=[],
    )

    async def reader(**kwargs: object) -> object:
        reads.append((kwargs["start"], kwargs["end"]))
        return SimpleNamespace(columns=columns)

    def build(**kwargs: object) -> object:
        builds.append(kwargs["requested_end"])
        return SimpleNamespace(
            boundary_time_ns=np.asarray([], dtype=np.int64),
            metadata=SimpleNamespace(
                fingerprint="a" * 64,
                requested_start=DEVELOPMENT_START,
                requested_end=kwargs["requested_end"],
                candle_count=0,
            ),
        )

    monkeypatch.setattr(loading, "read_ml_experiment_candles", reader)
    monkeypatch.setattr(loading, "build_ml_experiment_dataset", build)
    session = cast(AsyncSession, object())

    screening = await loading.load_canonical_ml_experiment_dataset(
        session=session, instrument=INSTRUMENT
    )
    refinement = await loading.load_canonical_ml_experiment_dataset(
        session=session, instrument=INSTRUMENT
    )

    assert reads == [
        (DEVELOPMENT_START, FINAL_TEST_END),
        (DEVELOPMENT_START, FINAL_TEST_END),
    ]
    assert builds == [FINAL_TEST_END, FINAL_TEST_END]
    assert screening.metadata.fingerprint == refinement.metadata.fingerprint
    assert screening.metadata.requested_end == refinement.metadata.requested_end


@pytest.mark.asyncio
async def test_development_loader_has_an_explicit_protected_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[tuple[object, object]] = []
    columns = candle_columns(
        open_times=[],
        close_times=[],
        open_prices=[],
        high_prices=[],
        low_prices=[],
        close_prices=[],
        volumes=[],
    )

    async def reader(**kwargs: object) -> object:
        reads.append((kwargs["start"], kwargs["end"]))
        return SimpleNamespace(columns=columns)

    monkeypatch.setattr(loading, "read_ml_experiment_candles", reader)
    dataset = await loading.load_ml_development_dataset(
        session=cast(AsyncSession, object()), instrument=INSTRUMENT
    )
    assert reads == [(DEVELOPMENT_START, FINAL_TEST_START)]
    assert dataset.metadata.requested_end == FINAL_TEST_START
    assert np.all(dataset.boundary_time_ns < datetime_to_ns(FINAL_TEST_START))
    assert len(dataset.final_test_indices) == 0
    assert (
        dataset.metadata.fingerprint
        != build_ml_experiment_dataset(
            instrument=INSTRUMENT, columns=columns, requested_end=FINAL_TEST_END
        ).metadata.fingerprint
    )


@pytest.mark.asyncio
async def test_development_loader_fails_typed_if_protected_boundary_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reader(**_kwargs: object) -> object:
        return SimpleNamespace(columns=object())

    def invalid_build(**_kwargs: object) -> object:
        return SimpleNamespace(
            boundary_time_ns=np.asarray(
                [datetime_to_ns(FINAL_TEST_START)], dtype=np.int64
            )
        )

    monkeypatch.setattr(loading, "read_ml_experiment_candles", reader)
    monkeypatch.setattr(loading, "build_ml_experiment_dataset", invalid_build)
    with pytest.raises(
        loading.DevelopmentDatasetProtectedEvidenceError,
        match=r"development_end_exclusive=2024-01-01T00:00:00\+00:00",
    ):
        await loading.load_ml_development_dataset(
            session=cast(AsyncSession, object()), instrument=INSTRUMENT
        )


def test_candles_after_the_exclusive_end_do_not_change_the_fingerprint() -> None:
    empty = candle_columns(
        open_times=[],
        close_times=[],
        open_prices=[],
        high_prices=[],
        low_prices=[],
        close_prices=[],
        volumes=[],
    )
    live_open = FINAL_TEST_END
    with_live_candle = candle_columns(
        open_times=[live_open],
        close_times=[live_open + timedelta(minutes=5) - timedelta(milliseconds=1)],
        open_prices=[100.0],
        high_prices=[101.0],
        low_prices=[99.0],
        close_prices=[100.5],
        volumes=[10.0],
    )
    original = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=empty, requested_end=FINAL_TEST_END
    )
    after_ingestion = build_ml_experiment_dataset(
        instrument=INSTRUMENT,
        columns=with_live_candle,
        requested_end=FINAL_TEST_END,
    )
    assert original.metadata.fingerprint == after_ingestion.metadata.fingerprint
    assert after_ingestion.metadata.candle_count == 0
