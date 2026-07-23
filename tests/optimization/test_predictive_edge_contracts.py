from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.jobs.analyze_predictive_edges import build_parser, parse_horizon, run
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.predictive_edge_service import predictive_edge_experiment_key
from app.optimization.predictive_edges import (
    PredictiveEdgeConclusion,
    PredictiveEdgeConfiguration,
    PredictiveEdgeResult,
)
from main import create_app


def test_horizon_parser_is_explicit() -> None:
    assert parse_horizon("30m") == timedelta(minutes=30)
    assert parse_horizon("1h") == timedelta(hours=1)
    assert parse_horizon("2d") == timedelta(days=2)
    with pytest.raises(Exception, match="horizon"):
        parse_horizon("30")


def test_experiment_identity_is_deterministic_and_material_changes_invalidate_it() -> (
    None
):
    instrument = MARKET_INSTRUMENTS[0].instrument
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    configuration = PredictiveEdgeConfiguration(horizons=(timedelta(minutes=30),))
    first = predictive_edge_experiment_key(
        dataset_fingerprint="a" * 64,
        instrument=instrument,
        range_start=start,
        range_end=end,
        configuration=configuration,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    )
    assert (
        predictive_edge_experiment_key(
            dataset_fingerprint="a" * 64,
            instrument=instrument,
            range_start=start,
            range_end=end,
            configuration=configuration,
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
        )
        == first
    )
    assert (
        predictive_edge_experiment_key(
            dataset_fingerprint="a" * 64,
            instrument=instrument,
            range_start=start,
            range_end=end,
            configuration=configuration,
            fee_rate=Decimal("0.002"),
            slippage_rate=Decimal("0.0005"),
        )
        != first
    )


def test_openapi_exposes_only_read_only_predictive_edge_routes() -> None:
    client = TestClient(create_app())
    paths = client.get("/openapi.json").json()["paths"]
    selected = {
        path: methods
        for path, methods in paths.items()
        if path.startswith("/optimization/predictive-edge-runs")
    }
    assert len(selected) == 11
    assert all(set(methods) == {"get"} for methods in selected.values())


@pytest.mark.asyncio
async def test_non_persisted_job_never_queries_reuse_or_mutates_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(new=set(), dirty=set(), deleted=set())

    class SessionContext:
        async def __aenter__(self) -> object:
            return session

        async def __aexit__(self, *args: object) -> None:
            return None

    engine = SimpleNamespace(dispose=AsyncMock())
    factory = Mock(return_value=SessionContext())
    dataset = SimpleNamespace(fingerprint="f" * 64)
    matrix = SimpleNamespace(dataset_fingerprint="f" * 64, rows=())
    configuration_result = PredictiveEdgeResult(
        PredictiveEdgeConclusion.INSUFFICIENT_HISTORY,
        0,
        3,
        "insufficient",
        PredictiveEdgeConfiguration(horizons=(timedelta(minutes=30),)),
        (),
        (),
        (),
        (),
    )
    reuse = AsyncMock()
    persist = AsyncMock()
    monkeypatch.setattr(
        "app.jobs.analyze_predictive_edges.create_database_engine",
        lambda settings: engine,
    )
    monkeypatch.setattr(
        "app.jobs.analyze_predictive_edges.create_session_factory",
        lambda value: factory,
    )
    monkeypatch.setattr(
        "app.jobs.analyze_predictive_edges.read_historical_dataset",
        AsyncMock(return_value=dataset),
    )
    monkeypatch.setattr(
        "app.jobs.analyze_predictive_edges.build_historical_feature_matrix",
        lambda value: matrix,
    )
    monkeypatch.setattr(
        "app.jobs.analyze_predictive_edges.build_predictive_windows",
        lambda **kwargs: (),
    )
    monkeypatch.setattr(
        "app.jobs.analyze_predictive_edges.analyze_predictive_edges",
        lambda **kwargs: configuration_result,
    )
    monkeypatch.setattr(
        "app.jobs.analyze_predictive_edges.find_reusable_predictive_edge_run", reuse
    )
    monkeypatch.setattr(
        "app.jobs.analyze_predictive_edges.persist_predictive_edge_result", persist
    )
    args = build_parser().parse_args(
        [
            "--asset",
            "BTC/USDT",
            "--provider",
            "binance",
            "--market-type",
            "spot",
            "--interval",
            "5m",
            "--from",
            "2024-01-01T00:00:00Z",
            "--to",
            "2026-01-01T00:00:00Z",
            "--horizon",
            "30m",
        ]
    )

    execution = await run(args, now=datetime(2026, 1, 2, tzinfo=UTC))

    assert execution.result is configuration_result
    reuse.assert_not_awaited()
    persist.assert_not_awaited()
    assert session.new == session.dirty == session.deleted == set()
