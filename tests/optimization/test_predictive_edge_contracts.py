from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.jobs.analyze_predictive_edges import parse_horizon
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.predictive_edge_service import predictive_edge_experiment_key
from app.optimization.predictive_edges import PredictiveEdgeConfiguration
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
    assert len(selected) == 5
    assert all(set(methods) == {"get"} for methods in selected.values())

