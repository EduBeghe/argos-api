import argparse
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.optimization.weekly_research as research_module
from app.jobs.run_weekly_strategy_research import run as run_job
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.edge_guided import (
    EdgeGuidedEvaluationResult,
    ResearchOutcome,
)
from app.optimization.edge_guided_memory import EdgeGuidedExecution
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    HistoricalDatasetRevision,
)
from app.optimization.predictive_edges import PredictiveEdgeConfiguration
from app.optimization.weekly_research import (
    execute_weekly_instrument_research,
    weekly_boundary,
    weekly_research_reuse_key,
)
from app.paper.config import SimulationConfiguration

INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)
IDENTITY = HistoricalDatasetIdentity(INSTRUMENT, HistoricalDatasetRange(START, END))
REVISION = HistoricalDatasetRevision(
    100,
    START,
    END - timedelta(minutes=5),
    END,
    "revision",
)


def test_weekly_boundary_is_deterministic_across_hour_and_day_rollover() -> None:
    expected = datetime(2026, 1, 4, 3, 17, tzinfo=UTC)
    assert weekly_boundary(datetime(2026, 1, 4, 3, 17, tzinfo=UTC)) == expected
    assert weekly_boundary(datetime(2026, 1, 10, 23, 59, tzinfo=UTC)) == expected
    assert weekly_boundary(datetime(2026, 1, 4, 3, 16, tzinfo=UTC)) == datetime(
        2025, 12, 28, 3, 17, tzinfo=UTC
    )


def test_reuse_identity_is_deterministic_and_includes_effective_costs() -> None:
    def key(fee_rate: str) -> str:
        return weekly_research_reuse_key(
            identity=IDENTITY,
            revision=REVISION,
            configuration=PredictiveEdgeConfiguration(),
            simulation=SimulationConfiguration(fee_rate=Decimal(fee_rate)),
            initial_cash=Decimal("10000"),
            maximum_candidates=12,
            boundary=datetime(2026, 1, 4, 3, 17, tzinfo=UTC),
        )

    first = key("0.001")
    assert first == key("0.001")
    assert first != key("0.002")


@pytest.mark.asyncio
async def test_exact_reuse_short_circuits_every_heavy_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = EdgeGuidedEvaluationResult(
        "dataset",
        1,
        1,
        12,
        3,
        (),
        ResearchOutcome.CANDIDATES_REJECTED_VALIDATION,
        "reused",
        {"edges_evaluated": 19, "unstable_edges": 19},
    )
    execution = EdgeGuidedExecution(None, "key", True, result, 1, 0, 22)
    revision = AsyncMock(return_value=REVISION)
    reusable = AsyncMock(return_value=execution)
    dataset_loader = AsyncMock(side_effect=AssertionError("dataset loaded"))
    matrix_builder = Mock(side_effect=AssertionError("matrix built"))
    edge_analyzer = Mock(side_effect=AssertionError("edge analysis executed"))
    candidate_evaluator = Mock(side_effect=AssertionError("candidate simulated"))
    monkeypatch.setattr(research_module, "read_historical_dataset_revision", revision)
    monkeypatch.setattr(research_module, "find_reusable_weekly_research_run", reusable)
    monkeypatch.setattr(research_module, "read_historical_dataset", dataset_loader)
    monkeypatch.setattr(
        research_module, "build_historical_feature_matrix", matrix_builder
    )
    monkeypatch.setattr(research_module, "analyze_predictive_edges", edge_analyzer)
    monkeypatch.setattr(
        research_module, "evaluate_edge_guided_candidates", candidate_evaluator
    )

    completed = await execute_weekly_instrument_research(
        session=SimpleNamespace(),  # type: ignore[arg-type]
        identity=IDENTITY,
        configuration=PredictiveEdgeConfiguration(),
        simulation=SimulationConfiguration(),
        initial_cash=Decimal("10000"),
        maximum_candidates=12,
        boundary=datetime(2026, 1, 4, 3, 17, tzinfo=UTC),
        persist=True,
    )

    assert completed.reused is True
    assert completed.execution.result.funnel["unstable_edges"] == 19
    assert completed.work.evaluations_computed == 0
    assert completed.work.evaluations_reused == 1
    assert completed.work.simulations_avoided == 22
    dataset_loader.assert_not_awaited()
    matrix_builder.assert_not_called()
    edge_analyzer.assert_not_called()
    candidate_evaluator.assert_not_called()


@pytest.mark.asyncio
async def test_weekly_dry_run_is_catalog_driven_and_does_not_start_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyze = AsyncMock(side_effect=AssertionError("analysis started"))
    monkeypatch.setattr(
        "app.jobs.run_weekly_strategy_research.execute_weekly_instrument_research",
        analyze,
    )
    args = argparse.Namespace(
        asset=None,
        lookback_days=730,
        maximum_candidates=12,
        force_recompute=False,
        dry_run=True,
    )

    summary = await run_job(args, now=datetime(2026, 1, 4, 3, 17, tzinfo=UTC))

    assert summary["status"] == "validated"
    assert summary["instruments"] == [
        ["BTC/USDT", "binance", "spot", "5m"],
        ["ETH/USDT", "binance", "spot", "5m"],
    ]
    analyze.assert_not_awaited()
