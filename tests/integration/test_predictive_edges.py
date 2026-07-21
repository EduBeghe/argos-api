from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.predictive_edge_memory import PredictiveEdgeRunRecord
from app.optimization.predictive_edge_read import (
    get_predictive_edge_run,
    list_predictive_edge_windows,
    list_predictive_edges,
)
from app.optimization.predictive_edge_service import persist_predictive_edge_result
from app.optimization.predictive_edges import (
    Condition,
    EvidenceThresholds,
    PredictiveEdgeConclusion,
    PredictiveEdgeConfiguration,
    PredictiveEdgeResult,
    analyze_predictive_edges,
)
from app.optimization.walk_forward import WalkForwardConfiguration, WalkForwardWindow
from app.paper.config import SimulationConfiguration

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2031, 1, 1, tzinfo=UTC)
EXPERIMENT_KEY = "predictive-edge-integration-" + "a" * 36


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as value:
        await value.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == EXPERIMENT_KEY
            )
        )
        await value.commit()
        yield value
        await value.rollback()
        await value.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == EXPERIMENT_KEY
            )
        )
        await value.commit()
    await engine.dispose()


def _matrix() -> HistoricalFeatureMatrix:
    rows: list[HistoricalFeatureRow] = []
    price = Decimal("100")
    for index in range(144):
        at = START + timedelta(minutes=30 * index)
        price += Decimal("1") if index % 2 else Decimal("-0.4")
        rows.append(
            HistoricalFeatureRow(
                at,
                price,
                price + (Decimal("1") if index % 2 else Decimal("-1")),
                price,
                Decimal("60") if index % 2 else Decimal("40"),
                Decimal("2"),
                Decimal("0.01") if index % 2 else Decimal("-0.01"),
                Decimal("0.02") if index % 2 else Decimal("-0.02"),
                Decimal("1.2"),
            )
        )
    return HistoricalFeatureMatrix("b" * 64, tuple(rows))


def _window() -> WalkForwardWindow:
    return WalkForwardWindow(
        1,
        START,
        START + timedelta(hours=24),
        START + timedelta(hours=24),
        START + timedelta(hours=48),
        START + timedelta(hours=48),
        START + timedelta(hours=72),
        48,
        "c" * 64,
        "d" * 64,
    )


async def test_predictive_edge_round_trip_preserves_selection_semantics(
    session: AsyncSession,
) -> None:
    configuration = PredictiveEdgeConfiguration(
        horizons=(timedelta(minutes=30),),
        conditions=(Condition.ALL,),
        windows=WalkForwardConfiguration(
            training_days=1,
            validation_days=1,
            test_days=1,
            step_days=1,
            maximum_windows=1,
            maximum_candidates=1,
            minimum_valid_windows=1,
        ),
        thresholds=EvidenceThresholds(
            minimum_observations=10,
            minimum_windows=1,
            minimum_stable_direction_ratio=Decimal("0.5"),
            minimum_absolute_correlation=Decimal("0"),
            minimum_directional_hit_rate=Decimal("0"),
            minimum_hit_rate_improvement_over_benchmark=Decimal("0"),
        ),
    )
    result = analyze_predictive_edges(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
        configuration=configuration,
    )
    execution = await persist_predictive_edge_result(
        session=session,
        experiment_key=EXPERIMENT_KEY,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(hours=72),
        dataset_fingerprint="b" * 64,
        windows=(_window(),),
        result=result,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=START,
    )
    assert execution.run_id is not None

    detail = await get_predictive_edge_run(session, execution.run_id)
    edges = await list_predictive_edges(session, run_id=execution.run_id, limit=200)

    assert detail.run.algorithm_version == 3
    assert detail.run.windows_available == 1
    assert edges.items
    assert all(item.semantic_kind.value for item in edges.items)
    assert all(
        item.windows_confirmed <= item.windows_selected_for_test for item in edges.items
    )
    windows = await list_predictive_edge_windows(
        session,
        run_id=execution.run_id,
        edge_id=edges.items[0].id,
        limit=10,
    )
    assert windows.items
    assert windows.items[0].bin_selections
    assert all(
        item.suggested_action.value
        in {"long_entry", "exit_or_avoid", "diagnostic_only"}
        for item in windows.items[0].bin_selections
    )


async def test_insufficient_history_is_not_persisted_as_completed(
    session: AsyncSession,
) -> None:
    configuration = PredictiveEdgeConfiguration(
        horizons=(timedelta(minutes=30),),
        thresholds=EvidenceThresholds(minimum_windows=2),
    )
    result = PredictiveEdgeResult(
        PredictiveEdgeConclusion.INSUFFICIENT_HISTORY,
        0,
        2,
        "requires at least 2 valid walk-forward windows",
        configuration,
        (),
        (),
        (),
        (),
    )
    execution = await persist_predictive_edge_result(
        session=session,
        experiment_key=EXPERIMENT_KEY,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(days=1),
        dataset_fingerprint="e" * 64,
        windows=(),
        result=result,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=START,
    )
    record = await session.scalar(
        select(PredictiveEdgeRunRecord).where(
            PredictiveEdgeRunRecord.id == execution.run_id
        )
    )
    assert record is not None
    assert record.status == "insufficient_history"
    assert record.windows_available == 0
    assert record.windows_required == 2
