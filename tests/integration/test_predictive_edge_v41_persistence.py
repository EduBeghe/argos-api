from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.predictive_edge_engine_v4 import analyze_predictive_edges_v4
from app.optimization.predictive_edge_engine_v41 import (
    PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
    ContextRegime,
    FrozenRegime,
    PhaseEvidence,
    PredictiveEdgeV41Result,
    RegimeConditionedRelationship,
    V41Candidate,
    V41Conclusion,
    V41PrimitiveCounters,
    V41StageCounters,
    _combine_phase_evidence,
)
from app.optimization.predictive_edge_memory import PredictiveEdgeRunRecord
from app.optimization.predictive_edge_read import (
    get_predictive_edge_diagnostics,
    list_predictive_edge_near_misses,
)
from app.optimization.predictive_edge_service import persist_predictive_edge_v4_result
from app.optimization.predictive_edges import DirectionMapping, SuggestedAction
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2032, 1, 1, tzinfo=UTC)
KEY = "v41-production-persistence-" + "1" * 39


@pytest_asyncio.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(get_settings().database_url)
    value = async_sessionmaker(engine, expire_on_commit=False)
    async with value() as session:
        await session.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == KEY
            )
        )
        await session.commit()
    yield value
    async with value() as session:
        await session.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == KEY
            )
        )
        await session.commit()
    await engine.dispose()


def _matrix() -> HistoricalFeatureMatrix:
    rows: list[HistoricalFeatureRow] = []
    price = Decimal("100")
    for index in range(420):
        signal = Decimal(1) if index % 2 else Decimal(-1)
        price *= Decimal(1) + signal * Decimal("0.001")
        rows.append(
            HistoricalFeatureRow(
                START + timedelta(minutes=5 * index),
                price,
                price,
                price,
                Decimal(50) + signal,
                Decimal(1),
                signal * Decimal("0.001"),
                signal * Decimal("0.002"),
                Decimal("1.2"),
            )
        )
    return HistoricalFeatureMatrix("v41-production-matrix", tuple(rows))


def _windows() -> tuple[WalkForwardWindow, ...]:
    result = []
    for ordinal, offset in enumerate((0, 120, 240), start=1):
        start = START + timedelta(minutes=5 * offset)
        result.append(
            WalkForwardWindow(
                ordinal,
                start,
                start + timedelta(hours=5),
                start + timedelta(hours=5),
                start + timedelta(hours=7, minutes=30),
                start + timedelta(hours=7, minutes=30),
                start + timedelta(hours=10),
                120,
                f"dataset-{ordinal}",
                f"selection-{ordinal}",
            )
        )
    return tuple(result)


def _window_evidence(*, stable: bool, economic: bool, passes: bool) -> PhaseEvidence:
    return PhaseEvidence(
        observations=12,
        gross_movement=Decimal("0.004"),
        aligned_gross_movement=Decimal("0.004"),
        net_movement=Decimal("0.001") if economic else Decimal("-0.0002"),
        directional_hit_rate=Decimal("0.60") if stable else Decimal("0.40"),
        benchmark=Decimal("0.52"),
        benchmark_improvement=Decimal("0.08") if passes else Decimal("-0.12"),
        standard_error=Decimal("0.0001"),
        stable=stable,
        economic=economic,
        passes=passes,
        usable_window_count=1,
        stable_window_count=int(stable),
        positive_aligned_window_count=1,
        confirmed_window_count=int(passes),
    )


def _v41_result() -> PredictiveEdgeV41Result:
    validation = _combine_phase_evidence(
        (
            _window_evidence(stable=True, economic=True, passes=True),
            _window_evidence(stable=True, economic=True, passes=True),
            _window_evidence(stable=True, economic=True, passes=True),
        )
    )
    test = _combine_phase_evidence(
        (
            _window_evidence(stable=True, economic=True, passes=True),
            _window_evidence(stable=True, economic=True, passes=True),
            _window_evidence(stable=False, economic=False, passes=False),
        )
    )
    regime = FrozenRegime(
        "atr_14_relative",
        1,
        ContextRegime.HIGH,
        Decimal("0.01"),
        None,
        120,
        "r" * 64,
    )
    relationship = RegimeConditionedRelationship(
        "return_12",
        1,
        "atr_14_relative",
        1,
        ContextRegime.HIGH,
        timedelta(minutes=30),
        timedelta(hours=4),
        regime,
        1,
        None,
        Decimal("-0.01"),
        DirectionMapping.POSITIVE,
        SuggestedAction.LONG_ENTRY,
        "balanced_v41",
        validation,
        validation,
        Decimal("0.0004"),
        "f" * 64,
    )
    candidate = V41Candidate(
        relationship,
        validation,
        test,
        True,
        False,
        V41Conclusion.ECONOMIC_INTERACTION_BUT_UNSTABLE.value,
    )
    return PredictiveEdgeV41Result(
        (),
        ("relative_volume_20",),
        (relationship,),
        (candidate,),
        (candidate,),
        ("relative_volume_20",),
        (),
        (),
        V41Conclusion.ECONOMIC_INTERACTION_BUT_UNSTABLE,
        V41StageCounters(16, 48, 0, 12),
        V41StageCounters(5, 15, 0, 5),
        V41StageCounters(192, 60, 0, 40),
        V41StageCounters(40, 40, 0, 1),
        V41StageCounters(1, 3, 0, 0),
        {},
        V41PrimitiveCounters(1, 100, 60, 40, 50, 3, 47),
    )


async def test_production_persistence_reconstructs_complete_v41_evidence(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    matrix = _matrix()
    windows = _windows()
    simulation = SimulationConfiguration()
    v4 = analyze_predictive_edges_v4(
        matrix=matrix, windows=windows, simulation=simulation
    )
    v41 = _v41_result()
    async with factory() as session:
        execution = await persist_predictive_edge_v4_result(
            session=session,
            experiment_key=KEY,
            cache_namespace="v4-integration",
            instrument=MARKET_INSTRUMENTS[0].instrument,
            range_start=START,
            range_end=START + timedelta(days=2),
            dataset_fingerprint="d" * 64,
            windows=windows,
            result=v4,
            fee_rate=simulation.fee_rate,
            slippage_rate=simulation.slippage_rate,
            started_at=START,
            regime_result=v41,
            regime_cache_namespace="v41-integration",
        )
        assert execution.run_id is not None
        run_id = execution.run_id

    async with factory() as restarted:
        diagnostics = await get_predictive_edge_diagnostics(restarted, run_id)
        actionable = await list_predictive_edge_near_misses(
            restarted, run_id=run_id, kind="actionable", limit=100
        )
        diagnostic = await list_predictive_edge_near_misses(
            restarted, run_id=run_id, kind="diagnostic", limit=100
        )

    item = next(value for value in actionable.items if value.engine == "v4.1")
    assert diagnostics.regime_conditioned_conclusion == v41.conclusion.value
    assert diagnostics.regime_conditioned_stage_counters == {
        "a1": {"generated": 16, "computed": 48, "reused": 0, "retained": 12},
        "a2": {"generated": 5, "computed": 15, "reused": 0, "retained": 5},
        "a3": {"generated": 192, "computed": 60, "reused": 0, "retained": 40},
        "validation": {"generated": 40, "computed": 40, "reused": 0, "retained": 1},
        "test": {"generated": 1, "computed": 3, "reused": 0, "retained": 0},
    }
    assert item.observations == 36
    assert item.usable_window_count == 3
    assert item.confirmed_window_count == 2
    assert item.passed_requirements
    assert item.failed_requirements
    assert len(item.requirement_margins) == 6
    assert item.normalized_missing_distance > 0
    assert item.category_rank == 1
    assert diagnostic.items[0].engine == "v4.1"
    assert diagnostic.items[0].category_rank == 1
    assert item.validation_evidence is not None
    assert item.validation_evidence.usable_window_count == 3
    assert item.test_evidence is not None
    assert item.test_evidence.confirmed_window_count == 2


def test_v41_contract_version_remains_explicit() -> None:
    assert PREDICTIVE_EDGE_V41_CONTRACT_VERSION == 42
