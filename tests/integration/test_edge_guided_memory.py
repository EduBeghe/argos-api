from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.edge_guided import (
    EdgeGuidedCandidate,
    EdgeGuidedCandidateEvaluation,
    EdgeGuidedCandidateStatus,
    EdgeGuidedEvaluationResult,
    EdgeGuidedSegmentEvaluation,
    EdgeGuidedWindowEvaluation,
)
from app.optimization.edge_guided_memory import (
    EdgeGuidedCandidateRecord,
    EdgeGuidedRunRecord,
    EdgeGuidedWindowRecord,
    find_reusable_edge_guided_run,
    persist_edge_guided_result,
)
from app.optimization.predictive_edge_memory import PredictiveEdgeRunRecord
from app.optimization.predictive_edge_service import (
    PredictiveEdgeExecution,
    persist_predictive_edge_result,
)
from app.optimization.predictive_edges import (
    Condition,
    DirectionMapping,
    EvidenceThresholds,
    PredictiveEdgeConclusion,
    PredictiveEdgeConfiguration,
    PredictiveEdgeResult,
    SuggestedAction,
)
from app.optimization.refinement import RefinementSegmentName
from app.paper.config import SimulationConfiguration

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2032, 1, 1, tzinfo=UTC)
PREDICTIVE_KEY = "edge-guided-parent-" + "a" * 45
EDGE_KEY = "edge-guided-memory-" + "b" * 45


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as value:
        await value.execute(
            delete(EdgeGuidedRunRecord).where(
                EdgeGuidedRunRecord.experiment_key == EDGE_KEY
            )
        )
        await value.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == PREDICTIVE_KEY
            )
        )
        await value.commit()
        yield value
        await value.rollback()
        await value.execute(
            delete(EdgeGuidedRunRecord).where(
                EdgeGuidedRunRecord.experiment_key == EDGE_KEY
            )
        )
        await value.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == PREDICTIVE_KEY
            )
        )
        await value.commit()
    await engine.dispose()


def metric(segment: RefinementSegmentName, value: str) -> EdgeGuidedSegmentEvaluation:
    decimal = Decimal(value)
    return EdgeGuidedSegmentEvaluation(
        segment,
        decimal,
        Decimal("0.012345678901234567"),
        3,
        Decimal("1.234567890123456789"),
        decimal,
        -decimal,
    )


def result() -> EdgeGuidedEvaluationResult:
    candidate = EdgeGuidedCandidate(
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "return_1",
        3,
        2,
        Condition.HIGH_VOLATILITY,
        Decimal("-0.020000000000000001"),
        Decimal("-0.010000000000000001"),
        SuggestedAction.LONG_ENTRY,
        timedelta(minutes=30),
        timedelta(minutes=30),
        DirectionMapping.NEGATIVE,
        "baseline_technical",
        1,
        "baseline_long_only",
        1,
        1,
        "USDT",
        Decimal("0.001"),
        Decimal("0.0005"),
    )
    training = metric(RefinementSegmentName.TRAINING, "0.011111111111111111")
    validation = metric(RefinementSegmentName.VALIDATION, "0.022222222222222222")
    test = metric(RefinementSegmentName.TEST, "0.033333333333333333")
    window = EdgeGuidedWindowEvaluation(
        1, Decimal("0.019876543210987654"), training, validation, test, True, True
    )
    evaluation = EdgeGuidedCandidateEvaluation(
        candidate,
        training,
        validation,
        test,
        (window,),
        1,
        1,
        0,
        0,
        "always_up",
        EdgeGuidedCandidateStatus.CONFIRMED,
        "confirmed",
    )
    return EdgeGuidedEvaluationResult("f" * 64, 1, 1, 12, 1, (evaluation,))


async def parent_run(session: AsyncSession) -> PredictiveEdgeExecution:
    configuration = PredictiveEdgeConfiguration(
        thresholds=EvidenceThresholds(minimum_windows=1)
    )
    empty = PredictiveEdgeResult(
        PredictiveEdgeConclusion.INSUFFICIENT_HISTORY,
        0,
        1,
        "fixture",
        configuration,
        (),
        (),
        (),
        (),
    )
    return await persist_predictive_edge_result(
        session=session,
        experiment_key=PREDICTIVE_KEY,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(days=1),
        dataset_fingerprint="f" * 64,
        windows=(),
        result=empty,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=START,
    )


async def test_complete_round_trip_and_reuse_are_atomic(session: AsyncSession) -> None:
    parent = await parent_run(session)
    run_id = parent.run_id
    assert run_id is not None
    original = result()
    first = await persist_edge_guided_result(
        session=session,
        experiment_key=EDGE_KEY,
        predictive_edge_run_id=run_id,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(days=1),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        decision_interval=timedelta(minutes=30),
        started_at=START,
        result=original,
    )
    reused = await find_reusable_edge_guided_run(session, EDGE_KEY)
    assert first.run_id is not None
    assert reused is not None
    assert reused.reused_existing_experiment is True
    assert reused.evaluations_computed == 0
    assert reused.evaluations_reused == 1
    assert reused.result == original
    assert reused.result.evaluations[0].test.total_return == Decimal(
        "0.033333333333333333"
    )
    assert (
        await session.scalar(select(func.count()).select_from(EdgeGuidedRunRecord)) == 1
    )
    assert (
        await session.scalar(
            select(func.count()).select_from(EdgeGuidedCandidateRecord)
        )
        == 1
    )
    assert (
        await session.scalar(select(func.count()).select_from(EdgeGuidedWindowRecord))
        == 1
    )
