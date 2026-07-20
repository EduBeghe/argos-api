import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.features.persistence import MarketFeatureRecord
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from app.optimization.experiment_service import (
    ABANDONED_EXPERIMENT_TIMEOUT,
    ExperimentExecution,
    ExperimentMemoryError,
    execute_refinement,
)
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import (
    HistoricalDataset,
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    read_historical_dataset,
)
from app.optimization.memory import (
    StrategyCandidateEvaluationRecord,
    StrategyRefinementRunRecord,
    reclaim_abandoned_run,
)
from app.optimization.refinement import RefinementBudget, refine_historical_dataset
from app.optimization.simulation import simulate_historical_strategy
from app.paper.persistence import PaperPositionRecord, PaperTradeRecord
from app.scores.persistence import MarketScoreRecord
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2024, 3, 1, tzinfo=UTC)
COUNT = 120
END = START + timedelta(minutes=5 * COUNT)
INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


async def execute_memory_refinement(
    session: AsyncSession,
    dataset: HistoricalDataset,
    started_at: datetime,
) -> ExperimentExecution:
    return await execute_refinement(
        session=session,
        dataset=dataset,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        budget=RefinementBudget(1, 1, 200, 60),
        persist=True,
        force_recompute=False,
        started_at=started_at,
    )


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as value:
        cleanup = delete(MarketCandleRecord).where(
            MarketCandleRecord.open_time >= START, MarketCandleRecord.open_time < END
        )
        await value.execute(cleanup)
        await value.commit()
        for index in range(COUNT):
            opened = START + timedelta(minutes=5 * index)
            value.add(
                to_record(
                    make_candle(
                        asset=Asset.BTC_USDT,
                        open_time=opened,
                        close_time=opened + timedelta(minutes=5, milliseconds=-1),
                        open_price=Decimal(index + 2),
                        high_price=Decimal(index + 3),
                        low_price=Decimal(index + 1),
                        close_price=Decimal(index + 2),
                        quote_asset_volume=Decimal(100 + index),
                    )
                )
            )
        await value.commit()
        yield value
        await value.rollback()
        await value.execute(cleanup)
        await value.commit()
    await engine.dispose()


async def test_keyset_multi_page_matrix_is_read_only(session: AsyncSession) -> None:
    feature_count_before = await session.scalar(
        select(func.count()).select_from(MarketFeatureRecord)
    )
    score_count_before = await session.scalar(
        select(func.count()).select_from(MarketScoreRecord)
    )
    dataset = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    matrix = build_historical_feature_matrix(dataset)
    assert dataset.actual_candle_count == COUNT
    assert len(matrix.rows) == COUNT - 49
    assert (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord))
        == feature_count_before
    )
    assert (
        await session.scalar(select(func.count()).select_from(MarketScoreRecord))
        == score_count_before
    )
    assert not session.new and not session.dirty and not session.deleted


async def test_historical_simulation_is_read_only(session: AsyncSession) -> None:
    counts_before = (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord)),
        await session.scalar(select(func.count()).select_from(MarketScoreRecord)),
        await session.scalar(select(func.count()).select_from(PaperTradeRecord)),
        await session.scalar(select(func.count()).select_from(PaperPositionRecord)),
    )
    dataset = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    result = simulate_historical_strategy(
        dataset=dataset,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    )
    counts_after = (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord)),
        await session.scalar(select(func.count()).select_from(MarketScoreRecord)),
        await session.scalar(select(func.count()).select_from(PaperTradeRecord)),
        await session.scalar(select(func.count()).select_from(PaperPositionRecord)),
    )
    assert result.decision_count > 0
    assert counts_after == counts_before
    assert not session.new and not session.dirty and not session.deleted


async def test_historical_refinement_is_read_only(session: AsyncSession) -> None:
    counts_before = (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord)),
        await session.scalar(select(func.count()).select_from(MarketScoreRecord)),
        await session.scalar(select(func.count()).select_from(PaperTradeRecord)),
        await session.scalar(select(func.count()).select_from(PaperPositionRecord)),
    )
    dataset = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    result = refine_historical_dataset(
        dataset=dataset,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        budget=RefinementBudget(1, 1, 200, 60),
    )
    counts_after = (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord)),
        await session.scalar(select(func.count()).select_from(MarketScoreRecord)),
        await session.scalar(select(func.count()).select_from(PaperTradeRecord)),
        await session.scalar(select(func.count()).select_from(PaperPositionRecord)),
    )
    assert result.counts.score_configurations_generated == 176
    assert result.counts.joint_candidates_evaluated == 60
    assert counts_after == counts_before
    assert not session.new and not session.dirty and not session.deleted


async def test_persisted_refinement_is_atomic_and_exact_retry_is_reused(
    session: AsyncSession,
) -> None:
    await session.execute(
        delete(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.dataset_fingerprint
            == "integration-refinement-memory"
        )
    )
    await session.commit()
    source = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    dataset = replace(source, fingerprint="integration-refinement-memory")
    first = await execute_refinement(
        session=session,
        dataset=dataset,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        budget=RefinementBudget(1, 1, 200, 60),
        persist=True,
        force_recompute=False,
        started_at=END + timedelta(days=2),
    )
    assert first.result is not None
    assert first.result.persisted
    assert first.reused is None
    run = await session.scalar(
        select(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.experiment_key == first.result.experiment_key
        )
    )
    assert run is not None and run.status == "completed"
    evaluation_count = await session.scalar(
        select(func.count())
        .select_from(StrategyCandidateEvaluationRecord)
        .where(StrategyCandidateEvaluationRecord.refinement_run_id == run.id)
    )
    assert evaluation_count and evaluation_count > 0

    second = await execute_refinement(
        session=session,
        dataset=dataset,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        budget=RefinementBudget(1, 1, 200, 60),
        persist=True,
        force_recompute=False,
        started_at=END + timedelta(days=3),
    )
    assert second.result is None
    assert second.reused is not None
    assert second.reused.reused_existing_experiment
    assert second.reused.evaluations_computed == 0
    assert (
        await session.scalar(
            select(func.count())
            .select_from(StrategyRefinementRunRecord)
            .where(
                StrategyRefinementRunRecord.experiment_key
                == first.result.experiment_key
            )
        )
        == 1
    )


async def test_failed_refinement_retains_failure_without_valid_evidence(
    session: AsyncSession,
) -> None:
    source = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    dataset = replace(source, fingerprint="integration-refinement-failure")
    with (
        patch(
            "app.optimization.experiment_service.refine_historical_strategy",
            side_effect=ExperimentMemoryError("deliberate safe failure"),
        ),
        pytest.raises(ExperimentMemoryError, match="deliberate"),
    ):
        await execute_refinement(
            session=session,
            dataset=dataset,
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            budget=RefinementBudget(1, 1, 200, 60),
            persist=True,
            force_recompute=False,
            started_at=END + timedelta(days=4),
        )
    run = await session.scalar(
        select(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.dataset_fingerprint == dataset.fingerprint
        )
    )
    assert run is not None
    assert run.status == "failed"
    assert run.failure_reason and "deliberate safe failure" in run.failure_reason
    assert (
        await session.scalar(
            select(func.count())
            .select_from(StrategyCandidateEvaluationRecord)
            .where(StrategyCandidateEvaluationRecord.refinement_run_id == run.id)
        )
        == 0
    )


async def test_recent_running_is_rejected_and_abandoned_run_completes(
    session: AsyncSession,
) -> None:
    source = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    dataset = replace(source, fingerprint="integration-abandoned-completion")
    initial_start = END + timedelta(days=5)
    await session.execute(
        delete(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.dataset_fingerprint == dataset.fingerprint
        )
    )
    await session.commit()
    with (
        patch(
            "app.optimization.experiment_service.refine_historical_strategy",
            side_effect=SystemExit("simulated process termination"),
        ),
        pytest.raises(SystemExit, match="simulated process termination"),
    ):
        await execute_memory_refinement(session, dataset, initial_start)

    with pytest.raises(ExperimentMemoryError, match="already running"):
        await execute_memory_refinement(
            session, dataset, initial_start + ABANDONED_EXPERIMENT_TIMEOUT
        )

    restarted_at = initial_start + ABANDONED_EXPERIMENT_TIMEOUT + timedelta(seconds=1)
    recovered = await execute_memory_refinement(session, dataset, restarted_at)
    assert recovered.result is not None
    run = await session.scalar(
        select(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.dataset_fingerprint == dataset.fingerprint
        )
    )
    assert run is not None
    assert run.status == "completed"
    assert run.started_at == restarted_at
    assert run.failed_at is None
    assert run.failure_reason is None


async def test_abandoned_run_has_one_atomic_recovery_claim(
    session: AsyncSession,
) -> None:
    source = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    dataset = replace(source, fingerprint="integration-concurrent-recovery")
    started_at = END + timedelta(days=6)
    await session.execute(
        delete(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.dataset_fingerprint == dataset.fingerprint
        )
    )
    await session.commit()
    with (
        patch(
            "app.optimization.experiment_service.refine_historical_strategy",
            side_effect=SystemExit,
        ),
        pytest.raises(SystemExit),
    ):
        await execute_memory_refinement(session, dataset, started_at)
    run = await session.scalar(
        select(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.dataset_fingerprint == dataset.fingerprint
        )
    )
    assert run is not None
    engine = session.bind
    assert isinstance(engine, AsyncEngine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    restarted_at = started_at + ABANDONED_EXPERIMENT_TIMEOUT + timedelta(seconds=1)

    async def claim() -> bool:
        async with factory() as independent:
            claimed = await reclaim_abandoned_run(
                independent,
                run_id=run.id,
                previous_started_at=started_at,
                stale_before=restarted_at - ABANDONED_EXPERIMENT_TIMEOUT,
                restarted_at=restarted_at,
            )
            await independent.commit()
            return claimed is not None

    assert sorted(await asyncio.gather(claim(), claim())) == [False, True]


async def test_recovered_experiment_failure_is_persisted(session: AsyncSession) -> None:
    source = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    dataset = replace(source, fingerprint="integration-abandoned-failure")
    started_at = END + timedelta(days=7)
    await session.execute(
        delete(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.dataset_fingerprint == dataset.fingerprint
        )
    )
    await session.commit()
    with (
        patch(
            "app.optimization.experiment_service.refine_historical_strategy",
            side_effect=SystemExit,
        ),
        pytest.raises(SystemExit),
    ):
        await execute_memory_refinement(session, dataset, started_at)
    with (
        patch(
            "app.optimization.experiment_service.refine_historical_strategy",
            side_effect=ExperimentMemoryError("recovered calculation failed"),
        ),
        pytest.raises(ExperimentMemoryError, match="recovered calculation failed"),
    ):
        await execute_memory_refinement(
            session,
            dataset,
            started_at + ABANDONED_EXPERIMENT_TIMEOUT + timedelta(seconds=1),
        )
    run = await session.scalar(
        select(StrategyRefinementRunRecord).where(
            StrategyRefinementRunRecord.dataset_fingerprint == dataset.fingerprint
        )
    )
    assert run is not None
    assert run.status == "failed"
    assert run.failed_at is not None
    assert run.failure_reason and "recovered calculation failed" in run.failure_reason
