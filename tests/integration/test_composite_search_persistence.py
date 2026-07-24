from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.composite_search import (
    CompositeCandidateResult,
    CompositeCandidateStatus,
    CompositeDepthFunnel,
    CompositeMetrics,
    CompositeOperationalSearchStage,
    CompositeRankingBreakdown,
    CompositeSearchBudget,
    CompositeSearchConclusion,
    CompositeSearchResult,
    CompositeSelectionTrack,
    build_candidate,
)
from app.optimization.composite_search_memory import (
    CompositeSearchCandidateRecord,
    persist_composite_search_result,
)
from app.optimization.composite_search_read import (
    get_composite_search,
    list_composite_candidates,
)
from app.optimization.feature_matrix import HistoricalFeatureRow
from app.optimization.predictive_edge_memory import PredictiveEdgeRunRecord
from app.paper.config import SimulationConfiguration

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2035, 1, 1, tzinfo=UTC)
EXPERIMENT_KEY = "c" * 64


@pytest_asyncio.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(get_settings().database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        await session.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == EXPERIMENT_KEY
            )
        )
        await session.commit()
    yield sessions
    async with sessions() as session:
        await session.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == EXPERIMENT_KEY
            )
        )
        await session.commit()
    await engine.dispose()


def _parent(run_id: UUID) -> PredictiveEdgeRunRecord:
    return PredictiveEdgeRunRecord(
        id=run_id,
        experiment_key=EXPERIMENT_KEY,
        status="completed",
        asset="BTC/USDT",
        provider="Binance",
        market_type="Spot",
        interval="5m",
        dataset_fingerprint="d" * 64,
        range_start=START,
        range_end=START + timedelta(days=730),
        algorithm_version=4,
        score_model_name="baseline_technical",
        score_model_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        neutral_threshold=Decimal("0.10"),
        decision_interval_seconds=1800,
        training_days=180,
        validation_days=60,
        test_days=60,
        step_days=60,
        maximum_windows=8,
        bin_count=3,
        minimum_observations=60,
        minimum_windows=3,
        minimum_stable_direction_ratio=Decimal("0.60"),
        minimum_absolute_correlation=Decimal("0.05"),
        minimum_directional_hit_rate=Decimal("0.55"),
        minimum_hit_rate_improvement_over_benchmark=Decimal("0.02"),
        windows_available=6,
        windows_required=3,
        conclusion_reason=None,
        windows_generated=6,
        aggregate_metrics_computed=10,
        metrics_reused=0,
        compute_avoided_estimate=0,
        reuse_count=0,
        started_at=START,
        completed_at=START + timedelta(hours=1),
        failed_at=None,
        failure_reason=None,
    )


def _result() -> CompositeSearchResult:
    rows = tuple(
        HistoricalFeatureRow(
            START + timedelta(minutes=5 * index),
            Decimal(100 + index),
            Decimal(99 + index),
            Decimal(98 + index),
            Decimal(40 + index),
            Decimal(1 + index) / 10,
            Decimal(index + 1) / 100,
            Decimal(index + 2) / 100,
            Decimal(index + 3) / 10,
        )
        for index in range(8)
    )
    candidate = build_candidate(
        weights=(("return_1", Decimal(1)), ("return_12", Decimal(1))),
        training_rows=rows,
        simulation=SimulationConfiguration(),
        cadence=timedelta(minutes=30),
        horizon=timedelta(minutes=30),
        operational_search_stage=CompositeOperationalSearchStage.LOCAL_REFINEMENT,
        parent_operational_configuration_fingerprint="o" * 64,
        operational_refinement_reason="top_training_operational_family",
    )
    metrics = CompositeMetrics(
        90,
        Decimal("0.2"),
        Decimal("0.6"),
        Decimal("0.02"),
        Decimal("0.01"),
        Decimal("0.01"),
        Decimal("0.67"),
        usable_window_count=3,
        total_return=Decimal("0.03"),
        absolute_pnl=Decimal("300"),
        max_drawdown=Decimal("0.04"),
        trade_count=6,
        fees_paid=Decimal("4.25"),
        final_equity=Decimal("10300"),
        production_baseline_return=Decimal("0.01"),
        excess_return_over_production=Decimal("0.02"),
        buy_and_hold_return=Decimal("0.04"),
        excess_return_over_buy_and_hold=Decimal("-0.01"),
        return_stability=Decimal("0.67"),
    )
    ranking = CompositeRankingBreakdown(
        Decimal("0.01"),
        Decimal("0.6"),
        Decimal("0.67"),
        Decimal(0),
        Decimal(0),
        Decimal("0.68"),
    )
    candidate_result = CompositeCandidateResult(
        candidate,
        None,
        2,
        CompositeCandidateStatus.PASSED_VALIDATION,
        ranking,
        metrics,
        metrics,
        None,
        ("training_economic_qualification", "validation_economic_qualification"),
        (),
        None,
        Decimal(0),
    )
    funnel = CompositeDepthFunnel(2, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0)
    return CompositeSearchResult(
        CompositeSearchConclusion.CANDIDATES_PASSED_VALIDATION,
        2,
        ("return_1", "return_12"),
        {},
        (funnel,),
        (candidate_result,),
        3,
        3,
        0,
        True,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        2,
        candidate.fingerprint,
        None,
    )


async def test_passed_validation_and_exact_counters_round_trip(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    parent_id = uuid4()
    async with factory() as session:
        session.add(_parent(parent_id))
        await session.commit()
        await persist_composite_search_result(
            session=session,
            predictive_edge_run_id=parent_id,
            experiment_key="s" * 64,
            instrument=MARKET_INSTRUMENTS[0].instrument,
            dataset_fingerprint="d" * 64,
            range_start=START,
            range_end=START + timedelta(days=730),
            budget=CompositeSearchBudget(),
            result=_result(),
            started_at=START,
        )

    async with factory() as session:
        summary = await get_composite_search(session, parent_id)
        page = await list_composite_candidates(
            session,
            parent_id,
            depth=2,
            status=CompositeCandidateStatus.PASSED_VALIDATION,
            selection_track=None,
            feature_name="return_1",
            limit=10,
            offset=0,
        )
        assert summary.budget_exhausted is True
        assert summary.total_evaluations == 3
        assert summary.cheap_evaluations_computed == 1
        assert summary.training_simulations_computed == 1
        assert summary.validation_simulations_computed == 1
        assert summary.test_simulations_computed == 0
        assert len(page.items) == 1
        assert page.items[0].status is CompositeCandidateStatus.PASSED_VALIDATION
        assert page.items[0].validation_metrics is not None
        assert page.items[0].test_metrics is None
        assert page.items[0].selection_track is CompositeSelectionTrack.ECONOMIC
        assert page.items[0].operational_search_stage == "local_refinement"
        assert page.items[0].parent_operational_configuration_fingerprint == "o" * 64
        assert (
            page.items[0].operational_refinement_reason
            == "top_training_operational_family"
        )

        economic_page = await list_composite_candidates(
            session,
            parent_id,
            depth=None,
            status=None,
            selection_track=CompositeSelectionTrack.ECONOMIC,
            feature_name=None,
            limit=10,
            offset=0,
        )
        exploratory_page = await list_composite_candidates(
            session,
            parent_id,
            depth=None,
            status=None,
            selection_track=CompositeSelectionTrack.EXPLORATORY,
            feature_name=None,
            limit=10,
            offset=0,
        )
        assert economic_page.total == 1
        assert exploratory_page.total == 0

        record_id = page.items[0].id
        with pytest.raises(IntegrityError):
            await session.execute(
                update(CompositeSearchCandidateRecord)
                .where(CompositeSearchCandidateRecord.id == record_id)
                .values(status="invalid")
            )
            await session.commit()
        await session.rollback()

        with pytest.raises(IntegrityError):
            await session.execute(
                update(CompositeSearchCandidateRecord)
                .where(CompositeSearchCandidateRecord.id == record_id)
                .values(selection_track="exploratory")
            )
            await session.commit()
        await session.rollback()
