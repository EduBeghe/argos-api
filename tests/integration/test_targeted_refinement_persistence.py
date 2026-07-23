from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.optimization.edge_guided import EdgeGuidedSegmentEvaluation
from app.optimization.predictive_edge_engine_v41 import (
    ContextRegime,
    FrozenRegime,
    PhaseEvidence,
    RegimeConditionedRelationship,
    V41Candidate,
)
from app.optimization.predictive_edge_memory import PredictiveEdgeRunRecord
from app.optimization.predictive_edges import DirectionMapping, SuggestedAction
from app.optimization.refinement import RefinementSegmentName
from app.optimization.targeted_refinement import (
    TargetedCandidateResult,
    TargetedCandidateStatus,
    TargetedConfiguration,
    TargetedRefinementConclusion,
    TargetedRefinementResult,
    TargetedRefinementSummary,
    TargetedRequirementMargin,
    TargetedSeed,
    _edge_candidate,
)
from app.optimization.targeted_refinement_memory import (
    persist_targeted_refinement_result,
)
from app.optimization.targeted_refinement_read import (
    get_targeted_candidate,
    get_targeted_refinement,
    list_targeted_candidates,
)
from app.paper.config import SimulationConfiguration

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2034, 1, 1, tzinfo=UTC)
EXPERIMENT_KEY = "t" * 64


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


def _seed() -> TargetedSeed:
    evidence = PhaseEvidence(
        96,
        Decimal("0.006"),
        Decimal("0.006"),
        Decimal("0.003"),
        Decimal("0.60"),
        Decimal("0.52"),
        Decimal("0.08"),
        Decimal("0.001"),
        False,
        True,
        False,
        6,
        2,
        4,
        0,
    )
    regime = FrozenRegime(
        "trend_regime", 2, ContextRegime.BULLISH, Decimal(0), None, 96, "r" * 64
    )
    relationship = RegimeConditionedRelationship(
        "ema_50_slope_48",
        2,
        "trend_regime",
        2,
        ContextRegime.BULLISH,
        timedelta(hours=6),
        timedelta(hours=24),
        regime,
        2,
        Decimal("-0.01"),
        Decimal("0.01"),
        DirectionMapping.POSITIVE,
        SuggestedAction.LONG_ENTRY,
        "balanced_v41",
        evidence,
        evidence,
        Decimal("0.001"),
        "s" * 64,
    )
    return TargetedSeed(
        V41Candidate(relationship, evidence, None, False, False, "unstable"),
        Decimal("0.25"),
    )


def _candidate(
    seed: TargetedSeed,
    *,
    fingerprint: str,
    status: TargetedCandidateStatus,
    distance: Decimal,
) -> TargetedCandidateResult:
    configuration = TargetedConfiguration(
        fingerprint,
        seed.candidate.relationship.fingerprint,
        timedelta(hours=6),
        timedelta(hours=24),
        2,
        Decimal("-0.01"),
        Decimal("0.01"),
    )
    candidate = _edge_candidate(configuration, seed, SimulationConfiguration())
    training = EdgeGuidedSegmentEvaluation(
        RefinementSegmentName.TRAINING,
        Decimal("0.02"),
        Decimal("0.04"),
        4,
        Decimal("2"),
        Decimal("0.01"),
        Decimal("0.005"),
    )
    validation = replace(training, segment=RefinementSegmentName.VALIDATION)
    test = replace(training, segment=RefinementSegmentName.TEST)
    margin = TargetedRequirementMargin(
        "isolated_test.stable_confirmation_ratio",
        Decimal("0.7"),
        Decimal("0.6"),
        Decimal("0.1"),
        Decimal("0.166666666666666667"),
        True,
    )
    return TargetedCandidateResult(
        configuration,
        candidate,
        status,
        None,
        training,
        validation,
        test if status is TargetedCandidateStatus.CONFIRMED else None,
        (margin.requirement,),
        (),
        (margin,),
        None,
        distance,
    )


async def test_production_boundary_round_trips_ranked_targeted_evidence(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    parent_id = uuid4()
    seed = _seed()
    confirmed = _candidate(
        seed,
        fingerprint="c" * 64,
        status=TargetedCandidateStatus.CONFIRMED,
        distance=Decimal("0.20"),
    )
    rejected = _candidate(
        seed,
        fingerprint="a" * 64,
        status=TargetedCandidateStatus.REJECTED_VALIDATION,
        distance=Decimal("0.01"),
    )
    summary = TargetedRefinementSummary(1, 2, 0, 2, 2, 2, 1, 1, 1, 6, 0, Decimal("1.5"))
    result = TargetedRefinementResult(
        TargetedRefinementConclusion.CONFIRMED,
        (seed,),
        (confirmed.configuration, rejected.configuration),
        (rejected, confirmed),
        summary,
    )
    async with factory() as session:
        session.add(_parent(parent_id))
        await session.commit()
        targeted_run_id = await persist_targeted_refinement_result(
            session=session,
            predictive_edge_run_id=parent_id,
            experiment_key="x" * 64,
            result=result,
        )

    async with factory() as restarted:
        persisted = await get_targeted_refinement(restarted, parent_id)
        page = await list_targeted_candidates(
            restarted, parent_id, status=None, limit=1, offset=0
        )
        second = await list_targeted_candidates(
            restarted, parent_id, status=None, limit=1, offset=1
        )
        filtered = await list_targeted_candidates(
            restarted,
            parent_id,
            status=TargetedCandidateStatus.REJECTED_VALIDATION,
            limit=10,
            offset=0,
        )
        detail = await get_targeted_candidate(restarted, parent_id, page.items[0].id)

    assert persisted.id == targeted_run_id
    assert persisted.best_targeted_candidate == "c" * 64
    assert persisted.best_targeted_candidate_id == page.items[0].id
    assert page.items[0].status is TargetedCandidateStatus.CONFIRMED
    assert page.has_more is True and page.next_cursor == "1"
    assert second.items[0].candidate_fingerprint == "a" * 64
    assert filtered.total == 1
    assert detail.seed_fingerprint == "s" * 64
    assert detail.configuration["fee_rate"] == "0.001"
    assert detail.training_metrics is not None
    assert detail.validation_metrics is not None
    assert detail.test_metrics is not None
    assert detail.passed_requirements
    assert detail.requirement_margins[0]["observed_value"] == "0.7"
