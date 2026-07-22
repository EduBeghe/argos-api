from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.edge_guided import (
    EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
    EdgeGuidedCandidate,
    EdgeGuidedCandidateEvaluation,
    EdgeGuidedCandidateStatus,
    EdgeGuidedEvaluationResult,
    EdgeGuidedSegmentEvaluation,
    EdgeGuidedWindowEvaluation,
)
from app.optimization.edge_guided_memory import (
    EdgeGuidedCandidateRecord,
    EdgeGuidedExecution,
    EdgeGuidedRunRecord,
    EdgeGuidedWindowRecord,
    find_reusable_edge_guided_run,
    find_reusable_weekly_research_run,
    persist_edge_guided_result,
)
from app.optimization.edge_guided_read import (
    get_edge_guided_candidate,
    get_edge_guided_run,
    list_edge_guided_candidates,
    list_edge_guided_runs,
)
from app.optimization.edge_guided_schemas import EdgeGuidedActionFilter
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
    existing_id = await session.scalar(
        select(PredictiveEdgeRunRecord.id).where(
            PredictiveEdgeRunRecord.experiment_key == PREDICTIVE_KEY
        )
    )
    if existing_id is not None:
        return PredictiveEdgeExecution(existing_id, True, None, 0, 0)
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
        reuse_key="weekly-" + "f" * 57,
        dataset_revision={"latest_ingested_at": START},
    )
    reused = await find_reusable_edge_guided_run(session, EDGE_KEY)
    assert first.run_id is not None
    assert reused is not None
    assert reused.reused_existing_experiment is True
    assert reused.evaluations_computed == 0
    assert reused.evaluations_reused == 1
    assert reused.result == original
    weekly_reused = await find_reusable_weekly_research_run(
        session, "weekly-" + "f" * 57
    )
    assert weekly_reused is not None
    assert weekly_reused.result == original
    assert (
        reused.result.qualification_contract_version
        == EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
    )
    round_trip = reused.result.evaluations[0]
    assert round_trip.training.sufficient_trades is True
    assert round_trip.training.positive_net_return is True
    assert round_trip.training.beats_production is True
    assert round_trip.training.beats_buy_and_hold is False
    assert round_trip.training.risk_acceptable is True
    assert round_trip.rejection_reasons == ()
    assert reused.result.evaluations[0].test is not None
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


async def test_validation_rejection_round_trip_preserves_nullable_test(
    session: AsyncSession,
) -> None:
    parent = await parent_run(session)
    assert parent.run_id is not None
    original = result()
    evaluation = original.evaluations[0]
    window = replace(
        evaluation.windows[0],
        test=None,
        validation_passed=False,
        test_confirmed=False,
    )
    rejected = replace(
        original,
        evaluations=(
            replace(
                evaluation,
                test=None,
                windows=(window,),
                valid_window_count=0,
                confirmed_window_count=0,
                rejected_validation_window_count=1,
                status=EdgeGuidedCandidateStatus.REJECTED_VALIDATION,
                explanation=(
                    "Isolated test was not executed because validation failed."
                ),
            ),
        ),
    )
    await persist_edge_guided_result(
        session=session,
        experiment_key=EDGE_KEY,
        predictive_edge_run_id=parent.run_id,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(days=1),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        decision_interval=timedelta(minutes=30),
        started_at=START,
        result=rejected,
    )

    loaded = await find_reusable_edge_guided_run(session, EDGE_KEY)
    assert loaded is not None
    assert loaded.result.evaluations[0].test is None
    assert loaded.result.evaluations[0].windows[0].test is None
    candidate_id = await session.scalar(
        select(EdgeGuidedCandidateRecord.id).where(
            EdgeGuidedCandidateRecord.run_id == loaded.run_id
        )
    )
    assert candidate_id is not None
    response = await get_edge_guided_candidate(session, candidate_id)
    assert response.test_metrics is None
    assert response.candidate_family is not None
    assert response.entry_rule is not None
    assert response.exit_rule is not None
    assert response.generation_contract_version == 3
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


async def persist_complete(session: AsyncSession) -> EdgeGuidedExecution:
    parent = await parent_run(session)
    assert parent.run_id is not None
    return await persist_edge_guided_result(
        session=session,
        experiment_key=EDGE_KEY,
        predictive_edge_run_id=parent.run_id,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(days=1),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        decision_interval=timedelta(minutes=30),
        started_at=START,
        result=result(),
    )


async def test_completed_run_missing_candidate_is_not_reused(
    session: AsyncSession,
) -> None:
    execution = await persist_complete(session)
    await session.execute(
        delete(EdgeGuidedCandidateRecord).where(
            EdgeGuidedCandidateRecord.run_id == execution.run_id
        )
    )
    await session.commit()

    assert await find_reusable_edge_guided_run(session, EDGE_KEY) is None


async def test_completed_run_missing_window_is_not_reused(
    session: AsyncSession,
) -> None:
    await persist_complete(session)
    await session.execute(delete(EdgeGuidedWindowRecord))
    await session.commit()

    assert await find_reusable_edge_guided_run(session, EDGE_KEY) is None


async def test_incomplete_run_is_atomically_replaced(session: AsyncSession) -> None:
    incomplete = await persist_complete(session)
    await session.execute(delete(EdgeGuidedWindowRecord))
    await session.commit()

    replacement = await persist_complete(session)

    assert replacement.run_id != incomplete.run_id
    assert replacement.reused_existing_experiment is False
    assert await find_reusable_edge_guided_run(session, EDGE_KEY) is not None
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


async def test_persisting_identical_experiment_twice_does_not_duplicate_graph(
    session: AsyncSession,
) -> None:
    first = await persist_complete(session)
    second = await persist_complete(session)

    assert second.run_id == first.run_id
    assert second.reused_existing_experiment is True
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


async def test_empty_v2_result_is_persisted_and_reused_without_candidates(
    session: AsyncSession,
) -> None:
    parent = await parent_run(session)
    assert parent.run_id is not None
    empty = EdgeGuidedEvaluationResult("f" * 64, 0, 0, 12, 1, ())
    first = await persist_edge_guided_result(
        session=session,
        experiment_key=EDGE_KEY,
        predictive_edge_run_id=parent.run_id,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(days=1),
        initial_cash=Decimal("10000"),
        simulation=SimulationConfiguration(),
        decision_interval=timedelta(minutes=30),
        started_at=START,
        result=empty,
    )
    reused = await find_reusable_edge_guided_run(session, EDGE_KEY)
    assert first.reused_existing_experiment is False
    assert reused is not None
    assert reused.reused_existing_experiment is True
    assert reused.evaluations_reused == 0
    assert reused.result.evaluations == ()
    assert (
        await session.scalar(
            select(func.count()).select_from(EdgeGuidedCandidateRecord)
        )
        == 0
    )
    assert (
        await session.scalar(select(func.count()).select_from(EdgeGuidedWindowRecord))
        == 0
    )


async def test_legacy_v1_run_is_readable_but_not_reusable_or_eligible(
    session: AsyncSession,
) -> None:
    execution = await persist_complete(session)
    assert execution.run_id is not None
    record = await session.get(EdgeGuidedRunRecord, execution.run_id)
    assert record is not None
    record.qualification_contract_version = 1
    await session.commit()

    assert await find_reusable_edge_guided_run(session, EDGE_KEY) is None
    run = await get_edge_guided_run(session, execution.run_id)
    assert run.qualification_contract_version == 1
    assert run.legacy is True
    candidates = await list_edge_guided_candidates(session, run_id=execution.run_id)
    assert candidates.items[0].legacy is True
    assert candidates.items[0].eligible_under_current_contract is False


async def test_insertion_failure_rolls_back_the_complete_graph(
    session: AsyncSession,
) -> None:
    parent = await parent_run(session)
    assert parent.run_id is not None
    original = result()
    duplicate = replace(
        original,
        generated_count=2,
        evaluated_count=2,
        evaluations=(original.evaluations[0], original.evaluations[0]),
    )

    with pytest.raises(IntegrityError):
        await persist_edge_guided_result(
            session=session,
            experiment_key=EDGE_KEY,
            predictive_edge_run_id=parent.run_id,
            instrument=MARKET_INSTRUMENTS[0].instrument,
            range_start=START,
            range_end=START + timedelta(days=1),
            initial_cash=Decimal("10000"),
            simulation=SimulationConfiguration(),
            decision_interval=timedelta(minutes=30),
            started_at=START,
            result=duplicate,
        )

    assert (
        await session.scalar(select(func.count()).select_from(EdgeGuidedRunRecord)) == 0
    )
    assert (
        await session.scalar(
            select(func.count()).select_from(EdgeGuidedCandidateRecord)
        )
        == 0
    )
    assert (
        await session.scalar(select(func.count()).select_from(EdgeGuidedWindowRecord))
        == 0
    )


def result_with_all_statuses() -> EdgeGuidedEvaluationResult:
    original = result()
    confirmed = original.evaluations[0]
    validation_window = replace(
        confirmed.windows[0],
        window_ordinal=2,
        test=None,
        validation_passed=False,
        test_confirmed=False,
    )
    validation_candidate = replace(
        confirmed.candidate,
        candidate_fingerprint="1" * 64,
        source_hypothesis_id="2" * 64,
        suggested_action=SuggestedAction.EXIT_OR_AVOID,
    )
    rejected_validation = replace(
        confirmed,
        candidate=validation_candidate,
        windows=(validation_window,),
        confirmed_window_count=0,
        rejected_validation_window_count=1,
        status=EdgeGuidedCandidateStatus.REJECTED_VALIDATION,
        explanation="validation rejected",
    )
    test_window = replace(confirmed.windows[0], window_ordinal=3, test_confirmed=False)
    test_candidate = replace(
        confirmed.candidate,
        candidate_fingerprint="3" * 64,
        source_hypothesis_id="4" * 64,
        condition=Condition.ALL,
    )
    rejected_test = replace(
        confirmed,
        candidate=test_candidate,
        windows=(test_window,),
        confirmed_window_count=0,
        rejected_test_window_count=1,
        status=EdgeGuidedCandidateStatus.REJECTED_TEST,
        explanation="test rejected",
    )
    return replace(
        original,
        generated_count=3,
        evaluated_count=3,
        evaluations=(confirmed, rejected_validation, rejected_test),
    )


async def test_read_api_queries_are_filtered_paginated_and_decimal_safe(
    session: AsyncSession,
) -> None:
    parent = await parent_run(session)
    assert parent.run_id is not None
    persisted = await persist_edge_guided_result(
        session=session,
        experiment_key=EDGE_KEY,
        predictive_edge_run_id=parent.run_id,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(days=1),
        initial_cash=Decimal("10000.123456789012345678"),
        simulation=SimulationConfiguration(),
        decision_interval=timedelta(minutes=30),
        started_at=START,
        result=result_with_all_statuses(),
    )
    assert persisted.run_id is not None

    runs = await list_edge_guided_runs(
        session, asset=MARKET_INSTRUMENTS[0].instrument.asset, limit=1, offset=0
    )
    assert runs.total == 1
    assert len(runs.items) == 1
    assert runs.items[0].initial_cash == Decimal("10000.123456789012345678")
    assert runs.items[0].candidate_status_counts.model_dump() == {
        "confirmed_candidate": 1,
        "rejected_validation": 1,
        "rejected_test": 1,
    }
    assert (
        runs.items[0].qualification_contract_version
        == EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
    )
    assert runs.items[0].legacy is False
    assert (await get_edge_guided_run(session, persisted.run_id)).id == persisted.run_id

    filtered = await list_edge_guided_candidates(
        session,
        run_id=persisted.run_id,
        status=EdgeGuidedCandidateStatus.REJECTED_VALIDATION,
        action=EdgeGuidedActionFilter.EXIT_OR_AVOID,
        condition=Condition.HIGH_VOLATILITY,
        predictor_name="return_1",
        limit=1,
        offset=0,
    )
    assert filtered.total == 1
    assert len(filtered.items) == 1
    assert not hasattr(filtered.items[0], "windows")
    assert isinstance(filtered.items[0].training_metrics.total_return, Decimal)
    assert filtered.items[0].eligible_under_current_contract is False
    assert isinstance(filtered.items[0].training_metrics.positive_net_return, bool)

    detail = await get_edge_guided_candidate(
        session, filtered.items[0].candidate_evaluation_id
    )
    assert tuple(item.window_ordinal for item in detail.windows) == (2,)
    assert detail.windows[0].test_metrics is None


async def test_empty_read_api_pages_do_not_write(session: AsyncSession) -> None:
    before = (
        await session.scalar(select(func.count()).select_from(EdgeGuidedRunRecord)),
        await session.scalar(
            select(func.count()).select_from(EdgeGuidedCandidateRecord)
        ),
        await session.scalar(select(func.count()).select_from(EdgeGuidedWindowRecord)),
    )
    assert (await list_edge_guided_runs(session)).items == ()
    assert (await list_edge_guided_candidates(session)).items == ()
    after = (
        await session.scalar(select(func.count()).select_from(EdgeGuidedRunRecord)),
        await session.scalar(
            select(func.count()).select_from(EdgeGuidedCandidateRecord)
        ),
        await session.scalar(select(func.count()).select_from(EdgeGuidedWindowRecord)),
    )
    assert after == before == (0, 0, 0)
