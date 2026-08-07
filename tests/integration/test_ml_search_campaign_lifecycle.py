from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ml.campaign import planning as campaign_planning
from app.ml.campaign import service as campaign_service
from app.ml.campaign.contracts import (
    ECONOMIC_POLICY_SEARCH_V2_PRESET,
    ECONOMIC_V2_DIVERSIFIED_FAMILIES,
    FAMILIES_PER_INSTRUMENT,
    CampaignRequest,
)
from app.ml.campaign.coordinator import coordinate_next_campaign
from app.ml.campaign.models import (
    MLSearchCampaignChildRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
    MLSearchCampaignSourceRecord,
)
from app.ml.campaign.planning import CampaignFamily, SourceInstrument
from app.ml.campaign.service import (
    CampaignConflictError,
    CampaignSourceError,
    cancel_campaign,
    enqueue_campaign,
    preview_campaign,
    retry_campaign,
)
from app.ml.contract import FUTURE_DURATIONS, temporal_fold_definitions
from app.ml.protected.audit import assemble_family_audit
from app.ml.protected.models import MLFinalValidationRecord, MLTemporalManifestRecord
from app.ml.protected.queue import (
    ProtectedWorkError,
    enqueue_final_validation,
    enqueue_manifest_reconstruction,
)
from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)
from app.ml.refinement.planning import build_refinement_plan
from app.ml.screening.config import ScreeningEngine
from app.ml.screening.models import (
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
PREFIX = "ml-campaign-lifecycle-"
NOW = datetime(2040, 1, 1, tzinfo=UTC)


def refinement_request(source_id: UUID, result_id: UUID) -> MLRefinementRequest:
    return MLRefinementRequest.model_validate(
        {
            "source_screening_run_id": source_id,
            "source_screening_result_id": result_id,
            "engines": ["ridge"],
            "future_durations_seconds": [1800],
            "search_space": [
                {
                    "engine": "ridge",
                    "alpha": {"values": [1.0]},
                    "fit_intercept": {"values": [True]},
                }
            ],
            "feature_selection": {"strategies": [{"strategy": "all"}]},
            "prediction_objectives": ["future_return_regression"],
            "confidence_buckets": ["0.10"],
            "fee_rate": "0.001",
            "slippage_rate": "0.0005",
            "maximum_trials": 1,
            "maximum_runtime_seconds": 60,
            "deterministic_seed": 1729,
        }
    )


def campaign(
    *,
    status: str = "running",
    first_claimed_at: datetime | None = NOW,
    covered: bool = False,
    maximum_evaluations: int = 3600,
) -> MLSearchCampaignRecord:
    identifier = uuid4()
    return MLSearchCampaignRecord(
        id=identifier,
        experiment_key=f"{PREFIX}{identifier}"[:64],
        status=status,
        outcome=None,
        contract_version=1,
        policy_version=1,
        classification_version=1,
        normalized_request={},
        maximum_evaluations=maximum_evaluations,
        maximum_runtime_seconds=60,
        families_total=1,
        families_covered=1 if covered else 0,
        families_uncovered=0 if covered else 1,
        evaluations_computed=0,
        evaluations_reused=0,
        current_stage=1,
        cancellation_requested_at=None,
        first_claimed_at=first_claimed_at,
        completed_at=None,
        failed_at=NOW if status == "failed" else None,
        failure_type="ChildRefinementFailed" if status == "failed" else None,
        failure_message="child failed" if status == "failed" else None,
        created_at=NOW,
    )


def family(
    campaign_id: UUID, source_id: UUID, *, state: str = "scheduled"
) -> MLSearchCampaignFamilyRecord:
    identifier = uuid4()
    return MLSearchCampaignFamilyRecord(
        id=identifier,
        campaign_id=campaign_id,
        source_screening_run_id=source_id,
        fingerprint=f"{identifier.hex:0<64}"[:64],
        dataset_fingerprint="d" * 64,
        asset="BTC/USDT",
        provider="binance",
        market_type="spot",
        interval="5m",
        engine="ridge",
        future_duration_seconds=1800,
        objective="future_return_regression",
        feature_variant="all",
        state=state,
        current_stage=0,
        evaluations_planned=1,
        evaluations_completed=0,
        evaluations_reused=0,
        best_evaluation_id=None,
        best_metrics=None,
        decision_reason=None,
        deterministic_ordinal=0,
        created_at=NOW,
    )


def refinement_run(
    source_id: UUID,
    result_id: UUID,
    *,
    status: str = "queued",
) -> MLRefinementRunRecord:
    request = refinement_request(source_id, result_id)
    plan = build_refinement_plan(request)
    identifier = uuid4()
    terminal = status in {"completed", "cancelled"}
    failed = status == "failed"
    started = status in {"running", "completed", "failed"}
    return MLRefinementRunRecord(
        id=identifier,
        experiment_key=f"{PREFIX}refinement-{identifier}"[:64],
        source_screening_run_id=source_id,
        source_screening_result_id=result_id,
        status=status,
        contract_version=1,
        classification_version=1,
        dataset_fingerprint="d" * 64,
        variable_registry_version=1,
        missing_data_policy_version=1,
        library_versions={},
        normalized_configuration=request.normalized_payload(),
        notes=None,
        maximum_trials=request.maximum_trials,
        maximum_runtime_seconds=request.maximum_runtime_seconds,
        combinations_planned=plan.total_combinations,
        evaluations_computed=0,
        evaluations_reused=0,
        evaluations_omitted=plan.omitted_by_trial_budget,
        evaluations_selected=len(plan.evaluations),
        budget_exhausted=plan.omitted_by_trial_budget > 0,
        termination_reason="failed" if failed else None,
        started_at=NOW if started else None,
        completed_at=NOW if terminal else None,
        failed_at=NOW if failed else None,
        failure_type="TestFailure" if failed else None,
        failure_message="expected" if failed else None,
        worker_id="worker" if status == "running" else None,
        claimed_at=NOW if status == "running" else None,
        heartbeat_at=NOW if status == "running" else None,
        cancellation_requested_at=None,
        elapsed_seconds=Decimal("1") if terminal or failed else None,
        created_at=NOW,
    )


def child(
    campaign_id: UUID,
    family_id: UUID,
    run_id: UUID,
    *,
    reused: bool,
) -> MLSearchCampaignChildRecord:
    return MLSearchCampaignChildRecord(
        id=uuid4(),
        campaign_id=campaign_id,
        family_id=family_id,
        refinement_run_id=run_id,
        stage=1,
        reserved_evaluations=1,
        reuses_existing_run=reused,
        created_at=NOW,
    )


def evaluation(
    run_id: UUID, fingerprint: str = "e" * 64
) -> MLRefinementEvaluationRecord:
    return MLRefinementEvaluationRecord(
        id=uuid4(),
        run_id=run_id,
        reused_from_evaluation_id=None,
        fingerprint=fingerprint,
        engine="ridge",
        future_duration_seconds=1800,
        objective="future_return_regression",
        hyperparameters={},
        feature_policy={},
        confidence_buckets=["0.10"],
        fold_evidence=[],
        aggregate_metrics={
            "selected_confidence_bucket": {
                "observation_count": 100,
                "folds_with_favorable_direction": 5,
                "folds_with_positive_net_return": 5,
                "median_net_return": "0.01",
                "worst_net_return": "-0.001",
            },
            "folds_beating_benchmark": 5,
            "complete_fold_count": 7,
        },
        classification="promising_for_strategy_simulation",
        classification_version=1,
        failed_rules=[],
        elapsed_seconds=Decimal("1"),
        created_at=NOW,
    )


@pytest_asyncio.fixture
async def factory() -> AsyncIterator[
    tuple[async_sessionmaker[AsyncSession], UUID, UUID]
]:
    engine = create_async_engine(get_settings().database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    source_id, result_id = uuid4(), uuid4()
    async with sessions() as session:
        session.add(
            MLModelScreeningRunRecord(
                id=source_id,
                experiment_key=f"{PREFIX}source".ljust(64, "s")[:64],
                status="completed",
                screening_version=1,
                experiment_contract_version=2,
                variable_registry_version=1,
                missing_data_policy_version=1,
                dataset_fingerprint="d" * 64,
                asset="BTC/USDT",
                provider="binance",
                market_type="spot",
                interval="5m",
                requested_engines=["ridge"],
                requested_future_durations=[1800],
                library_versions={},
                deterministic_seed=1729,
                configuration={},
                started_at=NOW,
                completed_at=NOW,
                failure_type=None,
                failure_message=None,
                evaluations_computed=1,
                evaluations_reused=0,
                results_generated=1,
                pareto_retained_count=1,
                elapsed_seconds=Decimal("1"),
                created_at=NOW,
            )
        )
        await session.flush()
        session.add(
            MLModelScreeningResultRecord(
                id=result_id,
                run_id=source_id,
                engine="ridge",
                future_duration_seconds=1800,
                folds_evaluated=7,
                total_validation_samples=700,
                median_mae_improvement_over_zero=Decimal("0.1"),
                worst_fold_mae_improvement_over_zero=Decimal("0.01"),
                median_directional_accuracy=Decimal("0.55"),
                worst_fold_directional_accuracy=Decimal("0.5"),
                median_pearson_correlation=Decimal("0.1"),
                median_spearman_correlation=Decimal("0.1"),
                folds_beating_zero=7,
                folds_with_positive_spearman=7,
                total_training_time_seconds=Decimal("1"),
                pareto_retained=True,
                display_rank=1,
                created_at=NOW,
            )
        )
        await session.commit()
    yield sessions, source_id, result_id
    async with sessions() as session:
        campaign_ids = select(MLSearchCampaignRecord.id).where(
            MLSearchCampaignRecord.experiment_key.like(f"{PREFIX}%")
        )
        await session.execute(
            delete(MLSearchCampaignRecord).where(
                MLSearchCampaignRecord.id.in_(campaign_ids)
            )
        )
        await session.execute(
            delete(MLRefinementRunRecord).where(
                MLRefinementRunRecord.source_screening_run_id == source_id
            )
        )
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.id == source_id
            )
        )
        await session.commit()
    await engine.dispose()


@pytest_asyncio.fixture
async def enqueue_factory() -> AsyncIterator[
    tuple[async_sessionmaker[AsyncSession], UUID]
]:
    engine = create_async_engine(get_settings().database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    source_id = uuid4()
    durations = tuple(int(item.total_seconds()) for item in FUTURE_DURATIONS)
    engines = tuple(item.value for item in ScreeningEngine)
    async with sessions() as session:
        session.add(
            MLModelScreeningRunRecord(
                id=source_id,
                experiment_key=f"{PREFIX}enqueue-source".ljust(64, "s")[:64],
                status="completed",
                screening_version=1,
                experiment_contract_version=2,
                variable_registry_version=1,
                missing_data_policy_version=1,
                dataset_fingerprint="q" * 64,
                asset="BTC/USDT",
                provider="binance",
                market_type="spot",
                interval="5m",
                requested_engines=list(engines),
                requested_future_durations=list(durations),
                library_versions={},
                deterministic_seed=1729,
                configuration={},
                started_at=NOW,
                completed_at=NOW,
                failure_type=None,
                failure_message=None,
                evaluations_computed=len(engines) * len(durations),
                evaluations_reused=0,
                results_generated=len(engines) * len(durations),
                pareto_retained_count=1,
                elapsed_seconds=Decimal("1"),
                created_at=NOW,
            )
        )
        await session.flush()
        session.add_all(
            [
                MLModelScreeningResultRecord(
                    id=uuid4(),
                    run_id=source_id,
                    engine=engine_name,
                    future_duration_seconds=duration,
                    folds_evaluated=7,
                    total_validation_samples=700,
                    median_mae_improvement_over_zero=Decimal("0.1"),
                    worst_fold_mae_improvement_over_zero=Decimal("0.01"),
                    median_directional_accuracy=Decimal("0.55"),
                    worst_fold_directional_accuracy=Decimal("0.5"),
                    median_pearson_correlation=Decimal("0.1"),
                    median_spearman_correlation=Decimal("0.1"),
                    folds_beating_zero=7,
                    folds_with_positive_spearman=7,
                    total_training_time_seconds=Decimal("1"),
                    pareto_retained=True,
                    display_rank=display_rank,
                    created_at=NOW,
                )
                for display_rank, (engine_name, duration) in enumerate(
                    (
                        (engine_name, duration)
                        for engine_name in engines
                        for duration in durations
                    ),
                    start=1,
                )
            ]
        )
        await session.commit()
    yield sessions, source_id
    async with sessions() as session:
        campaign_ids = select(MLSearchCampaignSourceRecord.campaign_id).where(
            MLSearchCampaignSourceRecord.source_screening_run_id == source_id
        )
        await session.execute(
            delete(MLSearchCampaignRecord).where(
                MLSearchCampaignRecord.id.in_(campaign_ids)
            )
        )
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.id == source_id
            )
        )
        await session.commit()
    await engine.dispose()


async def test_enqueue_flushes_parent_before_3600_children_and_is_idempotent(
    enqueue_factory: tuple[async_sessionmaker[AsyncSession], UUID],
) -> None:
    sessions, source_id = enqueue_factory
    request = CampaignRequest(source_screening_run_ids=(source_id,))
    async with sessions() as session:
        first = await enqueue_campaign(session, request, now=NOW)
    assert first.created is True
    assert first.campaign.status == "queued"

    async with sessions() as session:
        source_campaign_ids = tuple(
            (
                await session.scalars(
                    select(MLSearchCampaignSourceRecord.campaign_id).where(
                        MLSearchCampaignSourceRecord.campaign_id == first.campaign.id
                    )
                )
            ).all()
        )
        family_campaign_ids = tuple(
            (
                await session.scalars(
                    select(MLSearchCampaignFamilyRecord.campaign_id).where(
                        MLSearchCampaignFamilyRecord.campaign_id == first.campaign.id
                    )
                )
            ).all()
        )
    assert source_campaign_ids == (first.campaign.id,)
    assert len(family_campaign_ids) == FAMILIES_PER_INSTRUMENT
    assert set(family_campaign_ids) == {first.campaign.id}

    async with sessions() as session:
        second = await enqueue_campaign(session, request, now=NOW)
    assert second.created is False
    assert second.campaign.id == first.campaign.id
    async with sessions() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(MLSearchCampaignRecord)
                .where(
                    MLSearchCampaignRecord.experiment_key
                    == first.campaign.experiment_key
                )
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(MLSearchCampaignSourceRecord)
                .where(MLSearchCampaignSourceRecord.campaign_id == first.campaign.id)
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(MLSearchCampaignFamilyRecord)
                .where(MLSearchCampaignFamilyRecord.campaign_id == first.campaign.id)
            )
            == FAMILIES_PER_INSTRUMENT
        )


async def test_v2_enqueue_is_bounded_persisted_and_idempotent(
    enqueue_factory: tuple[async_sessionmaker[AsyncSession], UUID],
) -> None:
    sessions, source_id = enqueue_factory
    request = CampaignRequest(
        source_screening_run_ids=(source_id,),
        preset=ECONOMIC_POLICY_SEARCH_V2_PRESET,
        maximum_evaluations=ECONOMIC_V2_DIVERSIFIED_FAMILIES,
    )
    async with sessions() as session:
        first = await enqueue_campaign(session, request, now=NOW)
        second = await enqueue_campaign(session, request, now=NOW)
        families = tuple(
            (
                await session.scalars(
                    select(MLSearchCampaignFamilyRecord)
                    .where(
                        MLSearchCampaignFamilyRecord.campaign_id == first.campaign.id
                    )
                    .order_by(MLSearchCampaignFamilyRecord.deterministic_ordinal)
                )
            ).all()
        )
    assert first.created is True
    assert second.created is False
    assert second.campaign.id == first.campaign.id
    assert len(families) == ECONOMIC_V2_DIVERSIFIED_FAMILIES
    assert {item.preset for item in families} == {ECONOMIC_POLICY_SEARCH_V2_PRESET}
    assert {item.search_track for item in families} == {"diversified"}
    assert len({item.fingerprint for item in families}) == len(families)


async def test_v2_invalid_seed_rejects_the_whole_enqueue_without_writes(
    enqueue_factory: tuple[async_sessionmaker[AsyncSession], UUID],
) -> None:
    sessions, source_id = enqueue_factory
    request = CampaignRequest(
        source_screening_run_ids=(source_id,),
        preset=ECONOMIC_POLICY_SEARCH_V2_PRESET,
        seed_family_ids=(uuid4(),),
        maximum_evaluations=432,
    )
    async with sessions() as session:
        with pytest.raises(CampaignSourceError, match="seed families do not exist"):
            await enqueue_campaign(session, request, now=NOW)
        assert (
            await session.scalar(
                select(func.count())
                .select_from(MLSearchCampaignSourceRecord)
                .where(
                    MLSearchCampaignSourceRecord.source_screening_run_id == source_id
                )
            )
            == 0
        )


async def test_child_integrity_failure_rolls_back_flushed_parent(
    enqueue_factory: tuple[async_sessionmaker[AsyncSession], UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, source_id = enqueue_factory
    request = CampaignRequest(source_screening_run_ids=(source_id,))
    async with sessions() as session:
        _, _, experiment_key = await preview_campaign(session, request)
    original_build = campaign_planning.build_family_universe
    invalid_source_id = uuid4()

    def invalid_family_universe(
        sources: tuple[SourceInstrument, ...],
    ) -> tuple[CampaignFamily, ...]:
        universe = original_build(sources)
        invalid_source = replace(
            universe[0].source, source_run_id=str(invalid_source_id)
        )
        return (replace(universe[0], source=invalid_source), *universe[1:])

    monkeypatch.setattr(
        campaign_service, "build_family_universe", invalid_family_universe
    )
    async with sessions() as session:
        with pytest.raises(IntegrityError):
            await enqueue_campaign(session, request, now=NOW)

    async with sessions() as session:
        assert (
            await session.scalar(
                select(MLSearchCampaignRecord).where(
                    MLSearchCampaignRecord.experiment_key == experiment_key
                )
            )
            is None
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(MLSearchCampaignSourceRecord)
                .where(
                    MLSearchCampaignSourceRecord.source_screening_run_id == source_id
                )
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(MLSearchCampaignFamilyRecord)
                .where(
                    MLSearchCampaignFamilyRecord.source_screening_run_id.in_(
                        (source_id, invalid_source_id)
                    )
                )
            )
            == 0
        )


async def test_two_campaigns_can_share_one_refinement_run_and_cancel_independently(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, source_id, result_id = factory
    first, second = campaign(), campaign()
    first_family, second_family = (
        family(first.id, source_id),
        family(second.id, source_id),
    )
    run = refinement_run(source_id, result_id)
    async with sessions() as session:
        session.add_all((first, second))
        await session.flush()
        session.add_all((first_family, second_family, run))
        await session.flush()
        session.add_all(
            (
                child(first.id, first_family.id, run.id, reused=False),
                child(second.id, second_family.id, run.id, reused=True),
            )
        )
        await session.commit()

    async with sessions() as session:
        await cancel_campaign(session, first.id, now=NOW)
    async with sessions() as session:
        assert (
            await session.get(MLSearchCampaignRecord, first.id)
        ).status == "cancelled"  # type: ignore[union-attr]
        assert (await session.get(MLRefinementRunRecord, run.id)).status == "queued"  # type: ignore[union-attr]
        assert (
            await session.scalar(
                select(func.count(MLSearchCampaignChildRecord.id)).where(
                    MLSearchCampaignChildRecord.refinement_run_id == run.id
                )
            )
            == 2
        )

    async with sessions() as session:
        await cancel_campaign(session, second.id, now=NOW)
    async with sessions() as session:
        assert (await session.get(MLRefinementRunRecord, run.id)).status == "cancelled"  # type: ignore[union-attr]


@pytest.mark.parametrize("run_status", ["failed", "cancelled", "completed"])
async def test_retry_requeues_same_campaign_child_and_refinement_run(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID], run_status: str
) -> None:
    sessions, source_id, result_id = factory
    record = campaign(status="failed")
    family_record = family(record.id, source_id)
    run = refinement_run(source_id, result_id, status=run_status)
    link = child(record.id, family_record.id, run.id, reused=False)
    async with sessions() as session:
        session.add(record)
        await session.flush()
        session.add_all((family_record, run))
        await session.flush()
        session.add(link)
        await session.commit()

    async with sessions() as session:
        retried = await retry_campaign(session, record.id)
        assert retried.id == record.id
        assert retried.status == "queued"
    async with sessions() as session:
        persisted = await session.get(MLRefinementRunRecord, run.id)
        assert persisted is not None
        assert persisted.status == "queued"
        assert persisted.started_at is None
        assert persisted.failed_at is None
        assert persisted.completed_at is None
        assert (
            await session.scalar(
                select(func.count(MLSearchCampaignChildRecord.id)).where(
                    MLSearchCampaignChildRecord.campaign_id == record.id
                )
            )
            == 1
        )
        with pytest.raises(
            CampaignConflictError,
            match="only operationally failed campaigns can be retried",
        ):
            await retry_campaign(session, record.id)


async def test_failed_campaign_without_child_is_not_retryable(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, _, _ = factory
    record = campaign(status="failed")
    async with sessions() as session:
        session.add(record)
        await session.commit()
        with pytest.raises(
            CampaignConflictError, match="no retryable responsible child"
        ):
            await retry_campaign(session, record.id)


async def test_initial_coordinator_claim_sets_runtime_clock_once(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, source_id, _ = factory
    claimed_at = datetime(2040, 1, 2, tzinfo=UTC)
    record = campaign(status="queued", first_claimed_at=None)
    family_record = family(record.id, source_id, state="pending")
    async with sessions() as session:
        session.add(record)
        await session.flush()
        session.add(family_record)
        await session.commit()
    async with sessions() as session:
        assert await coordinate_next_campaign(session, now=claimed_at)
    async with sessions() as session:
        persisted = await session.get(MLSearchCampaignRecord, record.id)
        assert persisted is not None
        assert persisted.status == "running"
        assert persisted.first_claimed_at == claimed_at
        assert (
            await session.scalar(
                select(func.count(MLSearchCampaignChildRecord.id)).where(
                    MLSearchCampaignChildRecord.campaign_id == record.id
                )
            )
            == 1
        )


async def test_retry_reclaim_preserves_original_runtime_clock(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, source_id, result_id = factory
    record = campaign(status="failed", first_claimed_at=NOW)
    family_record = family(record.id, source_id)
    run = refinement_run(source_id, result_id, status="failed")
    link = child(record.id, family_record.id, run.id, reused=False)
    async with sessions() as session:
        session.add(record)
        await session.flush()
        session.add_all((family_record, run))
        await session.flush()
        session.add(link)
        await session.commit()
        await retry_campaign(session, record.id)
    async with sessions() as session:
        assert await coordinate_next_campaign(session, now=NOW.replace(second=30))
        await session.commit()
    async with sessions() as session:
        persisted = await session.get(MLSearchCampaignRecord, record.id)
        assert persisted is not None
        assert persisted.status == "running"
        assert persisted.first_claimed_at == NOW


@pytest.mark.parametrize(
    ("covered", "expected_outcome"),
    [(False, "coverage_incomplete"), (True, "search_budget_exhausted")],
)
async def test_retry_at_original_runtime_limit_schedules_no_new_child(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
    covered: bool,
    expected_outcome: str,
) -> None:
    sessions, source_id, result_id = factory
    record = campaign(status="failed", first_claimed_at=NOW, covered=covered)
    family_record = family(record.id, source_id)
    run = refinement_run(source_id, result_id, status="failed")
    link = child(record.id, family_record.id, run.id, reused=False)
    async with sessions() as session:
        session.add(record)
        await session.flush()
        session.add_all((family_record, run))
        await session.flush()
        session.add(link)
        await session.commit()
        await retry_campaign(session, record.id)
    async with sessions() as session:
        assert await coordinate_next_campaign(
            session,
            now=NOW.replace(minute=1),
        )
    async with sessions() as session:
        persisted = await session.get(MLSearchCampaignRecord, record.id)
        assert persisted is not None
        assert persisted.status == "completed"
        assert persisted.outcome == expected_outcome
        assert persisted.first_claimed_at == NOW
        assert (
            await session.scalar(
                select(func.count(MLSearchCampaignChildRecord.id)).where(
                    MLSearchCampaignChildRecord.campaign_id == record.id
                )
            )
            == 1
        )


async def test_completed_child_reconciliation_is_idempotent_across_sessions(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, source_id, result_id = factory
    record = campaign(maximum_evaluations=1)
    family_record = family(record.id, source_id)
    run = refinement_run(source_id, result_id, status="completed")
    run.evaluations_computed = 1
    run.evaluations_selected = 1
    evidence = evaluation(run.id)
    link = child(record.id, family_record.id, run.id, reused=False)
    async with sessions() as session:
        session.add(record)
        await session.flush()
        session.add_all((family_record, run))
        await session.flush()
        session.add_all((evidence, link))
        await session.commit()
    async with sessions() as session:
        assert await coordinate_next_campaign(session, now=NOW)
    async with sessions() as session:
        first_campaign = await session.get(MLSearchCampaignRecord, record.id)
        first_family = await session.get(MLSearchCampaignFamilyRecord, family_record.id)
        assert first_campaign is not None
        assert first_family is not None
        snapshot = (
            first_campaign.families_covered,
            first_campaign.families_uncovered,
            first_campaign.evaluations_computed,
            first_campaign.evaluations_reused,
            first_family.current_stage,
            first_family.state,
            first_family.best_evaluation_id,
            first_family.best_metrics,
            await session.scalar(
                select(func.count(MLSearchCampaignChildRecord.id)).where(
                    MLSearchCampaignChildRecord.campaign_id == record.id
                )
            ),
        )
    async with sessions() as session:
        assert not await coordinate_next_campaign(session, now=NOW)
    async with sessions() as session:
        final_campaign = await session.get(MLSearchCampaignRecord, record.id)
        final_family = await session.get(MLSearchCampaignFamilyRecord, family_record.id)
        assert final_campaign is not None
        assert final_family is not None
        assert (
            final_campaign.families_covered,
            final_campaign.families_uncovered,
            final_campaign.evaluations_computed,
            final_campaign.evaluations_reused,
            final_family.current_stage,
            final_family.state,
            final_family.best_evaluation_id,
            final_family.best_metrics,
            await session.scalar(
                select(func.count(MLSearchCampaignChildRecord.id)).where(
                    MLSearchCampaignChildRecord.campaign_id == record.id
                )
            ),
        ) == snapshot


async def test_legacy_confidence_bucket_audits_and_enqueues_idempotently(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, source_id, result_id = factory
    record = campaign(status="completed", covered=True, maximum_evaluations=1)
    family_record = family(record.id, source_id, state="strong_development_candidate")
    family_record.current_stage = 4
    family_record.objective = "future_direction_classification"
    run = refinement_run(source_id, result_id, status="completed")
    run.evaluations_computed = 1
    run.evaluations_selected = 1
    evidence = evaluation(run.id)
    evidence.objective = "future_direction_classification"
    evidence.confidence_buckets = ["0.05", "0.10", "0.20"]
    evidence.fold_evidence = [
        {
            "fold_name": definition.name,
            "validation_row_count": 100,
            "confidence_metrics": [],
        }
        for definition in temporal_fold_definitions()
    ]
    evidence.aggregate_metrics = {
        **evidence.aggregate_metrics,
        "selected_confidence_bucket": {
            **evidence.aggregate_metrics["selected_confidence_bucket"],  # type: ignore[dict-item]
            "fraction": "0.050000000000000000",
        },
        "complete_fold_count": 7,
    }
    family_record.best_evaluation_id = evidence.id
    link = child(record.id, family_record.id, run.id, reused=False)
    link.stage = 4
    try:
        async with sessions() as session:
            session.add(record)
            await session.flush()
            session.add(run)
            await session.flush()
            session.add(evidence)
            await session.flush()
            session.add(family_record)
            await session.flush()
            session.add(link)
            await session.commit()

        async with sessions() as session:
            audit = await assemble_family_audit(
                session, campaign_id=record.id, family_id=family_record.id
            )
            assert audit is not None
            assert audit.evidence_integrity.failure_codes == ()
            assert audit.temporal_provenance_integrity.failure_codes == (
                "development_manifest_missing",
            )
            persisted_evidence = await session.get(
                MLRefinementEvaluationRecord, evidence.id
            )
            assert persisted_evidence is not None
            nearby_metrics = dict(persisted_evidence.aggregate_metrics)
            raw_nearby_selected = nearby_metrics["selected_confidence_bucket"]
            assert isinstance(raw_nearby_selected, dict)
            nearby_selected = dict(raw_nearby_selected)
            nearby_selected["fraction"] = "0.050000000000000001"
            nearby_metrics["selected_confidence_bucket"] = nearby_selected
            persisted_evidence.aggregate_metrics = nearby_metrics
            await session.commit()
            nearby_audit = await assemble_family_audit(
                session, campaign_id=record.id, family_id=family_record.id
            )
            assert nearby_audit is not None
            assert "selected_confidence_bucket_mismatch" in (
                nearby_audit.evidence_integrity.failure_codes
            )
            nearby_selected["fraction"] = "0.050000000000000000"
            nearby_metrics["selected_confidence_bucket"] = nearby_selected
            persisted_evidence.aggregate_metrics = nearby_metrics
            await session.commit()
            first = await enqueue_manifest_reconstruction(
                session, campaign_id=record.id, family_id=family_record.id, now=NOW
            )
            second = await enqueue_manifest_reconstruction(
                session, campaign_id=record.id, family_id=family_record.id, now=NOW
            )
            assert first.id == second.id
            assert first.status == "queued"
            assert (
                await session.scalar(
                    select(func.count(MLTemporalManifestRecord.id)).where(
                        MLTemporalManifestRecord.evaluation_id == evidence.id
                    )
                )
                == 1
            )
            first.status = "completed"
            first.manifest_fingerprint = "m" * 64
            first.manifest = {}
            first.fingerprint_verified = True
            first.counts_verified = True
            first.completed_at = NOW
            await session.commit()

        async with sessions() as session:
            persisted_evidence = await session.get(
                MLRefinementEvaluationRecord, evidence.id
            )
            assert persisted_evidence is not None
            invalid_metrics = dict(persisted_evidence.aggregate_metrics)
            raw_invalid_selected = invalid_metrics["selected_confidence_bucket"]
            assert isinstance(raw_invalid_selected, dict)
            invalid_selected = dict(raw_invalid_selected)
            invalid_selected["fraction"] = "0.15"
            invalid_metrics["selected_confidence_bucket"] = invalid_selected
            persisted_evidence.aggregate_metrics = invalid_metrics
            await session.commit()
            with pytest.raises(ProtectedWorkError) as invalid:
                await enqueue_final_validation(
                    session, campaign_id=record.id, family_id=family_record.id, now=NOW
                )
            assert invalid.value.code == "invalid_confidence_bucket"
            assert (
                await session.scalar(
                    select(func.count(MLFinalValidationRecord.id)).where(
                        MLFinalValidationRecord.evaluation_id == evidence.id
                    )
                )
                == 0
            )
            invalid_selected["fraction"] = "0.050000000000000000"
            invalid_metrics["selected_confidence_bucket"] = invalid_selected
            persisted_evidence.aggregate_metrics = invalid_metrics
            await session.commit()
            validation = await enqueue_final_validation(
                session, campaign_id=record.id, family_id=family_record.id, now=NOW
            )
            assert validation.status == "queued"
            assert validation.protected_data_read_at is None
            assert validation.frozen_inputs["confidence_bucket_fraction"] == "0.05"
            selected = validation.frozen_inputs["selected_confidence_bucket"]
            assert isinstance(selected, dict)
            assert selected["fraction"] == "0.05"
    finally:
        async with sessions() as session:
            await session.execute(
                delete(MLFinalValidationRecord).where(
                    MLFinalValidationRecord.evaluation_id == evidence.id
                )
            )
            await session.execute(
                delete(MLTemporalManifestRecord).where(
                    MLTemporalManifestRecord.evaluation_id == evidence.id
                )
            )
            await session.commit()


async def test_other_evidence_failure_still_blocks_manifest_enqueue(
    factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    sessions, source_id, result_id = factory
    record = campaign(status="completed", covered=True, maximum_evaluations=1)
    family_record = family(record.id, source_id, state="strong_development_candidate")
    family_record.current_stage = 4
    run = refinement_run(source_id, result_id, status="completed")
    run.evaluations_computed = 1
    run.evaluations_selected = 1
    evidence = evaluation(run.id)
    evidence.confidence_buckets = ["0.05", "0.10", "0.20"]
    selected = evidence.aggregate_metrics["selected_confidence_bucket"]
    assert isinstance(selected, dict)
    selected["fraction"] = "0.050000000000000000"
    family_record.best_evaluation_id = evidence.id
    link = child(record.id, family_record.id, run.id, reused=False)
    link.stage = 4
    async with sessions() as session:
        session.add(record)
        await session.flush()
        session.add(run)
        await session.flush()
        session.add(evidence)
        await session.flush()
        session.add(family_record)
        await session.flush()
        session.add(link)
        await session.commit()
        with pytest.raises(ProtectedWorkError) as raised:
            await enqueue_manifest_reconstruction(
                session, campaign_id=record.id, family_id=family_record.id, now=NOW
            )
        assert raised.value.code == "development_evidence_invalid"
        assert (
            await session.scalar(
                select(func.count(MLTemporalManifestRecord.id)).where(
                    MLTemporalManifestRecord.evaluation_id == evidence.id
                )
            )
            == 0
        )

