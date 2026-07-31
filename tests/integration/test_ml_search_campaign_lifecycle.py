from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ml.campaign.models import (
    MLSearchCampaignChildRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
)
from app.ml.campaign.service import (
    CampaignConflictError,
    cancel_campaign,
    retry_campaign,
)
from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.models import MLRefinementRunRecord
from app.ml.refinement.planning import build_refinement_plan
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


def campaign(*, status: str = "running") -> MLSearchCampaignRecord:
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
        maximum_evaluations=3600,
        maximum_runtime_seconds=60,
        families_total=1,
        families_covered=0,
        families_uncovered=1,
        evaluations_computed=0,
        evaluations_reused=0,
        current_stage=1,
        cancellation_requested_at=None,
        first_claimed_at=NOW,
        completed_at=None,
        failed_at=NOW if status == "failed" else None,
        failure_type="ChildRefinementFailed" if status == "failed" else None,
        failure_message="child failed" if status == "failed" else None,
        created_at=NOW,
    )


def family(campaign_id: UUID, source_id: UUID) -> MLSearchCampaignFamilyRecord:
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
        state="scheduled",
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
        session.add_all((first, second, first_family, second_family, run))
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
        session.add_all((record, family_record, run, link))
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
