from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ml.campaign import development_reaudit
from app.ml.campaign.contracts import CampaignRequest
from app.ml.campaign.development_reaudit import (
    build_development_reaudit_report,
    report_json,
)
from app.ml.campaign.models import (
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
    MLSearchCampaignSourceRecord,
)
from app.ml.screening.models import MLModelScreeningRunRecord

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
NOW = datetime(2040, 1, 1, tzinfo=UTC)
PREFIX = "development-reaudit-test-"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        yield database_session
        await database_session.rollback()
        campaign_ids = select(MLSearchCampaignRecord.id).where(
            MLSearchCampaignRecord.experiment_key.like(f"{PREFIX}%")
        )
        await database_session.execute(
            delete(MLSearchCampaignRecord).where(
                MLSearchCampaignRecord.id.in_(campaign_ids)
            )
        )
        await database_session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.experiment_key.like(f"{PREFIX}%")
            )
        )
        await database_session.commit()
    await engine.dispose()


async def seed_empty_cohort(session: AsyncSession) -> UUID:
    source_id = uuid4()
    campaign_id = uuid4()
    request = CampaignRequest(source_screening_run_ids=(source_id,))
    session.add(
        MLModelScreeningRunRecord(
            id=source_id,
            experiment_key=f"{PREFIX}{source_id}"[:64],
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
    session.add(
        MLSearchCampaignRecord(
            id=campaign_id,
            experiment_key=f"{PREFIX}{campaign_id}"[:64],
            status="completed",
            outcome="no_strong_candidate",
            contract_version=1,
            policy_version=1,
            classification_version=1,
            normalized_request=request.normalized_payload(),
            maximum_evaluations=request.maximum_evaluations,
            maximum_runtime_seconds=request.maximum_runtime_seconds,
            families_total=1,
            families_covered=1,
            families_uncovered=0,
            evaluations_computed=1,
            evaluations_reused=0,
            current_stage=4,
            cancellation_requested_at=None,
            first_claimed_at=NOW,
            completed_at=NOW,
            failed_at=None,
            failure_type=None,
            failure_message=None,
            created_at=NOW,
        )
    )
    session.add(
        MLSearchCampaignSourceRecord(
            campaign_id=campaign_id,
            source_screening_run_id=source_id,
            dataset_fingerprint="d" * 64,
        )
    )
    session.add(
        MLSearchCampaignFamilyRecord(
            id=uuid4(),
            campaign_id=campaign_id,
            source_screening_run_id=source_id,
            fingerprint="f" * 64,
            dataset_fingerprint="d" * 64,
            asset="BTC/USDT",
            provider="binance",
            market_type="spot",
            interval="5m",
            engine="ridge",
            future_duration_seconds=1800,
            objective="future_return_regression",
            feature_variant="all",
            state="discarded",
            current_stage=1,
            evaluations_planned=1,
            evaluations_completed=1,
            evaluations_reused=0,
            best_evaluation_id=None,
            best_metrics=None,
            decision_reason="failed_stage_1",
            deterministic_ordinal=1,
            created_at=NOW,
        )
    )
    await session.commit()
    return campaign_id


async def test_empty_cohort_is_deterministic_and_read_only(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_id = await seed_empty_cohort(session)
    monkeypatch.setattr(development_reaudit, "FAMILIES_PER_INSTRUMENT", 1)
    before = await session.scalar(
        select(func.count()).select_from(MLSearchCampaignRecord)
    )
    first, first_invalid = await build_development_reaudit_report(
        session=session, campaign_id=campaign_id
    )
    second, second_invalid = await build_development_reaudit_report(
        session=session, campaign_id=campaign_id
    )
    after = await session.scalar(
        select(func.count()).select_from(MLSearchCampaignRecord)
    )

    assert not first_invalid and not second_invalid
    assert first["counts"] == {
        "eligible": 0,
        "selected": 0,
        "omitted_by_cap": 0,
        "completed": 0,
        "invalid": 0,
        "proxy_passed": 0,
        "proxy_failed": 0,
        "malformed_unselected_stage_four": 0,
    }
    assert report_json(first) == report_json(second)
    assert before == after
    persisted = await session.get(MLSearchCampaignRecord, campaign_id)
    assert persisted is not None
    assert persisted.status == "completed"
    assert persisted.outcome == "no_strong_candidate"

