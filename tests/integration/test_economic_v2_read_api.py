from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ml.campaign.economic_v2_queue import (
    claim_next_qualification,
    fail_stale_qualifications,
    heartbeat_qualification,
)
from app.ml.campaign.models import (
    MLEconomicQualificationCampaignRecord,
    MLEconomicQualificationRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
    MLSearchCampaignSourceRecord,
)
from app.ml.screening.models import MLModelScreeningRunRecord

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
NOW = datetime(2042, 1, 1, tzinfo=UTC)
PREFIX = "economic-v2-api-"


@pytest_asyncio.fixture
async def economic_v2_records() -> AsyncIterator[dict[str, object]]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    source_id = uuid4()
    campaign_id = uuid4()
    family_ids = (uuid4(), uuid4())
    qualification_ids = (uuid4(), uuid4())
    async with factory() as session:
        session.add(
            MLModelScreeningRunRecord(
                id=source_id,
                experiment_key=(PREFIX + "source").ljust(64, "s")[:64],
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
                requested_engines=["ridge", "lightgbm", "catboost"],
                requested_future_durations=[3600],
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
        campaign = MLSearchCampaignRecord(
            id=campaign_id,
            experiment_key=(PREFIX + "campaign").ljust(64, "c")[:64],
            preset="economic_policy_search_v2",
            status="completed",
            outcome="qualified_candidates_found",
            contract_version=2,
            policy_version=2,
            classification_version=2,
            normalized_request={},
            maximum_evaluations=432,
            maximum_runtime_seconds=60,
            families_total=2,
            families_covered=2,
            families_uncovered=0,
            evaluations_computed=2,
            evaluations_reused=0,
            current_stage=5,
            cancellation_requested_at=None,
            first_claimed_at=NOW,
            completed_at=NOW,
            failed_at=None,
            failure_type=None,
            failure_message=None,
            created_at=NOW,
        )
        session.add(campaign)
        await session.flush()
        session.add(
            MLSearchCampaignSourceRecord(
                campaign_id=campaign_id,
                source_screening_run_id=source_id,
                dataset_fingerprint="d" * 64,
            )
        )
        for index, family_id in enumerate(family_ids):
            session.add(
                MLSearchCampaignFamilyRecord(
                    id=family_id,
                    campaign_id=campaign_id,
                    source_screening_run_id=source_id,
                    fingerprint=f"{index + 10:064x}",
                    preset="economic_policy_search_v2",
                    search_track="diversified" if index == 0 else "seeded_local",
                    seed_family_id=None,
                    seed_provenance=[],
                    dataset_fingerprint="d" * 64,
                    asset="BTC/USDT",
                    provider="binance",
                    market_type="spot",
                    interval="5m",
                    engine="ridge" if index == 0 else "lightgbm",
                    future_duration_seconds=3600,
                    objective="future_return_regression",
                    feature_variant="all",
                    state="strong_development_candidate",
                    current_stage=4,
                    evaluations_planned=64,
                    evaluations_completed=64,
                    evaluations_reused=0,
                    best_evaluation_id=None,
                    best_metrics={},
                    selected_policy={"confidence_fraction": "0.10"},
                    selected_policy_fingerprint=f"{index + 20:064x}",
                    decision_reason="economic_v2_development_gate_passed",
                    deterministic_ordinal=index,
                    created_at=NOW,
                )
            )
        await session.flush()
        for index, qualification_id in enumerate(qualification_ids):
            qualification = MLEconomicQualificationRecord(
                id=qualification_id,
                campaign_id=campaign_id,
                family_id=family_ids[index],
                candidate_fingerprint=f"{index + 30:064x}",
                policy_fingerprint=f"{index + 20:064x}",
                qualification_version=1,
                provenance=(
                    "automatic_development_candidate"
                    if index == 0
                    else "diagnostic_stage3_near_miss"
                ),
                status="completed",
                classification="qualified" if index == 0 else "rejected",
                engine="ridge" if index == 0 else "lightgbm",
                objective="future_return_regression",
                search_track="diversified" if index == 0 else "seeded_local",
                seed_family_id=None,
                frozen_inputs={"evaluation_fingerprint": f"{index + 40:064x}"},
                temporal_manifest={"qualification_end_exclusive": "2025-01-01"},
                temporal_manifest_fingerprint=f"{index + 50:064x}",
                policy_identity={"confidence_fraction": "0.10"},
                scenario_metrics={"base": {"total_net_return": "0.050000000000000001"}},
                quarter_returns=["0.01", "0.02", "0.03", "-0.01"],
                trades=[],
                failed_rules=[] if index == 0 else ["minimum_base_return"],
                evidence_end_exclusive=datetime(2025, 1, 1, tzinfo=UTC),
                worker_id=None,
                claimed_at=None,
                heartbeat_at=None,
                cancellation_requested_at=None,
                completed_at=NOW,
                failed_at=None,
                failure_type=None,
                failure_message=None,
                created_at=NOW,
            )
            session.add(qualification)
            await session.flush()
            session.add(
                MLEconomicQualificationCampaignRecord(
                    campaign_id=campaign_id,
                    qualification_id=qualification_id,
                    family_id=family_ids[index],
                    created_at=NOW,
                )
            )
        await session.commit()
    yield {
        "campaign_id": campaign_id,
        "qualification_ids": qualification_ids,
        "sessions": factory,
    }
    async with factory() as session:
        await session.execute(
            delete(MLSearchCampaignRecord).where(
                MLSearchCampaignRecord.experiment_key.like(f"{PREFIX}%")
            )
        )
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.id == source_id
            )
        )
        await session.commit()
    await engine.dispose()


async def test_qualification_api_filters_paginates_and_preserves_evidence(
    client: TestClient, economic_v2_records: dict[str, object]
) -> None:
    campaign_id = economic_v2_records["campaign_id"]
    first = client.get(
        f"/optimization/ml-search-campaigns/{campaign_id}/qualifications",
        params={"limit": 1, "offset": 0},
    )
    assert first.status_code == 200
    assert first.json()["total"] == 2
    assert first.json()["has_more"] is True
    filtered = client.get(
        f"/optimization/ml-search-campaigns/{campaign_id}/qualifications",
        params={"classification": "qualified", "engine": "ridge"},
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    item = filtered.json()["items"][0]
    assert item["scenario_metrics"]["base"]["total_net_return"] == (
        "0.050000000000000001"
    )
    assert item["provenance"] == "automatic_development_candidate"
    diagnostic = client.get(
        f"/optimization/ml-search-campaigns/{campaign_id}/qualifications",
        params={"provenance": "diagnostic_stage3_near_miss"},
    )
    assert diagnostic.status_code == 200
    assert diagnostic.json()["total"] == 1
    assert diagnostic.json()["items"][0]["provenance"] == (
        "diagnostic_stage3_near_miss"
    )
    detail = client.get(
        f"/optimization/ml-search-campaigns/{campaign_id}/qualifications/{item['id']}"
    )
    assert detail.status_code == 200
    assert detail.json()["evidence_end_exclusive"].startswith("2025-01-01")


async def test_campaign_detail_exposes_track_counts(
    client: TestClient, economic_v2_records: dict[str, object]
) -> None:
    response = client.get(
        f"/optimization/ml-search-campaigns/{economic_v2_records['campaign_id']}"
    )
    assert response.status_code == 200
    assert response.json()["track_counts"] == {
        "diversified": {
            "planned": 1,
            "covered": 1,
            "survived": 1,
            "discarded": 0,
            "qualified": 1,
        },
        "seeded_local": {
            "planned": 1,
            "covered": 1,
            "survived": 1,
            "discarded": 0,
            "qualified": 0,
        },
    }


async def test_qualification_heartbeat_prevents_reclaim_and_lost_owner_cannot_renew(
    economic_v2_records: dict[str, object],
) -> None:
    sessions = economic_v2_records["sessions"]
    assert isinstance(sessions, async_sessionmaker)
    qualification_id = economic_v2_records["qualification_ids"][0]  # type: ignore[index]
    async with sessions() as session:
        await session.execute(
            update(MLEconomicQualificationRecord)
            .where(MLEconomicQualificationRecord.id == qualification_id)
            .values(
                status="queued",
                classification=None,
                scenario_metrics=None,
                quarter_returns=None,
                trades=None,
                failed_rules=None,
                completed_at=None,
                worker_id=None,
                claimed_at=None,
                heartbeat_at=None,
            )
        )
        await session.commit()
        claimed = await claim_next_qualification(session, worker_id="worker-a")
        assert claimed is not None and claimed.id == qualification_id
        assert await heartbeat_qualification(
            session, qualification_id=qualification_id, worker_id="worker-a"
        )
        assert (
            await fail_stale_qualifications(session, stale_after=timedelta(seconds=1))
            == 0
        )
        await session.execute(
            update(MLEconomicQualificationRecord)
            .where(MLEconomicQualificationRecord.id == qualification_id)
            .values(heartbeat_at=datetime(2000, 1, 1, tzinfo=UTC))
        )
        await session.commit()
        assert (
            await fail_stale_qualifications(session, stale_after=timedelta(seconds=1))
            == 1
        )
        assert not await heartbeat_qualification(
            session, qualification_id=qualification_id, worker_id="worker-a"
        )
        replacement = await claim_next_qualification(session, worker_id="worker-b")
        assert replacement is not None and replacement.id == qualification_id
        assert await heartbeat_qualification(
            session, qualification_id=qualification_id, worker_id="worker-b"
        )
