from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.contract import (
    FUTURE_DURATIONS,
    MISSING_DATA_POLICY_VERSION,
    ML_EXPERIMENT_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
)
from app.ml.screening.config import ML_MODEL_SCREENING_VERSION, ScreeningEngine
from app.ml.screening.models import (
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)

from .contracts import (
    CAMPAIGN_CLASSIFICATION_VERSION,
    CAMPAIGN_CONTRACT_VERSION,
    CAMPAIGN_POLICY_VERSION,
    CampaignRequest,
    CampaignStatus,
)
from .models import (
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
    MLSearchCampaignSourceRecord,
)
from .planning import SourceInstrument, build_family_universe


class CampaignNotFoundError(LookupError):
    pass


class CampaignConflictError(RuntimeError):
    pass


class CampaignSourceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CampaignEnqueueResult:
    campaign: MLSearchCampaignRecord
    created: bool


def _experiment_key(
    request: CampaignRequest, sources: tuple[SourceInstrument, ...]
) -> str:
    payload = request.normalized_payload()
    payload["source_dataset_fingerprints"] = sorted(
        (item.source_run_id, item.dataset_fingerprint) for item in sources
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def _validated_sources(
    session: AsyncSession, request: CampaignRequest
) -> tuple[SourceInstrument, ...]:
    rows = tuple(
        (
            await session.scalars(
                select(MLModelScreeningRunRecord).where(
                    MLModelScreeningRunRecord.id.in_(request.source_screening_run_ids)
                )
            )
        ).all()
    )
    if len(rows) != len(request.source_screening_run_ids):
        raise CampaignSourceError("one or more source screening runs do not exist")
    expected_pairs = len(ScreeningEngine) * len(FUTURE_DURATIONS)
    identities: set[tuple[str, str, str, str]] = set()
    sources: list[SourceInstrument] = []
    for row in rows:
        if row.status != "completed":
            raise CampaignSourceError(f"source screening run {row.id} is not completed")
        if (
            row.variable_registry_version != VARIABLE_REGISTRY_VERSION
            or row.missing_data_policy_version != MISSING_DATA_POLICY_VERSION
        ):
            raise CampaignSourceError(
                f"source screening run {row.id} has incompatible contract versions"
            )
        if (
            row.screening_version != ML_MODEL_SCREENING_VERSION
            or row.experiment_contract_version != ML_EXPERIMENT_CONTRACT_VERSION
        ):
            raise CampaignSourceError(
                f"source screening run {row.id} has incompatible screening versions"
            )
        expected_engines = {item.value for item in ScreeningEngine}
        expected_durations = {int(item.total_seconds()) for item in FUTURE_DURATIONS}
        if (
            set(row.requested_engines) != expected_engines
            or set(row.requested_future_durations) != expected_durations
        ):
            raise CampaignSourceError(
                f"source screening run {row.id} does not declare the complete "
                "engine-duration matrix"
            )
        result_count = int(
            await session.scalar(
                select(func.count())
                .select_from(MLModelScreeningResultRecord)
                .where(MLModelScreeningResultRecord.run_id == row.id)
            )
            or 0
        )
        if result_count != expected_pairs:
            raise CampaignSourceError(
                f"source screening run {row.id} has an incomplete "
                "engine-duration matrix"
            )
        identity = (row.asset, row.provider, row.market_type, row.interval)
        if identity in identities:
            raise CampaignSourceError("source instrument identities must be unique")
        identities.add(identity)
        sources.append(
            SourceInstrument(str(row.id), *identity, row.dataset_fingerprint)
        )
    return tuple(sources)


async def preview_campaign(
    session: AsyncSession, request: CampaignRequest
) -> tuple[tuple[SourceInstrument, ...], int, str]:
    sources = await _validated_sources(session, request)
    return (
        sources,
        len(build_family_universe(sources)),
        _experiment_key(request, sources),
    )


async def enqueue_campaign(
    session: AsyncSession, request: CampaignRequest, *, now: datetime | None = None
) -> CampaignEnqueueResult:
    sources = await _validated_sources(session, request)
    universe = build_family_universe(sources)
    key = _experiment_key(request, sources)
    existing = await session.scalar(
        select(MLSearchCampaignRecord).where(
            MLSearchCampaignRecord.experiment_key == key
        )
    )
    if existing is not None:
        return CampaignEnqueueResult(existing, False)
    timestamp = now or datetime.now(UTC)
    normalized_request = request.normalized_payload()
    normalized_request["source_dataset_fingerprints"] = sorted(
        (item.source_run_id, item.dataset_fingerprint) for item in sources
    )
    campaign = MLSearchCampaignRecord(
        id=uuid4(),
        experiment_key=key,
        status=CampaignStatus.QUEUED.value,
        outcome=None,
        contract_version=CAMPAIGN_CONTRACT_VERSION,
        policy_version=CAMPAIGN_POLICY_VERSION,
        classification_version=CAMPAIGN_CLASSIFICATION_VERSION,
        normalized_request=normalized_request,
        maximum_evaluations=request.maximum_evaluations,
        maximum_runtime_seconds=request.maximum_runtime_seconds,
        families_total=len(universe),
        families_covered=0,
        families_uncovered=len(universe),
        evaluations_computed=0,
        evaluations_reused=0,
        current_stage=1,
        cancellation_requested_at=None,
        first_claimed_at=None,
        completed_at=None,
        failed_at=None,
        failure_type=None,
        failure_message=None,
        created_at=timestamp,
    )
    session.add(campaign)
    for source in sources:
        session.add(
            MLSearchCampaignSourceRecord(
                campaign_id=campaign.id,
                source_screening_run_id=UUID(source.source_run_id),
                dataset_fingerprint=source.dataset_fingerprint,
            )
        )
    session.add_all(
        [
            MLSearchCampaignFamilyRecord(
                id=uuid4(),
                campaign_id=campaign.id,
                source_screening_run_id=UUID(item.source.source_run_id),
                fingerprint=item.fingerprint,
                dataset_fingerprint=item.source.dataset_fingerprint,
                asset=item.source.asset,
                provider=item.source.provider,
                market_type=item.source.market_type,
                interval=item.source.interval,
                engine=item.engine.value,
                future_duration_seconds=item.future_duration_seconds,
                objective=item.objective.value,
                feature_variant=item.feature_variant.value,
                state="pending",
                current_stage=0,
                evaluations_planned=0,
                evaluations_completed=0,
                evaluations_reused=0,
                best_evaluation_id=None,
                best_metrics=None,
                decision_reason=None,
                deterministic_ordinal=ordinal,
                created_at=timestamp,
            )
            for ordinal, item in enumerate(universe)
        ]
    )
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        winner = await session.scalar(
            select(MLSearchCampaignRecord).where(
                MLSearchCampaignRecord.experiment_key == key
            )
        )
        if winner is None:
            raise
        return CampaignEnqueueResult(winner, False)
    return CampaignEnqueueResult(campaign, True)


async def get_campaign(
    session: AsyncSession, campaign_id: UUID
) -> MLSearchCampaignRecord:
    record = await session.get(MLSearchCampaignRecord, campaign_id)
    if record is None:
        raise CampaignNotFoundError(f"ML search campaign {campaign_id} not found")
    return record


async def cancel_campaign(
    session: AsyncSession, campaign_id: UUID, *, now: datetime | None = None
) -> MLSearchCampaignRecord:
    record = await session.scalar(
        select(MLSearchCampaignRecord)
        .where(MLSearchCampaignRecord.id == campaign_id)
        .with_for_update()
    )
    if record is None:
        raise CampaignNotFoundError(f"ML search campaign {campaign_id} not found")
    if record.status in {"completed", "failed", "cancelled"}:
        raise CampaignConflictError(f"{record.status} campaign cannot be cancelled")
    timestamp = now or datetime.now(UTC)
    record.cancellation_requested_at = timestamp
    if record.status == "queued":
        record.status = "cancelled"
        record.completed_at = timestamp
    await session.commit()
    return record


async def retry_campaign(
    session: AsyncSession, campaign_id: UUID
) -> MLSearchCampaignRecord:
    record = await session.scalar(
        select(MLSearchCampaignRecord)
        .where(MLSearchCampaignRecord.id == campaign_id)
        .with_for_update()
    )
    if record is None:
        raise CampaignNotFoundError(f"ML search campaign {campaign_id} not found")
    if record.status != "failed":
        raise CampaignConflictError(
            "only operationally failed campaigns can be retried"
        )
    await session.execute(
        update(MLSearchCampaignRecord)
        .where(MLSearchCampaignRecord.id == campaign_id)
        .values(
            status="queued", failed_at=None, failure_type=None, failure_message=None
        )
    )
    await session.commit()
    await session.refresh(record)
    return record
