import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.campaign.models import (
    MLSearchCampaignChildRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
)
from app.ml.refinement.models import MLRefinementEvaluationRecord, MLRefinementRunRecord

from .audit import assemble_family_audit
from .contracts import (
    DEVELOPMENT_MANIFEST_RECONSTRUCTION_VERSION,
    FINAL_VALIDATION_POLICY_VERSION,
    FINAL_VALIDATION_SIMULATION_VERSION,
    TEMPORAL_MANIFEST_VERSION,
)
from .models import MLFinalValidationRecord, MLTemporalManifestRecord


class ProtectedWorkError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def _eligible_family(
    session: AsyncSession, campaign_id: UUID, family_id: UUID
) -> tuple[
    MLSearchCampaignRecord,
    MLSearchCampaignFamilyRecord,
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
]:
    campaign = await session.get(MLSearchCampaignRecord, campaign_id)
    family = await session.get(MLSearchCampaignFamilyRecord, family_id)
    if campaign is None or family is None or family.campaign_id != campaign_id:
        raise ProtectedWorkError("family_not_found", "campaign family was not found")
    if (
        campaign.status != "completed"
        or family.current_stage != 4
        or family.state != "strong_development_candidate"
    ):
        raise ProtectedWorkError(
            "family_not_eligible",
            "family is not a completed stage-4 strong development candidate",
        )
    if family.best_evaluation_id is None:
        raise ProtectedWorkError(
            "best_evaluation_missing", "family has no best evaluation"
        )
    evaluation = await session.get(
        MLRefinementEvaluationRecord, family.best_evaluation_id
    )
    if evaluation is None:
        raise ProtectedWorkError(
            "best_evaluation_missing", "best evaluation does not exist"
        )
    child = await session.scalar(
        select(MLSearchCampaignChildRecord).where(
            MLSearchCampaignChildRecord.campaign_id == campaign_id,
            MLSearchCampaignChildRecord.family_id == family_id,
            MLSearchCampaignChildRecord.refinement_run_id == evaluation.run_id,
        )
    )
    if child is None:
        raise ProtectedWorkError(
            "best_evaluation_child_mismatch",
            "best evaluation is not owned by a family child",
        )
    run = await session.get(MLRefinementRunRecord, evaluation.run_id)
    if run is None or run.status != "completed":
        raise ProtectedWorkError(
            "refinement_run_incomplete", "winning refinement run is not complete"
        )
    return campaign, family, evaluation, run


async def enqueue_manifest_reconstruction(
    session: AsyncSession,
    *,
    campaign_id: UUID,
    family_id: UUID,
    now: datetime | None = None,
) -> MLTemporalManifestRecord:
    _, _, evaluation, run = await _eligible_family(session, campaign_id, family_id)
    audit = await assemble_family_audit(
        session, campaign_id=campaign_id, family_id=family_id
    )
    if audit is None or not audit.evidence_integrity.valid:
        raise ProtectedWorkError(
            "development_evidence_invalid",
            "persisted non-temporal development evidence failed integrity checks",
        )
    existing = await session.scalar(
        select(MLTemporalManifestRecord).where(
            MLTemporalManifestRecord.evaluation_id == evaluation.id,
            MLTemporalManifestRecord.manifest_version == TEMPORAL_MANIFEST_VERSION,
            MLTemporalManifestRecord.dataset_fingerprint == run.dataset_fingerprint,
        )
    )
    if existing is not None and existing.status in {"queued", "running", "completed"}:
        return existing
    timestamp = now or datetime.now(UTC)
    if existing is not None:
        existing.status = "queued"
        existing.worker_id = None
        existing.claimed_at = None
        existing.heartbeat_at = None
        existing.failure_code = None
        existing.failure_message = None
        existing.completed_at = None
        await session.commit()
        return existing
    record = MLTemporalManifestRecord(
        id=uuid4(),
        evaluation_id=evaluation.id,
        source="reconstructed_from_contract",
        status="queued",
        manifest_version=TEMPORAL_MANIFEST_VERSION,
        split_policy_version=DEVELOPMENT_MANIFEST_RECONSTRUCTION_VERSION,
        dataset_fingerprint=run.dataset_fingerprint,
        created_at=timestamp,
    )
    session.add(record)
    await session.commit()
    return record


async def enqueue_final_validation(
    session: AsyncSession,
    *,
    campaign_id: UUID,
    family_id: UUID,
    now: datetime | None = None,
) -> MLFinalValidationRecord:
    _, family, evaluation, run = await _eligible_family(session, campaign_id, family_id)
    if evaluation.objective != "future_direction_classification":
        raise ProtectedWorkError(
            "unsupported_objective",
            "only future-direction classification is deployable",
        )
    manifest = await session.scalar(
        select(MLTemporalManifestRecord).where(
            MLTemporalManifestRecord.evaluation_id == evaluation.id,
            MLTemporalManifestRecord.status == "completed",
            MLTemporalManifestRecord.fingerprint_verified.is_(True),
            MLTemporalManifestRecord.counts_verified.is_(True),
        )
    )
    if manifest is None or manifest.manifest_fingerprint is None:
        raise ProtectedWorkError(
            "development_manifest_not_ready", "a valid development manifest is required"
        )
    audit = await assemble_family_audit(
        session, campaign_id=campaign_id, family_id=family_id
    )
    if audit is None or not audit.overall_valid:
        raise ProtectedWorkError(
            "development_manifest_not_ready",
            "development evidence or temporal provenance is invalid",
        )
    frozen: dict[str, object] = {
        "campaign_id": str(campaign_id),
        "family_id": str(family_id),
        "family_fingerprint": family.fingerprint,
        "instrument": {
            "asset": family.asset,
            "provider": family.provider,
            "market_type": family.market_type,
            "interval": family.interval,
        },
        "evaluation_id": str(evaluation.id),
        "evaluation_fingerprint": evaluation.fingerprint,
        "dataset_fingerprint": run.dataset_fingerprint,
        "manifest_id": str(manifest.id),
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "engine": evaluation.engine,
        "hyperparameters": evaluation.hyperparameters,
        "feature_policy": evaluation.feature_policy,
        "objective": evaluation.objective,
        "future_duration_seconds": evaluation.future_duration_seconds,
        "selected_confidence_bucket": evaluation.aggregate_metrics.get(
            "selected_confidence_bucket"
        ),
        "normalized_configuration": run.normalized_configuration,
        "library_versions": run.library_versions,
        "policy_version": FINAL_VALIDATION_POLICY_VERSION,
        "simulation_version": FINAL_VALIDATION_SIMULATION_VERSION,
    }
    key = _sha(frozen)
    existing = await session.scalar(
        select(MLFinalValidationRecord).where(
            MLFinalValidationRecord.experiment_key == key
        )
    )
    if existing is not None:
        return existing
    record = MLFinalValidationRecord(
        id=uuid4(),
        experiment_key=key,
        campaign_id=campaign_id,
        family_id=family_id,
        evaluation_id=evaluation.id,
        temporal_manifest_id=manifest.id,
        status="queued",
        frozen_inputs=frozen,
        created_at=now or datetime.now(UTC),
    )
    session.add(record)
    await session.commit()
    return record


async def claim_next[WorkRecord: (MLTemporalManifestRecord, MLFinalValidationRecord)](
    session: AsyncSession,
    model: type[WorkRecord],
    *,
    worker_id: str,
    now: datetime | None = None,
) -> WorkRecord | None:
    timestamp = now or datetime.now(UTC)
    row = cast(
        WorkRecord | None,
        await session.scalar(
            select(model)
            .where(model.status == "queued")
            .order_by(model.created_at, model.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        ),
    )
    if row is None:
        return None
    row.status = "running"
    row.worker_id = worker_id
    row.claimed_at = timestamp
    row.heartbeat_at = timestamp
    await session.commit()
    return row


async def retry_final_validation(
    session: AsyncSession, validation_id: UUID
) -> MLFinalValidationRecord:
    record = await session.get(
        MLFinalValidationRecord, validation_id, with_for_update=True
    )
    if record is None:
        raise ProtectedWorkError(
            "final_validation_not_found", "final validation was not found"
        )
    if record.protected_data_read_at is not None:
        raise ProtectedWorkError(
            "protected_evidence_already_consumed",
            "protected evidence has already been consumed",
        )
    if record.status not in {"failed", "cancelled"}:
        return record
    record.status = "queued"
    record.worker_id = None
    record.claimed_at = None
    record.heartbeat_at = None
    record.failure_code = None
    record.failure_message = None
    record.completed_at = None
    await session.commit()
    return record


async def cancel_work[WorkRecord: (MLTemporalManifestRecord, MLFinalValidationRecord)](
    session: AsyncSession, model: type[WorkRecord], work_id: UUID
) -> WorkRecord:
    record = cast(
        WorkRecord | None,
        await session.scalar(
            select(model).where(model.id == work_id).with_for_update()
        ),
    )
    if record is None:
        raise ProtectedWorkError("protected_work_not_found", "work item was not found")
    if isinstance(record, MLFinalValidationRecord) and record.protected_data_read_at:
        raise ProtectedWorkError(
            "protected_evidence_already_consumed",
            "protected evidence has already been consumed",
        )
    if record.status == "queued":
        record.status = "cancelled"
        record.completed_at = datetime.now(UTC)
    elif record.status == "running":
        record.cancellation_requested_at = datetime.now(UTC)
    await session.commit()
    return record


async def retry_manifest(
    session: AsyncSession, manifest_id: UUID
) -> MLTemporalManifestRecord:
    record = await session.get(
        MLTemporalManifestRecord, manifest_id, with_for_update=True
    )
    if record is None:
        raise ProtectedWorkError(
            "development_manifest_not_found", "development manifest was not found"
        )
    if record.status not in {"failed", "cancelled"}:
        return record
    record.status = "queued"
    record.worker_id = None
    record.claimed_at = None
    record.heartbeat_at = None
    record.failure_code = None
    record.failure_message = None
    record.completed_at = None
    await session.commit()
    return record


async def fail_stale_protected_work(
    session: AsyncSession, *, stale_after: timedelta, now: datetime | None = None
) -> int:
    timestamp = now or datetime.now(UTC)
    cutoff = timestamp - stale_after
    count = await _fail_stale_model(session, MLTemporalManifestRecord, cutoff=cutoff)
    count += await _fail_stale_model(session, MLFinalValidationRecord, cutoff=cutoff)
    if count:
        await session.commit()
    return count


async def _fail_stale_model[
    WorkRecord: (MLTemporalManifestRecord, MLFinalValidationRecord)
](session: AsyncSession, model: type[WorkRecord], *, cutoff: datetime) -> int:
    rows = tuple(
        (
            await session.scalars(
                select(model)
                .where(model.status == "running", model.heartbeat_at < cutoff)
                .with_for_update(skip_locked=True)
            )
        ).all()
    )
    for record in rows:
        record.status = "failed"
        record.failure_code = "stale_worker_lease"
        record.failure_message = "worker lease expired"
        record.worker_id = None
        record.claimed_at = None
        record.heartbeat_at = None
    return len(rows)
