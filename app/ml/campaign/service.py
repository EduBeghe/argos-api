from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.contract import (
    FUTURE_DURATIONS,
    MISSING_DATA_POLICY_VERSION,
    ML_EXPERIMENT_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
)
from app.ml.protected.models import MLFinalValidationRecord, MLTemporalManifestRecord
from app.ml.refinement.contracts import MLRefinementRequest, PredictionObjective
from app.ml.refinement.models import MLRefinementEvaluationRecord, MLRefinementRunRecord
from app.ml.refinement.persistence import (
    completed_run_is_complete,
    incompatible_run_evaluations,
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
    CAMPAIGN_PRESET,
    ECONOMIC_POLICY_SEARCH_V2_PRESET,
    ECONOMIC_V2_CLASSIFICATION_VERSION,
    ECONOMIC_V2_CONTRACT_VERSION,
    ECONOMIC_V2_POLICY_VERSION,
    CampaignRequest,
    CampaignStatus,
    FeatureVariant,
    seeded_local_search_space,
)
from .economic_v2_planning import (
    EconomicV2Family,
    SeedFamily,
    build_economic_v2_universe,
)
from .models import (
    MLEconomicQualificationCampaignRecord,
    MLEconomicQualificationRecord,
    MLSearchCampaignChildRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
    MLSearchCampaignSourceRecord,
)
from .planning import CampaignFamily, SourceInstrument, build_family_universe
from .refinement_runs import reset_refinement_run

logger = logging.getLogger(__name__)


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


async def _validated_seeds(
    session: AsyncSession,
    request: CampaignRequest,
    sources: tuple[SourceInstrument, ...],
) -> tuple[SeedFamily, ...]:
    if request.preset != ECONOMIC_POLICY_SEARCH_V2_PRESET:
        return ()
    if not request.seed_family_ids:
        return ()
    rows = tuple(
        (
            await session.scalars(
                select(MLSearchCampaignFamilyRecord).where(
                    MLSearchCampaignFamilyRecord.id.in_(request.seed_family_ids)
                )
            )
        ).all()
    )
    if len(rows) != len(request.seed_family_ids):
        raise CampaignSourceError("one or more seed families do not exist")
    source_by_id = {item.source_run_id: item for item in sources}
    seeds: list[SeedFamily] = []
    for family in rows:
        campaign = await session.get(MLSearchCampaignRecord, family.campaign_id)
        evaluation = (
            await session.get(MLRefinementEvaluationRecord, family.best_evaluation_id)
            if family.best_evaluation_id is not None
            else None
        )
        run = (
            await session.get(MLRefinementRunRecord, evaluation.run_id)
            if evaluation is not None
            else None
        )
        source = source_by_id.get(str(family.source_screening_run_id))
        protected = (
            await session.scalar(
                select(MLFinalValidationRecord.id)
                .where(
                    MLFinalValidationRecord.evaluation_id == family.best_evaluation_id
                )
                .limit(1)
            )
            if family.best_evaluation_id is not None
            else None
        )
        run_complete = (
            await completed_run_is_complete(session, run) if run is not None else False
        )
        if (
            campaign is None
            or campaign.status != "completed"
            or campaign.families_uncovered != 0
            or campaign.families_covered != campaign.families_total
            or source is None
            or source.dataset_fingerprint != family.dataset_fingerprint
            or evaluation is None
            or run is None
            or run.status != "completed"
            or not run_complete
            or run.source_screening_run_id != family.source_screening_run_id
            or evaluation.run_id != run.id
            or evaluation.engine != family.engine
            or evaluation.objective != family.objective
            or evaluation.future_duration_seconds != family.future_duration_seconds
            or evaluation.feature_policy.get("variant") != family.feature_variant
            or protected is not None
        ):
            raise CampaignSourceError(
                f"seed family {family.id} lacks compatible owned development evidence"
            )
        hyperparameters = tuple(
            sorted(
                (str(name), value)
                for name, value in evaluation.hyperparameters.items()
                if isinstance(value, (bool, int, float))
            )
        )
        if len(hyperparameters) != len(evaluation.hyperparameters):
            raise CampaignSourceError(
                f"seed family {family.id} has non-canonical hyperparameters"
            )
        try:
            seeded_local_search_space(
                engine=ScreeningEngine(family.engine),
                hyperparameters=dict(hyperparameters),
            )
        except ValueError as error:
            raise CampaignSourceError(
                f"seed family {family.id} has non-canonical hyperparameters"
            ) from error
        seeds.append(
            SeedFamily(
                family.id,
                str(family.source_screening_run_id),
                ScreeningEngine(family.engine),
                family.future_duration_seconds,
                PredictionObjective(family.objective),
                FeatureVariant(family.feature_variant),
                evaluation.fingerprint,
                hyperparameters,
            )
        )
    return tuple(sorted(seeds, key=lambda item: item.id))


async def _campaign_universe(
    session: AsyncSession,
    request: CampaignRequest,
    sources: tuple[SourceInstrument, ...],
) -> tuple[CampaignFamily | EconomicV2Family, ...]:
    if request.preset == CAMPAIGN_PRESET:
        return build_family_universe(sources)
    seeds = await _validated_seeds(session, request, sources)
    universe = build_economic_v2_universe(sources, seeds)
    if request.maximum_evaluations < len(universe):
        raise CampaignSourceError(
            "maximum_evaluations cannot cover stage 1; "
            f"exact minimum is {len(universe)}"
        )
    tracks = Counter(item.search_track for item in universe)
    objectives = Counter(item.objective.value for item in universe)
    logger.info(
        "Economic v2 campaign planned",
        extra={
            "structured_fields": {
                "preset": request.preset,
                "families": len(universe),
                "by_track": dict(sorted(tracks.items())),
                "by_objective": dict(sorted(objectives.items())),
                "source_count": len(sources),
            }
        },
    )
    return universe


async def preview_campaign(
    session: AsyncSession, request: CampaignRequest
) -> tuple[tuple[SourceInstrument, ...], int, str]:
    sources = await _validated_sources(session, request)
    universe = await _campaign_universe(session, request, sources)
    return (
        sources,
        len(universe),
        _experiment_key(request, sources),
    )


async def enqueue_campaign(
    session: AsyncSession, request: CampaignRequest, *, now: datetime | None = None
) -> CampaignEnqueueResult:
    sources = await _validated_sources(session, request)
    universe = await _campaign_universe(session, request, sources)
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
        preset=request.preset,
        status=CampaignStatus.QUEUED.value,
        outcome=None,
        contract_version=(
            CAMPAIGN_CONTRACT_VERSION
            if request.preset == CAMPAIGN_PRESET
            else ECONOMIC_V2_CONTRACT_VERSION
        ),
        policy_version=(
            CAMPAIGN_POLICY_VERSION
            if request.preset == CAMPAIGN_PRESET
            else ECONOMIC_V2_POLICY_VERSION
        ),
        classification_version=(
            CAMPAIGN_CLASSIFICATION_VERSION
            if request.preset == CAMPAIGN_PRESET
            else ECONOMIC_V2_CLASSIFICATION_VERSION
        ),
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
    try:
        await session.flush()
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
                    preset=request.preset,
                    search_track=(
                        item.search_track
                        if isinstance(item, EconomicV2Family)
                        else "exhaustive"
                    ),
                    seed_family_id=(
                        item.seed_family_id
                        if isinstance(item, EconomicV2Family)
                        else None
                    ),
                    seed_provenance=(
                        [str(seed_id) for seed_id in item.also_seeded_by]
                        if isinstance(item, EconomicV2Family)
                        else []
                    ),
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
                    selected_policy=None,
                    selected_policy_fingerprint=None,
                    decision_reason=None,
                    deterministic_ordinal=ordinal,
                    created_at=timestamp,
                )
                for ordinal, item in enumerate(universe)
            ]
        )
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
    if record.preset == ECONOMIC_POLICY_SEARCH_V2_PRESET and record.current_stage == 5:
        qualifications = tuple(
            (
                await session.scalars(
                    select(MLEconomicQualificationRecord)
                    .where(
                        MLEconomicQualificationRecord.id.in_(
                            select(
                                MLEconomicQualificationCampaignRecord.qualification_id
                            ).where(
                                MLEconomicQualificationCampaignRecord.campaign_id
                                == campaign_id
                            )
                        ),
                        MLEconomicQualificationRecord.status.in_(("queued", "running")),
                    )
                    .with_for_update()
                )
            ).all()
        )
        waiting = False
        for qualification in qualifications:
            shared = int(
                await session.scalar(
                    select(func.count())
                    .select_from(MLEconomicQualificationCampaignRecord)
                    .join(
                        MLSearchCampaignRecord,
                        MLSearchCampaignRecord.id
                        == MLEconomicQualificationCampaignRecord.campaign_id,
                    )
                    .where(
                        MLEconomicQualificationCampaignRecord.qualification_id
                        == qualification.id,
                        MLEconomicQualificationCampaignRecord.campaign_id
                        != campaign_id,
                        MLSearchCampaignRecord.status.in_(("queued", "running")),
                        MLSearchCampaignRecord.cancellation_requested_at.is_(None),
                    )
                )
                or 0
            )
            if shared:
                continue
            if qualification.status == "queued":
                qualification.status = "cancelled"
                qualification.completed_at = timestamp
            else:
                qualification.cancellation_requested_at = timestamp
                waiting = True
        if not waiting:
            record.status = "cancelled"
            record.completed_at = timestamp
        await session.commit()
        return record
    child = await session.scalar(
        select(MLSearchCampaignChildRecord)
        .where(MLSearchCampaignChildRecord.campaign_id == campaign_id)
        .order_by(MLSearchCampaignChildRecord.created_at.desc())
        .with_for_update()
        .limit(1)
    )
    run = None
    if child is not None:
        run = await session.scalar(
            select(MLRefinementRunRecord)
            .where(MLRefinementRunRecord.id == child.refinement_run_id)
            .with_for_update()
        )
    if run is None or run.status not in {"queued", "running"}:
        record.status = "cancelled"
        record.completed_at = timestamp
    else:
        other_dependents = int(
            await session.scalar(
                select(func.count())
                .select_from(MLSearchCampaignChildRecord)
                .join(
                    MLSearchCampaignRecord,
                    MLSearchCampaignRecord.id
                    == MLSearchCampaignChildRecord.campaign_id,
                )
                .where(
                    MLSearchCampaignChildRecord.refinement_run_id == run.id,
                    MLSearchCampaignChildRecord.campaign_id != campaign_id,
                    MLSearchCampaignRecord.status.in_(("queued", "running")),
                    MLSearchCampaignRecord.cancellation_requested_at.is_(None),
                )
            )
            or 0
        )
        if other_dependents:
            record.status = "cancelled"
            record.completed_at = timestamp
        elif run.status == "queued":
            run.status = "cancelled"
            run.completed_at = timestamp
            run.termination_reason = "cancelled_by_campaign"
            record.status = "cancelled"
            record.completed_at = timestamp
        elif run.cancellation_requested_at is None:
            run.cancellation_requested_at = timestamp
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
    if record.cancellation_requested_at is not None:
        raise CampaignConflictError(
            "campaign retry is unavailable after cancellation was requested"
        )
    child = await session.scalar(
        select(MLSearchCampaignChildRecord)
        .where(MLSearchCampaignChildRecord.campaign_id == campaign_id)
        .order_by(
            MLSearchCampaignChildRecord.created_at.desc(),
            MLSearchCampaignChildRecord.id.desc(),
        )
        .with_for_update()
        .limit(1)
    )
    if child is None:
        raise CampaignConflictError(
            "failed campaign has no retryable responsible child"
        )
    run = await session.scalar(
        select(MLRefinementRunRecord)
        .where(MLRefinementRunRecord.id == child.refinement_run_id)
        .with_for_update()
    )
    if run is None:
        raise CampaignConflictError(
            "failed campaign responsible child has no refinement run"
        )
    request_payload = dict(run.normalized_configuration)
    request_payload["notes"] = run.notes
    try:
        refinement_request = MLRefinementRequest.model_validate(request_payload)
    except (TypeError, ValueError) as error:
        raise CampaignConflictError(
            "failed campaign child configuration cannot be reconstructed"
        ) from error
    incompatible = await incompatible_run_evaluations(session, run, refinement_request)
    reset_required = run.status in {"failed", "cancelled"} or bool(incompatible)
    resume_shared = run.status in {"queued", "running"}
    if run.status == "completed":
        reset_required = not await completed_run_is_complete(
            session, run, refinement_request
        )
        resume_shared = not reset_required
    if not reset_required and not resume_shared:
        raise CampaignConflictError(
            "failed campaign responsible child is not retryable"
        )
    if reset_required:
        incompatible_ids = tuple(row.id for row in incompatible)
        if incompatible_ids:
            protected_reference = False
            for model, column in (
                (
                    MLSearchCampaignFamilyRecord,
                    MLSearchCampaignFamilyRecord.best_evaluation_id,
                ),
                (MLFinalValidationRecord, MLFinalValidationRecord.evaluation_id),
                (MLTemporalManifestRecord, MLTemporalManifestRecord.evaluation_id),
                (
                    MLRefinementEvaluationRecord,
                    MLRefinementEvaluationRecord.reused_from_evaluation_id,
                ),
            ):
                reference = await session.scalar(
                    select(model.id).where(column.in_(incompatible_ids)).limit(1)
                )
                if reference is not None:
                    protected_reference = True
                    break
            if protected_reference:
                raise CampaignConflictError(
                    "incompatible child evidence is referenced and cannot be repaired"
                )
            await session.execute(
                delete(MLRefinementEvaluationRecord).where(
                    MLRefinementEvaluationRecord.id.in_(incompatible_ids),
                    MLRefinementEvaluationRecord.run_id == run.id,
                )
            )
            logger.warning(
                "Removed incompatible campaign child evidence for retry",
                extra={
                    "structured_fields": {
                        "campaign_id": str(record.id),
                        "run_id": str(run.id),
                        "evaluation_ids": [str(item) for item in incompatible_ids],
                        "reason_code": "incompatible_economic_policy_evidence",
                    }
                },
            )
        try:
            reset_refinement_run(run)
        except (TypeError, ValueError) as error:
            raise CampaignConflictError(
                "failed campaign child configuration cannot be reconstructed"
            ) from error
    logger.info(
        "ML search campaign retry accepted",
        extra={
            "structured_fields": {
                "campaign_id": str(record.id),
                "refinement_run_id": str(run.id),
                "refinement_reset": reset_required,
            }
        },
    )
    record.status = "queued"
    record.failed_at = None
    record.failure_type = None
    record.failure_message = None
    await session.commit()
    await session.refresh(record)
    return record
