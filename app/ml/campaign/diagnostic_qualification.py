from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.protected.models import MLFinalValidationRecord
from app.ml.refinement.models import MLRefinementEvaluationRecord, MLRefinementRunRecord
from app.ml.refinement.persistence import completed_run_is_complete

from .contracts import (
    DIAGNOSTIC_QUALIFICATION_PROVENANCE,
    DIAGNOSTIC_QUALIFICATION_REQUEST_VERSION,
    ECONOMIC_POLICY_SEARCH_V2_PRESET,
    ECONOMIC_V2_QUALIFICATION_END,
    ECONOMIC_V2_QUALIFICATION_VERSION,
    DiagnosticQualificationRequest,
)
from .economic_v2_identity import qualification_identity, qualification_identity_payload
from .economic_v2_policy import strategy_policy_grid
from .models import (
    MLEconomicQualificationCampaignRecord,
    MLEconomicQualificationRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
)

logger = logging.getLogger(__name__)
_QUALIFICATION_END = datetime(2025, 1, 1, tzinfo=UTC)


class DiagnosticQualificationNotFoundError(LookupError):
    pass


class DiagnosticQualificationConflictError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class DiagnosticQualificationItemResult:
    family_id: UUID
    qualification_id: UUID
    status: str
    created: bool


@dataclass(frozen=True, slots=True)
class DiagnosticQualificationEnqueueResult:
    campaign_id: UUID
    items: tuple[DiagnosticQualificationItemResult, ...]


def _conflict(reason_code: str) -> DiagnosticQualificationConflictError:
    return DiagnosticQualificationConflictError(reason_code)


def _stage_three_only_benchmark_miss(metrics: dict[str, object]) -> bool:
    try:
        return (
            bool(metrics.get("complete_and_finite", False))
            and int(str(metrics["observations"])) >= 100
            and int(str(metrics["positive_net_folds"])) >= 4
            and int(str(metrics["folds_beating_benchmark"])) == 2
            and Decimal(str(metrics["median_net_return"])) > 0
            and Decimal(str(metrics["worst_net_return"])) >= Decimal("-0.05")
            and int(str(metrics["round_trips"])) >= 10
            and int(str(metrics["inactive_folds"])) <= 2
        )
    except (KeyError, TypeError, ValueError):
        return False


def _has_post_development_evidence(value: object, key: str = "") -> bool:
    """Reject persisted evaluation timestamps that enter qualification/final data."""

    if isinstance(value, dict):
        return any(
            _has_post_development_evidence(item, str(name))
            for name, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_post_development_evidence(item, key) for item in value)
    if isinstance(value, str) and any(
        token in key.lower() for token in ("time", "date", "start", "end", "boundary")
    ):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return parsed.astimezone(UTC) >= datetime(2024, 1, 1, tzinfo=UTC)
    if isinstance(value, (int, float)) and any(
        token in key.lower() for token in ("time", "date", "start", "end", "boundary")
    ):
        numeric = float(value)
        if numeric >= 1_000_000_000_000_000_000:
            numeric /= 1_000_000_000
        elif numeric >= 1_000_000_000_000:
            numeric /= 1_000
        if numeric >= 1_000_000_000:
            return datetime.fromtimestamp(numeric, tz=UTC) >= datetime(
                2024, 1, 1, tzinfo=UTC
            )
    return False


def _configuration_contains(
    configuration: dict[str, object], key: str, expected: object
) -> bool:
    values = configuration.get(key)
    return isinstance(values, list) and expected in values


async def _validated_evidence(
    session: AsyncSession, family: MLSearchCampaignFamilyRecord
) -> tuple[MLRefinementEvaluationRecord, MLRefinementRunRecord]:
    if (
        family.state != "discarded"
        or family.current_stage != 3
        or family.best_evaluation_id is None
        or family.selected_policy is None
        or family.selected_policy_fingerprint is None
        or not _stage_three_only_benchmark_miss(family.best_metrics or {})
    ):
        raise _conflict("family_not_eligible_stage3_near_miss")
    if (
        str(family.selected_policy.get("policy_fingerprint"))
        != family.selected_policy_fingerprint
    ):
        raise _conflict("family_policy_identity_mismatch")
    try:
        requested_policy = (
            family.selected_policy_fingerprint,
            str(family.selected_policy["confidence_fraction"]),
            str(family.selected_policy["holding_multiplier"]),
            str(family.selected_policy["regime_policy"]),
            int(str(family.selected_policy["holding_candles"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _conflict("family_policy_identity_mismatch") from error
    try:
        policy_matches = any(
            (
                item.fingerprint,
                str(item.confidence_fraction),
                str(item.holding_multiplier),
                item.regime_policy,
                item.holding_candles,
            )
            == requested_policy
            for item in strategy_policy_grid(family.future_duration_seconds)
        )
    except ValueError as error:
        raise _conflict("family_policy_identity_mismatch") from error
    if not policy_matches:
        raise _conflict("family_policy_identity_mismatch")
    evaluation = await session.get(
        MLRefinementEvaluationRecord, family.best_evaluation_id
    )
    run = (
        await session.get(MLRefinementRunRecord, evaluation.run_id)
        if evaluation is not None
        else None
    )
    protected = await session.scalar(
        select(MLFinalValidationRecord.id)
        .where(MLFinalValidationRecord.evaluation_id == family.best_evaluation_id)
        .limit(1)
    )
    if protected is not None:
        raise _conflict("family_has_protected_evidence")
    if (
        evaluation is None
        or run is None
        or run.status != "completed"
        or not await completed_run_is_complete(session, run)
        or evaluation.run_id != run.id
        or run.source_screening_run_id != family.source_screening_run_id
        or run.dataset_fingerprint != family.dataset_fingerprint
        or not _configuration_contains(
            run.normalized_configuration, "engines", family.engine
        )
        or not _configuration_contains(
            run.normalized_configuration,
            "future_durations_seconds",
            family.future_duration_seconds,
        )
        or not _configuration_contains(
            run.normalized_configuration, "prediction_objectives", family.objective
        )
        or evaluation.engine != family.engine
        or evaluation.objective != family.objective
        or evaluation.future_duration_seconds != family.future_duration_seconds
        or evaluation.feature_policy.get("variant") != family.feature_variant
        or _has_post_development_evidence(evaluation.fold_evidence)
    ):
        raise _conflict("family_refinement_evidence_incompatible")
    return evaluation, run


def _manifest() -> dict[str, object]:
    return {
        "qualification_start": "2024-01-01T00:00:00+00:00",
        "qualification_end_exclusive": ECONOMIC_V2_QUALIFICATION_END,
        "status": "not_read",
        "request_version": DIAGNOSTIC_QUALIFICATION_REQUEST_VERSION,
    }


async def enqueue_diagnostic_qualifications(
    session: AsyncSession,
    *,
    campaign_id: UUID,
    request: DiagnosticQualificationRequest,
    _race_retry: bool = False,
) -> DiagnosticQualificationEnqueueResult:
    """Atomically validate and enqueue explicitly selected 2024 diagnostics."""

    campaign = await session.scalar(
        select(MLSearchCampaignRecord)
        .where(MLSearchCampaignRecord.id == campaign_id)
        .with_for_update()
    )
    if campaign is None:
        await session.rollback()
        raise DiagnosticQualificationNotFoundError("campaign_not_found")
    if (
        campaign.preset != ECONOMIC_POLICY_SEARCH_V2_PRESET
        or campaign.status != "completed"
        or campaign.outcome != "no_strong_candidate"
        or campaign.families_uncovered != 0
        or campaign.families_covered != campaign.families_total
        or campaign.cancellation_requested_at is not None
    ):
        await session.rollback()
        raise _conflict("campaign_not_eligible_for_diagnostic_qualification")

    families = tuple(
        (
            await session.scalars(
                select(MLSearchCampaignFamilyRecord)
                .where(
                    MLSearchCampaignFamilyRecord.campaign_id == campaign_id,
                    MLSearchCampaignFamilyRecord.id.in_(request.canonical_family_ids()),
                )
                .order_by(MLSearchCampaignFamilyRecord.id)
                .with_for_update()
            )
        ).all()
    )
    if len(families) != len(request.family_ids):
        await session.rollback()
        raise DiagnosticQualificationNotFoundError("family_not_found")
    family_by_id = {family.id: family for family in families}
    validated: dict[
        UUID, tuple[MLRefinementEvaluationRecord, MLRefinementRunRecord]
    ] = {}
    try:
        for family in families:
            if family.preset != ECONOMIC_POLICY_SEARCH_V2_PRESET:
                raise _conflict("family_preset_incompatible")
            validated[family.id] = await _validated_evidence(session, family)

        now = datetime.now(UTC)
        results: list[DiagnosticQualificationItemResult] = []
        for family_id in request.family_ids:
            family = family_by_id[family_id]
            evaluation, run = validated[family_id]
            manifest = _manifest()
            manifest_fingerprint = qualification_identity(manifest)
            frozen = qualification_identity_payload(
                family=family,
                evaluation=evaluation,
                run=run,
                policy=family.selected_policy or {},
                provenance=DIAGNOSTIC_QUALIFICATION_PROVENANCE,
            )
            fingerprint = qualification_identity(frozen)
            record = await session.scalar(
                select(MLEconomicQualificationRecord).where(
                    MLEconomicQualificationRecord.candidate_fingerprint == fingerprint,
                    MLEconomicQualificationRecord.qualification_version
                    == ECONOMIC_V2_QUALIFICATION_VERSION,
                )
            )
            created = record is None
            if record is None:
                record = MLEconomicQualificationRecord(
                    id=uuid4(),
                    campaign_id=campaign_id,
                    family_id=family.id,
                    candidate_fingerprint=fingerprint,
                    policy_fingerprint=family.selected_policy_fingerprint or "",
                    qualification_version=ECONOMIC_V2_QUALIFICATION_VERSION,
                    provenance=DIAGNOSTIC_QUALIFICATION_PROVENANCE,
                    status="queued",
                    classification=None,
                    engine=family.engine,
                    objective=family.objective,
                    search_track=family.search_track,
                    seed_family_id=family.seed_family_id,
                    frozen_inputs=frozen,
                    temporal_manifest=manifest,
                    temporal_manifest_fingerprint=manifest_fingerprint,
                    policy_identity=family.selected_policy or {},
                    scenario_metrics=None,
                    quarter_returns=None,
                    trades=None,
                    failed_rules=None,
                    evidence_end_exclusive=_QUALIFICATION_END,
                    worker_id=None,
                    claimed_at=None,
                    heartbeat_at=None,
                    cancellation_requested_at=None,
                    completed_at=None,
                    failed_at=None,
                    failure_type=None,
                    failure_message=None,
                    created_at=now,
                )
                session.add(record)
                await session.flush()
            elif (
                record.frozen_inputs != frozen
                or record.provenance != DIAGNOSTIC_QUALIFICATION_PROVENANCE
            ):
                raise _conflict("qualification_identity_mismatch")
            elif record.status in {"failed", "cancelled"}:
                raise _conflict(
                    f"qualification_{record.status}_requires_explicit_retry"
                )
            association = await session.get(
                MLEconomicQualificationCampaignRecord, (campaign_id, record.id)
            )
            if association is None:
                session.add(
                    MLEconomicQualificationCampaignRecord(
                        campaign_id=campaign_id,
                        qualification_id=record.id,
                        family_id=family.id,
                        created_at=now,
                    )
                )
            elif association.family_id != family.id:
                raise _conflict("qualification_association_mismatch")
            results.append(
                DiagnosticQualificationItemResult(
                    family_id=family.id,
                    qualification_id=record.id,
                    status=record.status,
                    created=created,
                )
            )
        await session.commit()
    except (DiagnosticQualificationConflictError, DiagnosticQualificationNotFoundError):
        await session.rollback()
        raise
    except IntegrityError:
        await session.rollback()
        if _race_retry:
            raise
        # A concurrent winner may now be reusable; retry against durable identities.
        return await enqueue_diagnostic_qualifications(
            session, campaign_id=campaign_id, request=request, _race_retry=True
        )
    except Exception:
        await session.rollback()
        raise
    logger.info(
        "Diagnostic qualification request accepted",
        extra={
            "structured_fields": {
                "campaign_id": str(campaign_id),
                "provenance": DIAGNOSTIC_QUALIFICATION_PROVENANCE,
                "status": "accepted",
                "reason_code": "created_or_reused",
                "evidence_end_exclusive": ECONOMIC_V2_QUALIFICATION_END,
                "qualification_count": len(results),
            }
        },
    )
    return DiagnosticQualificationEnqueueResult(campaign_id, tuple(results))
