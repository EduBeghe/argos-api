import math
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.campaign.models import (
    MLSearchCampaignChildRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
)
from app.ml.contract import temporal_fold_definitions
from app.ml.refinement.models import MLRefinementEvaluationRecord, MLRefinementRunRecord

from .models import MLTemporalManifestRecord
from .schemas import CampaignFamilyAuditResponse, IntegritySection


def _finite(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, Decimal)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


async def assemble_family_audit(
    session: AsyncSession, *, campaign_id: UUID, family_id: UUID
) -> CampaignFamilyAuditResponse | None:
    campaign = await session.get(MLSearchCampaignRecord, campaign_id)
    family = await session.get(MLSearchCampaignFamilyRecord, family_id)
    if campaign is None or family is None or family.campaign_id != campaign_id:
        return None
    children = tuple(
        (
            await session.scalars(
                select(MLSearchCampaignChildRecord)
                .where(MLSearchCampaignChildRecord.family_id == family_id)
                .order_by(MLSearchCampaignChildRecord.stage)
            )
        ).all()
    )
    evaluation = (
        await session.get(MLRefinementEvaluationRecord, family.best_evaluation_id)
        if family.best_evaluation_id
        else None
    )
    owner = next(
        (
            item
            for item in children
            if evaluation and item.refinement_run_id == evaluation.run_id
        ),
        None,
    )
    run = (
        await session.get(MLRefinementRunRecord, evaluation.run_id)
        if evaluation
        else None
    )
    evidence_failures: list[str] = []
    if evaluation is None:
        evidence_failures.append("best_evaluation_missing")
    if owner is None:
        evidence_failures.append("best_evaluation_child_mismatch")
    folds = evaluation.fold_evidence if evaluation else []
    if len(folds) != 7:
        evidence_failures.append("incomplete_fold_count")
    expected_fold_names = tuple(item.name for item in temporal_fold_definitions())
    actual_fold_names = tuple(str(item.get("fold_name")) for item in folds)
    if actual_fold_names != expected_fold_names:
        evidence_failures.append("fold_name_or_order_mismatch")
    if (evaluation and not _finite(evaluation.fold_evidence)) or (
        evaluation and not _finite(evaluation.aggregate_metrics)
    ):
        evidence_failures.append("non_finite_metrics")
    if run and run.dataset_fingerprint != family.dataset_fingerprint:
        evidence_failures.append("dataset_identity_mismatch")
    manifest = (
        await session.scalar(
            select(MLTemporalManifestRecord)
            .where(MLTemporalManifestRecord.evaluation_id == evaluation.id)
            .order_by(MLTemporalManifestRecord.created_at.desc())
            .limit(1)
        )
        if evaluation
        else None
    )
    temporal_failures: list[str] = []
    if manifest is None:
        temporal_failures.append("development_manifest_missing")
    elif manifest.status != "completed":
        temporal_failures.append(f"development_manifest_{manifest.status}")
    elif not manifest.fingerprint_verified:
        temporal_failures.append("canonical_fingerprint_mismatch")
    elif not manifest.counts_verified:
        temporal_failures.append("fold_count_mismatch")
    selected = (
        evaluation.aggregate_metrics.get("selected_confidence_bucket")
        if evaluation
        else None
    )
    if evaluation and (
        not isinstance(selected, dict)
        or selected.get("fraction") is None
        or str(selected.get("fraction")) not in evaluation.confidence_buckets
    ):
        evidence_failures.append("selected_confidence_bucket_mismatch")
    diagnostic = None
    if evaluation:
        nominal = sum(_integer(item.get("validation_row_count")) for item in folds)
        stride = max(1, evaluation.future_duration_seconds // 300)
        upper = nominal // stride
        diagnostic = {
            "horizon_five_minute_intervals": stride,
            "nominal_observations": nominal,
            "non_overlapping_observation_upper_bound": upper,
            "upper_bound_ratio": str(Decimal(upper) / Decimal(nominal))
            if nominal
            else None,
        }
    evidence_integrity = IntegritySection(
        valid=not evidence_failures, failure_codes=tuple(evidence_failures)
    )
    temporal_integrity = IntegritySection(
        valid=not temporal_failures,
        failure_codes=tuple(temporal_failures),
        provenance_strength=(
            "exact_persisted"
            if manifest and manifest.source == "persisted_at_execution"
            else "deterministic_reconstruction"
            if manifest
            else None
        ),
    )
    return CampaignFamilyAuditResponse(
        campaign_id=campaign.id,
        family_id=family.id,
        family_fingerprint=family.fingerprint,
        dataset_fingerprint=family.dataset_fingerprint,
        deterministic_ordinal=family.deterministic_ordinal,
        state=family.state,
        stage=family.current_stage,
        best_evaluation_id=family.best_evaluation_id,
        best_evaluation_child_id=owner.id if owner else None,
        campaign_identity={
            "experiment_key": campaign.experiment_key,
            "status": campaign.status,
            "outcome": campaign.outcome,
            "contract_version": campaign.contract_version,
            "policy_version": campaign.policy_version,
            "classification_version": campaign.classification_version,
            "normalized_request": campaign.normalized_request,
            "maximum_evaluations": campaign.maximum_evaluations,
            "maximum_runtime_seconds": campaign.maximum_runtime_seconds,
        },
        family_identity={
            "source_screening_run_id": str(family.source_screening_run_id),
            "instrument": {
                "asset": family.asset,
                "provider": family.provider,
                "market_type": family.market_type,
                "interval": family.interval,
            },
            "engine": family.engine,
            "future_duration_seconds": family.future_duration_seconds,
            "objective": family.objective,
            "feature_variant": family.feature_variant,
            "best_metrics": family.best_metrics,
            "decision_reason": family.decision_reason,
        },
        source_identity={
            "screening_run_id": str(family.source_screening_run_id),
            "refinement_run_id": str(run.id) if run else None,
            "source_screening_result_id": str(run.source_screening_result_id)
            if run
            else None,
            "dataset_fingerprint": run.dataset_fingerprint if run else None,
            "contract_version": run.contract_version if run else None,
            "classification_version": run.classification_version if run else None,
            "variable_registry_version": run.variable_registry_version if run else None,
            "missing_data_policy_version": run.missing_data_policy_version
            if run
            else None,
            "library_versions": run.library_versions if run else None,
            "normalized_configuration": run.normalized_configuration if run else None,
        },
        child_stages=tuple(
            {
                "id": str(child.id),
                "stage": child.stage,
                "refinement_run_id": str(child.refinement_run_id),
                "reserved_evaluations": child.reserved_evaluations,
                "reuses_existing_run": child.reuses_existing_run,
            }
            for child in children
        ),
        winning_evaluation={
            "id": str(evaluation.id),
            "run_id": str(evaluation.run_id),
            "fingerprint": evaluation.fingerprint,
            "engine": evaluation.engine,
            "future_duration_seconds": evaluation.future_duration_seconds,
            "objective": evaluation.objective,
            "hyperparameters": evaluation.hyperparameters,
            "feature_policy": evaluation.feature_policy,
            "confidence_buckets": evaluation.confidence_buckets,
            "aggregate_metrics": evaluation.aggregate_metrics,
            "classification": evaluation.classification,
            "classification_version": evaluation.classification_version,
            "failed_rules": evaluation.failed_rules,
            "elapsed_seconds": str(evaluation.elapsed_seconds),
        }
        if evaluation
        else None,
        winning_configuration={
            "hyperparameters": evaluation.hyperparameters,
            "feature_policy": evaluation.feature_policy,
            "objective": evaluation.objective,
            "selected_confidence_bucket": selected,
        }
        if evaluation
        else None,
        fold_evidence=folds,
        temporal_manifest=manifest.manifest
        if manifest and manifest.status == "completed"
        else None,
        reconstructed_manifest_status=manifest.status if manifest else "missing",
        reconstructed_manifest_id=manifest.id if manifest else None,
        reconstructed_manifest_failure={
            "code": manifest.failure_code or "manifest_failed",
            "message": manifest.failure_message or "development manifest failed",
        }
        if manifest and manifest.status in {"failed", "cancelled"}
        else None,
        temporal_provenance={
            "source": manifest.source,
            "manifest_version": manifest.manifest_version,
            "split_policy_version": manifest.split_policy_version,
            "manifest_fingerprint": manifest.manifest_fingerprint,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "reconstructed_at": manifest.completed_at,
            "source_evaluation_id": str(manifest.evaluation_id),
        }
        if manifest and manifest.status == "completed"
        else None,
        evidence_integrity=evidence_integrity,
        temporal_provenance_integrity=temporal_integrity,
        overall_valid=evidence_integrity.valid and temporal_integrity.valid,
        independent_sample_diagnostic=diagnostic,
    )
