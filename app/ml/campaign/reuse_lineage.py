from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.contract import DEVELOPMENT_END
from app.ml.protected.contracts import (
    DEVELOPMENT_MANIFEST_RECONSTRUCTION_VERSION,
    TEMPORAL_MANIFEST_VERSION,
)
from app.ml.protected.models import MLTemporalManifestRecord
from app.ml.protected.temporal import (
    TemporalManifestError,
    validate_persisted_development_manifest,
)
from app.ml.refinement.models import MLRefinementEvaluationRecord, MLRefinementRunRecord
from app.ml.refinement.persistence import completed_run_is_complete

MAXIMUM_REUSE_LINEAGE_EVALUATIONS = 64


class ReuseLineageError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class ResolvedDevelopmentEvidence:
    selected_evaluation: MLRefinementEvaluationRecord
    selected_run: MLRefinementRunRecord
    evaluation_ids: tuple[UUID, ...]
    manifest_owner_evaluation_id: UUID
    manifest: MLTemporalManifestRecord


def _scientifically_identical(
    selected: MLRefinementEvaluationRecord,
    candidate: MLRefinementEvaluationRecord,
) -> bool:
    fields = (
        "fingerprint",
        "engine",
        "objective",
        "future_duration_seconds",
        "hyperparameters",
        "feature_policy",
        "confidence_buckets",
        "fold_evidence",
        "aggregate_metrics",
        "classification",
        "classification_version",
        "failed_rules",
    )
    return all(
        getattr(selected, field) == getattr(candidate, field) for field in fields
    )


async def _compatible_manifests(
    session: AsyncSession,
    *,
    evaluation_id: UUID,
    dataset_fingerprint: str,
) -> tuple[MLTemporalManifestRecord, ...]:
    rows = tuple(
        (
            await session.scalars(
                select(MLTemporalManifestRecord)
                .where(
                    MLTemporalManifestRecord.evaluation_id == evaluation_id,
                    MLTemporalManifestRecord.manifest_version
                    == TEMPORAL_MANIFEST_VERSION,
                    MLTemporalManifestRecord.split_policy_version
                    == DEVELOPMENT_MANIFEST_RECONSTRUCTION_VERSION,
                    MLTemporalManifestRecord.status == "completed",
                    MLTemporalManifestRecord.fingerprint_verified.is_(True),
                    MLTemporalManifestRecord.counts_verified.is_(True),
                    MLTemporalManifestRecord.manifest.is_not(None),
                    MLTemporalManifestRecord.manifest_fingerprint.is_not(None),
                )
                .order_by(MLTemporalManifestRecord.id)
            )
        ).all()
    )
    compatible: list[MLTemporalManifestRecord] = []
    for row in rows:
        if (
            row.dataset_fingerprint != dataset_fingerprint
            or row.manifest is None
            or row.manifest_fingerprint is None
        ):
            continue
        try:
            validate_persisted_development_manifest(
                payload=row.manifest,
                persisted_fingerprint=row.manifest_fingerprint,
                canonical_dataset_fingerprint=dataset_fingerprint,
                development_end_exclusive=DEVELOPMENT_END,
            )
        except TemporalManifestError:
            continue
        compatible.append(row)
    return tuple(compatible)


async def resolve_development_evidence_lineage(
    session: AsyncSession,
    *,
    selected_evaluation: MLRefinementEvaluationRecord,
    selected_run: MLRefinementRunRecord,
    dataset_fingerprint: str,
) -> ResolvedDevelopmentEvidence:
    """Resolve one immutable development manifest through a bounded reuse chain."""

    if (
        selected_evaluation.run_id != selected_run.id
        or selected_run.status != "completed"
        or selected_run.dataset_fingerprint != dataset_fingerprint
        or not await completed_run_is_complete(session, selected_run)
    ):
        raise ReuseLineageError("family_reuse_lineage_invalid")
    lineage: list[UUID] = []
    seen: set[UUID] = set()
    manifests: list[tuple[int, UUID, MLTemporalManifestRecord]] = []
    current = selected_evaluation
    for depth in range(MAXIMUM_REUSE_LINEAGE_EVALUATIONS):
        if current.id in seen:
            raise ReuseLineageError("family_reuse_lineage_invalid")
        seen.add(current.id)
        lineage.append(current.id)
        run = await session.get(MLRefinementRunRecord, current.run_id)
        if (
            run is None
            or run.status != "completed"
            or run.dataset_fingerprint != dataset_fingerprint
            or not await completed_run_is_complete(session, run)
            or not _scientifically_identical(selected_evaluation, current)
        ):
            raise ReuseLineageError("family_reuse_lineage_invalid")
        for manifest in await _compatible_manifests(
            session,
            evaluation_id=current.id,
            dataset_fingerprint=dataset_fingerprint,
        ):
            manifests.append((depth, current.id, manifest))
        source_id = current.reused_from_evaluation_id
        if source_id is None:
            break
        if source_id in seen:
            raise ReuseLineageError("family_reuse_lineage_invalid")
        source = await session.get(MLRefinementEvaluationRecord, source_id)
        if source is None:
            raise ReuseLineageError("family_reuse_lineage_invalid")
        current = source
    else:
        if current.reused_from_evaluation_id is not None:
            raise ReuseLineageError("family_reuse_lineage_invalid")

    if not manifests:
        raise ReuseLineageError("family_temporal_manifest_missing_or_incompatible")
    identities = {
        (
            item.manifest_fingerprint,
            item.dataset_fingerprint,
            item.manifest_version,
            item.split_policy_version,
        )
        for _, _, item in manifests
    }
    if len(identities) != 1:
        raise ReuseLineageError("family_temporal_manifest_ambiguous")
    _, owner_id, manifest = min(manifests, key=lambda item: (item[0], str(item[2].id)))
    return ResolvedDevelopmentEvidence(
        selected_evaluation=selected_evaluation,
        selected_run=selected_run,
        evaluation_ids=tuple(lineage),
        manifest_owner_evaluation_id=owner_id,
        manifest=manifest,
    )
