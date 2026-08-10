import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from app.ml.contract import CANDLE_INTERVAL, SPLIT_GAP, temporal_fold_definitions
from app.ml.dataset import MLExperimentDataset
from app.ml.gaps import ns_to_datetime
from app.ml.refinement.models import MLRefinementEvaluationRecord

from .contracts import TEMPORAL_MANIFEST_VERSION


class TemporalManifestError(RuntimeError):
    """The immutable development evidence cannot be reconstructed exactly."""


@dataclass(frozen=True, slots=True)
class ReconstructedTemporalManifest:
    payload: dict[str, object]
    fingerprint: str


def temporal_manifest_fingerprint(payload: dict[str, object]) -> str:
    """Return the canonical fingerprint used by persisted temporal manifests."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_persisted_development_manifest(
    *,
    payload: dict[str, object],
    persisted_fingerprint: str,
    canonical_dataset_fingerprint: str,
    development_end_exclusive: datetime,
) -> None:
    """Validate the immutable development-only semantics without loading candles."""

    if temporal_manifest_fingerprint(payload) != persisted_fingerprint:
        raise TemporalManifestError("persisted temporal manifest fingerprint mismatch")
    if payload.get("manifest_version") != TEMPORAL_MANIFEST_VERSION:
        raise TemporalManifestError("persisted temporal manifest version mismatch")
    if payload.get("canonical_dataset_fingerprint") != canonical_dataset_fingerprint:
        raise TemporalManifestError("persisted temporal manifest dataset mismatch")
    if development_end_exclusive.tzinfo is None:
        raise TemporalManifestError("development boundary must be timezone-aware")
    boundary = development_end_exclusive.astimezone(UTC)
    folds = payload.get("folds")
    definitions = temporal_fold_definitions()
    if not isinstance(folds, list) or len(folds) != len(definitions):
        raise TemporalManifestError("persisted temporal manifest folds are missing")

    def parsed(value: object, field: str) -> datetime:
        if not isinstance(value, str):
            raise TemporalManifestError(f"temporal manifest {field} is invalid")
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise TemporalManifestError(
                f"temporal manifest {field} is invalid"
            ) from error
        if result.tzinfo is None:
            raise TemporalManifestError(f"temporal manifest {field} is naive")
        return result.astimezone(UTC)

    first_boundary = parsed(payload.get("first_boundary"), "first_boundary")
    if first_boundary >= boundary:
        raise TemporalManifestError("development manifest starts outside development")
    latest_validation_end: datetime | None = None
    for ordinal, (raw_fold, definition) in enumerate(
        zip(folds, definitions, strict=True)
    ):
        if not isinstance(raw_fold, dict):
            raise TemporalManifestError("temporal manifest fold is invalid")
        training_start = parsed(raw_fold.get("training_start"), "training_start")
        training_end = parsed(
            raw_fold.get("training_end_exclusive"), "training_end_exclusive"
        )
        validation_start = parsed(raw_fold.get("validation_start"), "validation_start")
        validation_end = parsed(
            raw_fold.get("validation_end_exclusive"), "validation_end_exclusive"
        )
        target_end = parsed(
            raw_fold.get("target_latest_endpoint_exclusive"),
            "target_latest_endpoint_exclusive",
        )
        if not (
            raw_fold.get("ordinal") == ordinal
            and raw_fold.get("fold_name") == definition.name
            and training_start == definition.training_start
            and training_end == definition.training_end
            and validation_start == definition.validation_start
            and validation_end == definition.validation_end
            and training_start < training_end <= validation_start < validation_end
            and validation_end == target_end
            and validation_end <= boundary
        ):
            raise TemporalManifestError(
                f"temporal manifest fold {ordinal} crosses development evidence"
            )
        for phase_name in ("training_eligibility", "validation_eligibility"):
            phase = raw_fold.get(phase_name)
            if not isinstance(phase, dict):
                raise TemporalManifestError(
                    f"temporal manifest {phase_name} is invalid"
                )
            for field in ("first_eligible_boundary", "last_eligible_boundary"):
                value = phase.get(field)
                if value is not None and parsed(value, field) >= boundary:
                    raise TemporalManifestError(
                        f"temporal manifest {field} crosses development evidence"
                    )
        latest_validation_end = max(
            validation_end, latest_validation_end or validation_end
        )
    if latest_validation_end != boundary:
        raise TemporalManifestError(
            "development manifest does not end at the protected boundary"
        )


def reconstruct_temporal_manifest(
    *,
    dataset: MLExperimentDataset,
    evaluation: MLRefinementEvaluationRecord,
    canonical_dataset_fingerprint: str,
) -> ReconstructedTemporalManifest:
    duration = timedelta(seconds=evaluation.future_duration_seconds)
    target = dataset.targets.for_duration(duration)
    eligible = dataset.feature_eligible & target.eligible
    persisted = evaluation.fold_evidence
    if len(persisted) != len(dataset.folds):
        raise TemporalManifestError("persisted fold count does not match the contract")
    folds: list[dict[str, object]] = []
    for ordinal, (fold, stored) in enumerate(
        zip(dataset.folds, persisted, strict=True)
    ):
        name = stored.get("fold_name")
        if name != fold.definition.name:
            raise TemporalManifestError(f"fold order mismatch at ordinal {ordinal}")
        training_count = int(np.count_nonzero(eligible[fold.training_indices]))
        validation_count = int(np.count_nonzero(eligible[fold.validation_indices]))
        if stored.get("training_row_count") != training_count:
            raise TemporalManifestError(f"training count mismatch for {name}")
        if stored.get("validation_row_count") != validation_count:
            raise TemporalManifestError(f"validation count mismatch for {name}")
        definition = fold.definition

        def phase_counts(
            indices: np.ndarray[tuple[int], np.dtype[np.int64]],
        ) -> dict[str, object]:
            feature_count = int(np.count_nonzero(dataset.feature_eligible[indices]))
            target_count = int(np.count_nonzero(target.eligible[indices]))
            final_indices = indices[eligible[indices]]
            return {
                "candidate_boundary_count": len(indices),
                "feature_eligible_count": feature_count,
                "target_eligible_count": target_count,
                "final_eligible_count": len(final_indices),
                "feature_excluded_count": len(indices) - feature_count,
                "target_or_joint_excluded_count": feature_count - len(final_indices),
                "first_eligible_boundary": ns_to_datetime(
                    int(dataset.boundary_time_ns[final_indices[0]])
                ).isoformat()
                if len(final_indices)
                else None,
                "last_eligible_boundary": ns_to_datetime(
                    int(dataset.boundary_time_ns[final_indices[-1]])
                ).isoformat()
                if len(final_indices)
                else None,
            }

        folds.append(
            {
                "ordinal": ordinal,
                "fold_name": name,
                "training_start": definition.training_start.isoformat(),
                "training_end_exclusive": definition.training_end.isoformat(),
                "validation_start": definition.validation_start.isoformat(),
                "validation_end_exclusive": definition.validation_end.isoformat(),
                "purge_embargo_seconds": int(SPLIT_GAP.total_seconds()),
                "purge_embargo_start": definition.training_end.isoformat(),
                "purge_embargo_end_exclusive": definition.validation_start.isoformat(),
                "target_duration_seconds": int(duration.total_seconds()),
                "target_latest_endpoint_exclusive": (
                    definition.validation_end.isoformat()
                ),
                "training_row_count": training_count,
                "validation_row_count": validation_count,
                "training_eligibility": phase_counts(fold.training_indices),
                "validation_eligibility": phase_counts(fold.validation_indices),
            }
        )
    payload: dict[str, object] = {
        "manifest_version": TEMPORAL_MANIFEST_VERSION,
        "canonical_dataset_fingerprint": canonical_dataset_fingerprint,
        "development_dataset_fingerprint": dataset.metadata.fingerprint,
        "candle_interval_seconds": int(CANDLE_INTERVAL.total_seconds()),
        "future_duration_seconds": evaluation.future_duration_seconds,
        "feature_exclusion_counts": dataset.feature_exclusions.as_dict(),
        "target_exclusion_counts": target.exclusions.as_dict(),
        "first_boundary": ns_to_datetime(int(dataset.boundary_time_ns[0])).isoformat(),
        "folds": folds,
    }
    return ReconstructedTemporalManifest(
        payload, temporal_manifest_fingerprint(payload)
    )
