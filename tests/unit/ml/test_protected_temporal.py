from copy import deepcopy

import pytest

from app.ml.contract import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    temporal_fold_definitions,
)
from app.ml.protected.temporal import (
    TemporalManifestError,
    temporal_manifest_fingerprint,
    validate_persisted_development_manifest,
)


def development_manifest() -> dict[str, object]:
    folds: list[dict[str, object]] = []
    for ordinal, definition in enumerate(temporal_fold_definitions()):
        folds.append(
            {
                "ordinal": ordinal,
                "fold_name": definition.name,
                "training_start": definition.training_start.isoformat(),
                "training_end_exclusive": definition.training_end.isoformat(),
                "validation_start": definition.validation_start.isoformat(),
                "validation_end_exclusive": definition.validation_end.isoformat(),
                "target_latest_endpoint_exclusive": (
                    definition.validation_end.isoformat()
                ),
                "training_eligibility": {
                    "first_eligible_boundary": None,
                    "last_eligible_boundary": None,
                },
                "validation_eligibility": {
                    "first_eligible_boundary": None,
                    "last_eligible_boundary": None,
                },
            }
        )
    return {
        "manifest_version": 1,
        "canonical_dataset_fingerprint": "d" * 64,
        "first_boundary": DEVELOPMENT_START.isoformat(),
        "folds": folds,
    }


def validate(payload: dict[str, object]) -> None:
    validate_persisted_development_manifest(
        payload=payload,
        persisted_fingerprint=temporal_manifest_fingerprint(payload),
        canonical_dataset_fingerprint="d" * 64,
        development_end_exclusive=DEVELOPMENT_END,
    )


def test_current_development_manifest_accepts_exact_end_exclusive_boundary() -> None:
    validate(development_manifest())


def test_development_manifest_rejects_post_boundary_evidence() -> None:
    payload = deepcopy(development_manifest())
    folds = payload["folds"]
    assert isinstance(folds, list)
    last = folds[-1]
    assert isinstance(last, dict)
    eligibility = last["validation_eligibility"]
    assert isinstance(eligibility, dict)
    eligibility["last_eligible_boundary"] = "2025-01-01T00:00:00+00:00"

    with pytest.raises(TemporalManifestError, match="crosses development evidence"):
        validate(payload)


def test_development_manifest_rejects_noncanonical_fingerprint() -> None:
    payload = development_manifest()
    with pytest.raises(TemporalManifestError, match="fingerprint mismatch"):
        validate_persisted_development_manifest(
            payload=payload,
            persisted_fingerprint="x" * 64,
            canonical_dataset_fingerprint="d" * 64,
            development_end_exclusive=DEVELOPMENT_END,
        )
