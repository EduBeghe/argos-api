from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from app.ml.contract import (
    MISSING_DATA_POLICY_VERSION,
    ML_EXPERIMENT_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
)
from app.ml.protected.models import MLTemporalManifestRecord
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)

from .contracts import (
    AUTOMATIC_QUALIFICATION_PROVENANCE,
    CAMPAIGN_FEE_RATE,
    CAMPAIGN_SLIPPAGE_RATE,
    DIAGNOSTIC_QUALIFICATION_REQUEST_VERSION,
    ECONOMIC_V2_CONTRACT_VERSION,
    ECONOMIC_V2_DEVELOPMENT_END,
    ECONOMIC_V2_DEVELOPMENT_START,
    ECONOMIC_V2_POLICY_VERSION,
    ECONOMIC_V2_QUALIFICATION_END,
    ECONOMIC_V2_QUALIFICATION_VERSION,
    ECONOMIC_V2_SEED,
)
from .models import MLSearchCampaignFamilyRecord


def canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def qualification_identity_payload(
    *,
    family: MLSearchCampaignFamilyRecord,
    evaluation: MLRefinementEvaluationRecord,
    run: MLRefinementRunRecord,
    policy: dict[str, object],
    provenance: str = AUTOMATIC_QUALIFICATION_PROVENANCE,
    development_manifest: MLTemporalManifestRecord | None = None,
) -> dict[str, object]:
    """Build the complete immutable scientific identity for 2024 qualification."""

    identity = {
        "qualification_version": ECONOMIC_V2_QUALIFICATION_VERSION,
        "campaign_contract_version": ECONOMIC_V2_CONTRACT_VERSION,
        "campaign_policy_version": ECONOMIC_V2_POLICY_VERSION,
        "experiment_contract_version": ML_EXPERIMENT_CONTRACT_VERSION,
        "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "missing_data_policy_version": MISSING_DATA_POLICY_VERSION,
        "campaign_preset": family.preset,
        "family_association": family.fingerprint,
        "family_fingerprint": family.fingerprint,
        "source_screening_run_id": str(family.source_screening_run_id),
        "dataset_fingerprint": family.dataset_fingerprint,
        "instrument": {
            "asset": family.asset,
            "provider": family.provider,
            "market_type": family.market_type,
            "interval": family.interval,
        },
        "engine": family.engine,
        "objective": family.objective,
        "future_duration_seconds": family.future_duration_seconds,
        "feature_variant": family.feature_variant,
        "search_track": family.search_track,
        "seed_family_id": (
            str(family.seed_family_id) if family.seed_family_id is not None else None
        ),
        "seed_provenance": sorted(family.seed_provenance),
        "deterministic_seed": ECONOMIC_V2_SEED,
        "evaluation_id": str(evaluation.id),
        "evaluation_fingerprint": evaluation.fingerprint,
        "evaluation_hyperparameters": evaluation.hyperparameters,
        "evaluation_feature_policy": evaluation.feature_policy,
        "refinement_run_id": str(run.id),
        "refinement_experiment_key": run.experiment_key,
        "refinement_contract_version": run.contract_version,
        "refinement_classification_version": run.classification_version,
        "refinement_dataset_fingerprint": run.dataset_fingerprint,
        "refinement_configuration": run.normalized_configuration,
        "library_versions": run.library_versions,
        "policy_identity": policy,
        "policy_fingerprint": str(policy["policy_fingerprint"]),
        "development_start": ECONOMIC_V2_DEVELOPMENT_START,
        "development_end_exclusive": ECONOMIC_V2_DEVELOPMENT_END,
        "qualification_end_exclusive": ECONOMIC_V2_QUALIFICATION_END,
        "fee_rate": str(CAMPAIGN_FEE_RATE),
        "slippage_rate": str(CAMPAIGN_SLIPPAGE_RATE),
    }
    # Preserve historical automatic identities byte-for-byte. Diagnostic provenance is
    # identity-bearing so it can never collide with automatic scientific evidence.
    if provenance != AUTOMATIC_QUALIFICATION_PROVENANCE:
        if (
            development_manifest is None
            or development_manifest.manifest_fingerprint is None
        ):
            raise ValueError(
                "diagnostic qualification requires a verified development manifest"
            )
        identity["qualification_provenance"] = provenance
        diagnostic_manifest = {
            "qualification_start": ECONOMIC_V2_DEVELOPMENT_END,
            "qualification_end_exclusive": ECONOMIC_V2_QUALIFICATION_END,
            "status": "not_read",
            "request_version": DIAGNOSTIC_QUALIFICATION_REQUEST_VERSION,
        }
        identity.update(
            {
                "diagnostic_request_version": DIAGNOSTIC_QUALIFICATION_REQUEST_VERSION,
                "qualification_start": ECONOMIC_V2_DEVELOPMENT_END,
                "temporal_manifest_fingerprint": qualification_identity(
                    diagnostic_manifest
                ),
                "development_temporal_manifest_id": str(development_manifest.id),
                "development_temporal_manifest_fingerprint": (
                    development_manifest.manifest_fingerprint
                ),
                "development_temporal_manifest_version": (
                    development_manifest.manifest_version
                ),
                "development_split_policy_version": (
                    development_manifest.split_policy_version
                ),
                "cost_multipliers": ["1", "1.5", "2"],
            }
        )
    return identity


def qualification_identity(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def identical_payload(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return canonical_json(left) == canonical_json(right)
