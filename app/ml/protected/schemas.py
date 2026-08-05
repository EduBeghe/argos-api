from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class IntegritySection(BaseModel):
    valid: bool
    failure_codes: tuple[str, ...] = ()
    provenance_strength: str | None = None


class CampaignFamilyAuditResponse(BaseModel):
    campaign_id: UUID
    family_id: UUID
    family_fingerprint: str
    dataset_fingerprint: str
    deterministic_ordinal: int
    state: str
    stage: int
    best_evaluation_id: UUID | None
    best_evaluation_child_id: UUID | None
    campaign_identity: dict[str, object]
    family_identity: dict[str, object]
    source_identity: dict[str, object]
    child_stages: tuple[dict[str, object], ...]
    winning_evaluation: dict[str, object] | None
    winning_configuration: dict[str, object] | None
    fold_evidence: list[dict[str, object]]
    temporal_manifest: dict[str, object] | None
    reconstructed_manifest_status: str | None
    reconstructed_manifest_id: UUID | None
    reconstructed_manifest_failure: dict[str, str] | None
    temporal_provenance: dict[str, object] | None
    evidence_integrity: IntegritySection
    temporal_provenance_integrity: IntegritySection
    overall_valid: bool
    independent_sample_diagnostic: dict[str, object] | None


class TemporalManifestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    evaluation_id: UUID
    source: str
    status: str
    manifest_version: int
    dataset_fingerprint: str
    manifest_fingerprint: str | None
    manifest: dict[str, object] | None
    fingerprint_verified: bool | None
    counts_verified: bool | None
    failure_code: str | None
    failure_message: str | None
    created_at: datetime
    completed_at: datetime | None


class FinalValidationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    experiment_key: str
    campaign_id: UUID
    family_id: UUID
    evaluation_id: UUID
    temporal_manifest_id: UUID
    status: str
    classification: str | None
    frozen_inputs: dict[str, object]
    execution_manifest: dict[str, object] | None
    scenario_metrics: dict[str, object] | None
    failed_rules: list[str] | None
    protected_data_read_at: datetime | None
    failure_code: str | None
    failure_message: str | None
    created_at: datetime
    completed_at: datetime | None


class FinalValidationPage(BaseModel):
    items: tuple[FinalValidationResponse, ...]
    next_cursor: str | None
    has_more: bool


class PaperDeploymentCreateRequest(BaseModel):
    final_validation_id: UUID
    display_name: str = Field(min_length=1, max_length=80)
    make_primary: bool = False


class ArtifactMetadataResponse(BaseModel):
    id: UUID
    format_version: int
    compression: str
    byte_length: int
    sha256: str
    engine: str


class PaperDeploymentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    display_name: str
    lifecycle: str
    strategy_kind: str
    strategy_specification_id: UUID | None
    current_run_id: UUID | None
    is_primary: bool
    created_at: datetime
    paused_at: datetime | None
    archived_at: datetime | None
    current_cash: Decimal | None = None
    open_position_count: int = 0
    artifact: ArtifactMetadataResponse | None = None


class PaperDeploymentPage(BaseModel):
    items: tuple[PaperDeploymentResponse, ...]
    next_cursor: str | None
    has_more: bool
