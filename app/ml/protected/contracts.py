from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

DEVELOPMENT_MANIFEST_RECONSTRUCTION_VERSION = 1
TEMPORAL_MANIFEST_VERSION = 1
FINAL_VALIDATION_POLICY_VERSION = 1
FINAL_VALIDATION_SIMULATION_VERSION = 1
EXECUTABLE_STRATEGY_VERSION = 1
ARTIFACT_FORMAT_VERSION = 1
PROTECTED_START = datetime(2024, 1, 1, tzinfo=UTC)
PROTECTED_END = datetime(2026, 1, 1, tzinfo=UTC)
DECISION_CADENCE = timedelta(minutes=30)
INITIAL_CASH = Decimal("10000")
BASE_FEE_RATE = Decimal("0.001")
BASE_SLIPPAGE_RATE = Decimal("0.0005")
MAXIMUM_ARTIFACT_BYTES = 32 * 1024 * 1024
PAPER_CASH_BUFFER_RATE = Decimal("0.000001")


class WorkStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TemporalProvenance(StrEnum):
    PERSISTED_AT_EXECUTION = "persisted_at_execution"
    RECONSTRUCTED_FROM_CONTRACT = "reconstructed_from_contract"


class FinalValidationClassification(StrEnum):
    PASSED = "passed"
    REJECTED = "rejected"
    INVALID = "invalid"
    PASSED_EVIDENCE_ARTIFACT_FAILED = "passed_evidence_artifact_failed"


class DeploymentLifecycle(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class StrategyKind(StrEnum):
    BASELINE = "baseline"
    ML_FIXED_HORIZON_LONG_ONLY = "ml_fixed_horizon_long_only"
