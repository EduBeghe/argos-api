from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Final

DEVELOPMENT_MANIFEST_RECONSTRUCTION_VERSION = 1
TEMPORAL_MANIFEST_VERSION = 1
FINAL_VALIDATION_POLICY_VERSION = 1
FINAL_VALIDATION_SIMULATION_VERSION = 2
EXECUTABLE_STRATEGY_VERSION = 1
ARTIFACT_FORMAT_VERSION = 1
PROTECTED_START = datetime(2024, 1, 1, tzinfo=UTC)
PROTECTED_END = datetime(2026, 1, 1, tzinfo=UTC)
DECISION_CADENCE = timedelta(minutes=30)
DECISION_CADENCE_SECONDS = 1800
SUPPORTED_CONFIDENCE_BUCKET_FRACTIONS = frozenset(
    {Decimal("0.05"), Decimal("0.10"), Decimal("0.20")}
)
CONFIDENCE_BUCKET_LEGACY_QUANTUM = Decimal("0.000000000000000001")
INITIAL_CASH = Decimal("10000")
BASE_FEE_RATE = Decimal("0.001")
BASE_SLIPPAGE_RATE = Decimal("0.0005")
MAXIMUM_ARTIFACT_BYTES = 32 * 1024 * 1024
PAPER_CASH_BUFFER_RATE = Decimal("0.000001")
INCOMPATIBLE_FINAL_VALIDATION_CONTRACT: Final = "incompatible_final_validation_contract"


class ConfidenceBucketContractError(ValueError):
    """A persisted confidence-bucket identity is invalid or unsupported."""


def legacy_confidence_bucket_representation(value: Decimal) -> str:
    """Return the exact historical decimal_metric(float(value)) representation."""

    return str(Decimal(str(float(value))).quantize(CONFIDENCE_BUCKET_LEGACY_QUANTUM))


def canonicalize_confidence_bucket(
    selected_fraction: object, configured_buckets: object
) -> Decimal:
    """Resolve only canonical or Argos-generated legacy bucket representations.

    Configured identities must be the exact supported Decimal strings. Selection accepts
    either that configured string or the exact fixed-scale value historically emitted by
    ``decimal_metric(float(configured_bucket))``. No tolerance or nearest-bucket rule is
    applied.
    """

    if not isinstance(configured_buckets, (list, tuple)) or not configured_buckets:
        raise ConfidenceBucketContractError("configured confidence buckets are invalid")
    configured: dict[str, Decimal] = {}
    identities: set[Decimal] = set()
    for raw in configured_buckets:
        if type(raw) is not str:
            raise ConfidenceBucketContractError(
                "configured confidence bucket must be a canonical string"
            )
        try:
            parsed = Decimal(raw)
        except (InvalidOperation, ValueError) as error:
            raise ConfidenceBucketContractError(
                "configured confidence bucket is malformed"
            ) from error
        canonical = next(
            (item for item in SUPPORTED_CONFIDENCE_BUCKET_FRACTIONS if item == parsed),
            None,
        )
        if canonical is None or not parsed.is_finite() or raw != str(canonical):
            raise ConfidenceBucketContractError(
                "configured confidence bucket is unsupported"
            )
        if canonical in identities:
            raise ConfidenceBucketContractError(
                "configured confidence bucket identity is duplicated"
            )
        identities.add(canonical)
        configured[str(canonical)] = canonical

    if type(selected_fraction) is not str:
        raise ConfidenceBucketContractError(
            "selected confidence bucket must be a persisted decimal string"
        )
    matches = tuple(
        canonical
        for canonical in configured.values()
        if selected_fraction
        in {
            str(canonical),
            legacy_confidence_bucket_representation(canonical),
        }
    )
    if len(matches) != 1:
        raise ConfidenceBucketContractError(
            "selected confidence bucket does not match one configured identity"
        )
    return matches[0]


class FinalValidationContractError(RuntimeError):
    code = INCOMPATIBLE_FINAL_VALIDATION_CONTRACT


def validate_final_validation_contract(frozen_inputs: object) -> None:
    """Reject frozen validation evidence that is not the exact current contract."""

    expected = (
        ("simulation_version", FINAL_VALIDATION_SIMULATION_VERSION),
        ("policy_version", FINAL_VALIDATION_POLICY_VERSION),
        ("decision_cadence_seconds", DECISION_CADENCE_SECONDS),
    )
    inputs = frozen_inputs if isinstance(frozen_inputs, dict) else {}
    for field, required in expected:
        value = inputs.get(field)
        if type(value) is not int or value != required:
            raise FinalValidationContractError(
                f"final-validation contract field {field} is incompatible"
            )


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
