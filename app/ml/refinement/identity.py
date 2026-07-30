import re
from dataclasses import dataclass

from app.ml.contract import (
    MISSING_DATA_POLICY_VERSION,
    ML_EXPERIMENT_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
)
from app.ml.dataset import MLDatasetMetadata
from app.ml.screening.models import MLModelScreeningRunRecord

from .models import MLRefinementRunRecord

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DatasetIdentityError(RuntimeError):
    def __init__(
        self, message: str, diagnostics: "DatasetIdentityDiagnostics | None" = None
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class DatasetFingerprintChanged(DatasetIdentityError):
    pass


class DatasetProvenanceInvalid(DatasetIdentityError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetIdentityDiagnostics:
    source_fingerprint: str
    queued_fingerprint: str
    rebuilt_fingerprint: str
    instrument: str
    historical_start: str
    historical_end_exclusive: str
    candle_count: int
    contract_version: int
    variable_registry_version: int
    missing_data_policy_version: int

    def structured_fields(self) -> dict[str, str | int]:
        return {
            "source_screening_fingerprint": self.source_fingerprint,
            "queued_refinement_fingerprint": self.queued_fingerprint,
            "rebuilt_fingerprint": self.rebuilt_fingerprint,
            "instrument": self.instrument,
            "historical_start": self.historical_start,
            "historical_end_exclusive": self.historical_end_exclusive,
            "candle_count": self.candle_count,
            "experiment_contract_version": self.contract_version,
            "variable_registry_version": self.variable_registry_version,
            "missing_data_policy_version": self.missing_data_policy_version,
        }


def validate_dataset_identity(
    *,
    source: MLModelScreeningRunRecord,
    queued: MLRefinementRunRecord,
    rebuilt: MLDatasetMetadata,
) -> DatasetIdentityDiagnostics:
    diagnostics = DatasetIdentityDiagnostics(
        source.dataset_fingerprint,
        queued.dataset_fingerprint,
        rebuilt.fingerprint,
        "/".join(
            (
                rebuilt.instrument.asset.value,
                rebuilt.instrument.provider.value,
                rebuilt.instrument.market_type.value,
                rebuilt.instrument.interval.value,
            )
        ),
        rebuilt.requested_start.isoformat(),
        rebuilt.requested_end.isoformat(),
        rebuilt.candle_count,
        source.experiment_contract_version,
        source.variable_registry_version,
        source.missing_data_policy_version,
    )
    fingerprints = {
        "source": diagnostics.source_fingerprint,
        "queue": diagnostics.queued_fingerprint,
        "rebuilt": diagnostics.rebuilt_fingerprint,
    }
    malformed = [
        name for name, value in fingerprints.items() if not _SHA256.fullmatch(value)
    ]
    if malformed:
        raise DatasetProvenanceInvalid(
            f"malformed dataset fingerprint: {', '.join(malformed)}", diagnostics
        )
    versions = (
        (
            "experiment contract",
            source.experiment_contract_version,
            ML_EXPERIMENT_CONTRACT_VERSION,
        ),
        (
            "variable registry",
            source.variable_registry_version,
            VARIABLE_REGISTRY_VERSION,
        ),
        (
            "missing-data policy",
            source.missing_data_policy_version,
            MISSING_DATA_POLICY_VERSION,
        ),
        (
            "queued variable registry",
            queued.variable_registry_version,
            VARIABLE_REGISTRY_VERSION,
        ),
        (
            "queued missing-data policy",
            queued.missing_data_policy_version,
            MISSING_DATA_POLICY_VERSION,
        ),
    )
    incompatible = [name for name, actual, expected in versions if actual != expected]
    if incompatible:
        raise DatasetProvenanceInvalid(
            f"incompatible dataset provenance: {', '.join(incompatible)}", diagnostics
        )
    mismatches: list[str] = []
    if source.dataset_fingerprint != queued.dataset_fingerprint:
        mismatches.append("source-versus-queue")
    if source.dataset_fingerprint != rebuilt.fingerprint:
        mismatches.append("source-versus-rebuilt")
    if queued.dataset_fingerprint != rebuilt.fingerprint:
        mismatches.append("queue-versus-rebuilt")
    if mismatches:
        message = (
            f"dataset fingerprint mismatch ({', '.join(mismatches)}); "
            f"source={source.dataset_fingerprint}; "
            f"queue={queued.dataset_fingerprint}; "
            f"rebuilt={rebuilt.fingerprint}; "
            f"range=[{diagnostics.historical_start},"
            f"{diagnostics.historical_end_exclusive}); "
            f"candles={diagnostics.candle_count}"
        )
        raise DatasetFingerprintChanged(message, diagnostics)
    return diagnostics
