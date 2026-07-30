from types import SimpleNamespace
from typing import cast

import pytest

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.contract import (
    DEVELOPMENT_START,
    FINAL_TEST_END,
    MISSING_DATA_POLICY_VERSION,
    ML_EXPERIMENT_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
)
from app.ml.dataset import MLExperimentDataset
from app.ml.refinement.identity import (
    DatasetFingerprintChanged,
    DatasetProvenanceInvalid,
    validate_dataset_identity,
)
from app.ml.refinement.models import MLRefinementRunRecord
from app.ml.screening.models import MLModelScreeningRunRecord

INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def records(
    *,
    source_fingerprint: str = "a" * 64,
    queued_fingerprint: str = "a" * 64,
    rebuilt_fingerprint: str = "a" * 64,
) -> tuple[MLModelScreeningRunRecord, MLRefinementRunRecord, MLExperimentDataset]:
    source = SimpleNamespace(
        dataset_fingerprint=source_fingerprint,
        experiment_contract_version=ML_EXPERIMENT_CONTRACT_VERSION,
        variable_registry_version=VARIABLE_REGISTRY_VERSION,
        missing_data_policy_version=MISSING_DATA_POLICY_VERSION,
    )
    queued = SimpleNamespace(
        dataset_fingerprint=queued_fingerprint,
        variable_registry_version=VARIABLE_REGISTRY_VERSION,
        missing_data_policy_version=MISSING_DATA_POLICY_VERSION,
    )
    dataset = SimpleNamespace(
        metadata=SimpleNamespace(
            fingerprint=rebuilt_fingerprint,
            instrument=INSTRUMENT,
            requested_start=DEVELOPMENT_START,
            requested_end=FINAL_TEST_END,
            candle_count=42,
        )
    )
    return (
        cast(MLModelScreeningRunRecord, source),
        cast(MLRefinementRunRecord, queued),
        cast(MLExperimentDataset, dataset),
    )


def test_three_way_identity_accepts_the_unchanged_pinned_dataset() -> None:
    source, queued, dataset = records()
    diagnostics = validate_dataset_identity(
        source=source, queued=queued, dataset=dataset
    )
    assert diagnostics.candle_count == 42
    assert diagnostics.historical_end_exclusive == FINAL_TEST_END.isoformat()


def test_historical_mutation_reports_each_mismatched_identity() -> None:
    source, queued, dataset = records(rebuilt_fingerprint="b" * 64)
    with pytest.raises(DatasetFingerprintChanged) as captured:
        validate_dataset_identity(source=source, queued=queued, dataset=dataset)
    message = str(captured.value)
    assert "source-versus-rebuilt" in message
    assert "queue-versus-rebuilt" in message
    assert captured.value.diagnostics is not None
    assert captured.value.diagnostics.candle_count == 42


@pytest.mark.parametrize("fingerprint", ["", "not-a-sha256", "A" * 64])
def test_malformed_persisted_fingerprint_fails_before_work(fingerprint: str) -> None:
    source, queued, dataset = records(source_fingerprint=fingerprint)
    with pytest.raises(DatasetProvenanceInvalid, match="malformed"):
        validate_dataset_identity(source=source, queued=queued, dataset=dataset)


def test_incompatible_contract_version_is_not_live_data_drift() -> None:
    source, queued, dataset = records()
    source.variable_registry_version += 1
    with pytest.raises(DatasetProvenanceInvalid, match="variable registry"):
        validate_dataset_identity(source=source, queued=queued, dataset=dataset)
