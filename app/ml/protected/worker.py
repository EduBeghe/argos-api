from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import InstrumentCapability, resolve_instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.campaign.models import MLSearchCampaignFamilyRecord
from app.ml.loading import (
    load_canonical_ml_experiment_dataset,
    load_canonical_ml_experiment_metadata,
    load_ml_development_dataset,
)
from app.ml.reader import read_ml_experiment_candles
from app.ml.refinement.calculation import _selected_columns, fit_refinement_model
from app.ml.refinement.models import MLRefinementEvaluationRecord, MLRefinementRunRecord
from app.ml.registry import VARIABLE_REGISTRY
from app.ml.screening.models import MLModelScreeningRunRecord

from .artifacts import (
    TRUSTED_PROVENANCE,
    load_trusted_artifact,
    serialize_trusted_artifact,
)
from .contracts import ARTIFACT_FORMAT_VERSION, EXECUTABLE_STRATEGY_VERSION
from .models import (
    MLFinalValidationRecord,
    MLFinalValidationTradeRecord,
    MLModelArtifactRecord,
    MLStrategySpecificationRecord,
    MLTemporalManifestRecord,
)
from .queue import _sha
from .temporal import reconstruct_temporal_manifest
from .validation import calculate_protected_validation


def _json(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


async def process_manifest_reconstruction(
    session: AsyncSession,
    *,
    record: MLTemporalManifestRecord,
    worker_id: str,
) -> None:
    if record.status != "running" or record.worker_id != worker_id:
        raise RuntimeError("worker does not own development-manifest lease")
    evaluation = await session.get(MLRefinementEvaluationRecord, record.evaluation_id)
    run = (
        await session.get(MLRefinementRunRecord, evaluation.run_id)
        if evaluation
        else None
    )
    if evaluation is None or run is None:
        raise RuntimeError("manifest source evaluation is missing")
    source = await session.get(MLModelScreeningRunRecord, run.source_screening_run_id)
    if source is None:
        raise RuntimeError("manifest source screening run is missing")
    definition = resolve_instrument(
        asset=Asset(source.asset),
        provider=Provider(source.provider),
        market_type=MarketType(source.market_type),
        interval=Interval(source.interval),
        capability=InstrumentCapability.SCORING,
    )
    identity = await load_canonical_ml_experiment_metadata(
        session=session, instrument=definition.instrument
    )
    if identity.fingerprint != record.dataset_fingerprint:
        raise RuntimeError("canonical_dataset_fingerprint_mismatch")
    development = await load_ml_development_dataset(
        session=session, instrument=definition.instrument
    )
    reconstructed = reconstruct_temporal_manifest(
        dataset=development,
        evaluation=evaluation,
        canonical_dataset_fingerprint=identity.fingerprint,
    )
    record.manifest = reconstructed.payload
    record.manifest_fingerprint = reconstructed.fingerprint
    record.fingerprint_verified = True
    record.counts_verified = True
    record.status = "completed"
    record.completed_at = datetime.now(UTC)
    record.worker_id = None
    record.claimed_at = None
    record.heartbeat_at = None
    await session.commit()


async def process_final_validation(
    session: AsyncSession,
    *,
    record: MLFinalValidationRecord,
    worker_id: str,
) -> None:
    if record.status != "running" or record.worker_id != worker_id:
        raise RuntimeError("worker does not own final-validation lease")
    family = await session.get(MLSearchCampaignFamilyRecord, record.family_id)
    evaluation = await session.get(MLRefinementEvaluationRecord, record.evaluation_id)
    if family is None or evaluation is None:
        raise RuntimeError("frozen final-validation source is missing")
    await session.refresh(record)
    if record.cancellation_requested_at is not None:
        record.status = "cancelled"
        record.completed_at = datetime.now(UTC)
        record.worker_id = None
        record.claimed_at = None
        record.heartbeat_at = None
        await session.commit()
        return
    manifest = await session.get(MLTemporalManifestRecord, record.temporal_manifest_id)
    if (
        manifest is None
        or manifest.status != "completed"
        or not manifest.fingerprint_verified
        or not manifest.counts_verified
        or manifest.manifest_fingerprint
        != record.frozen_inputs.get("manifest_fingerprint")
        or str(manifest.id) != record.frozen_inputs.get("manifest_id")
        or manifest.evaluation_id != record.evaluation_id
    ):
        raise RuntimeError("frozen development manifest ownership is invalid")
    instrument = resolve_instrument(
        asset=Asset(family.asset),
        provider=Provider(family.provider),
        market_type=MarketType(family.market_type),
        interval=Interval(family.interval),
        capability=InstrumentCapability.SCORING,
    ).instrument
    record.protected_data_read_at = datetime.now(UTC)
    await session.commit()
    dataset = await load_canonical_ml_experiment_dataset(
        session=session, instrument=instrument
    )
    if dataset.metadata.fingerprint != record.frozen_inputs.get("dataset_fingerprint"):
        raise RuntimeError("canonical dataset fingerprint changed before validation")
    candles = await read_ml_experiment_candles(
        session=session,
        instrument=instrument,
        start=dataset.metadata.requested_start,
        end=dataset.metadata.requested_end,
    )
    selected = record.frozen_inputs.get("selected_confidence_bucket")
    if not isinstance(selected, dict) or "fraction" not in selected:
        raise RuntimeError("frozen selected confidence bucket is invalid")
    configuration = record.frozen_inputs.get("normalized_configuration")
    seed_value = (
        configuration.get("deterministic_seed", 0)
        if isinstance(configuration, dict)
        else 0
    )
    if not isinstance(seed_value, int):
        raise RuntimeError("frozen deterministic seed is invalid")
    calculation = calculate_protected_validation(
        dataset=dataset,
        candles=candles.columns,
        instrument=instrument,
        evaluation=evaluation,
        confidence_fraction=Decimal(str(selected["fraction"])),
        seed=seed_value,
    )
    scenarios = {
        "base": calculation.base,
        "stress_1_5x": calculation.stressed_1_5x,
        "stress_2x": calculation.stressed_2x,
    }
    record.scenario_metrics = _json(
        {
            "scenarios": {
                name: {
                    key: item
                    for key, item in asdict(result).items()
                    if key not in {"trades", "equity_curve"}
                }
                for name, result in scenarios.items()
            },
            "buy_and_hold": asdict(calculation.buy_and_hold),
            "half_year_returns": calculation.half_year_returns,
        }
    )  # type: ignore[assignment]
    record.failed_rules = list(calculation.failed_rules)
    record.classification = "passed" if not calculation.failed_rules else "rejected"
    record.execution_manifest = {
        "source": "persisted_at_execution",
        "manifest_version": 1,
        "evaluation_id": str(evaluation.id),
        "evaluation_fingerprint": evaluation.fingerprint,
        "canonical_dataset_fingerprint": dataset.metadata.fingerprint,
        "dataset_requested_start": dataset.metadata.requested_start.isoformat(),
        "dataset_requested_end_exclusive": dataset.metadata.requested_end.isoformat(),
        "protected_start": "2024-01-01T00:00:00+00:00",
        "protected_end_exclusive": "2026-01-01T00:00:00+00:00",
        "future_duration_seconds": evaluation.future_duration_seconds,
        "development_row_count": calculation.development_row_count,
        "protected_row_count": calculation.protected_row_count,
        "first_protected_boundary": calculation.first_protected_boundary.isoformat(),
        "last_protected_boundary": calculation.last_protected_boundary.isoformat(),
        "feature_exclusion_counts": dataset.feature_exclusions.as_dict(),
        "selected_features": list(calculation.selected_feature_names),
    }
    for scenario, simulation in scenarios.items():
        for ordinal, trade in enumerate(simulation.trades):
            session.add(
                MLFinalValidationTradeRecord(
                    id=uuid4(),
                    validation_id=record.id,
                    scenario=scenario,
                    ordinal=ordinal,
                    entry_at=trade.entry_at,
                    exit_at=trade.exit_at,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    quantity=trade.quantity,
                    fees=trade.fees,
                    slippage=trade.slippage,
                    net_return=trade.net_return,
                    exit_reason=trade.exit_reason,
                )
            )
    record.status = "completed"
    record.completed_at = datetime.now(UTC)
    record.worker_id = None
    record.claimed_at = None
    record.heartbeat_at = None
    await session.commit()
    if calculation.failed_rules:
        return
    target = dataset.targets.for_duration(
        timedelta(seconds=evaluation.future_duration_seconds)
    )
    rows = np.flatnonzero(dataset.feature_eligible & target.eligible)
    full_columns = _selected_columns(
        calculation.candidate, dataset.features[rows], target.values[rows]
    )
    columns = np.asarray(full_columns, dtype=np.int64)
    selected_feature_names = tuple(VARIABLE_REGISTRY[index].name for index in columns)
    deployable_model = fit_refinement_model(
        candidate=calculation.candidate,
        training_x=dataset.features[np.ix_(rows, columns)],
        labels=(target.values[rows] > 0).astype(np.int64),
        seed=seed_value,
    )
    specification_payload: dict[str, object] = {
        **record.frozen_inputs,
        "strategy_name": "ml_fixed_horizon_long_only",
        "strategy_version": EXECUTABLE_STRATEGY_VERSION,
        "selected_features": list(selected_feature_names),
        "signed_score_threshold": str(selected["fraction"]),
        "decision_cadence_seconds": 1800,
        "allocation_percentage": "100",
    }
    specification = MLStrategySpecificationRecord(
        id=uuid4(),
        final_validation_id=record.id,
        strategy_name="ml_fixed_horizon_long_only",
        strategy_version=EXECUTABLE_STRATEGY_VERSION,
        specification=specification_payload,
        fingerprint=_sha(specification_payload),
        created_at=datetime.now(UTC),
    )
    serialized = serialize_trusted_artifact(
        {"model": deployable_model, "features": selected_feature_names}
    )
    versions = record.frozen_inputs.get("library_versions")
    if not isinstance(versions, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in versions.items()
    ):
        raise RuntimeError("frozen library versions are invalid")
    loaded = load_trusted_artifact(
        serialized.payload,
        expected_sha256=serialized.sha256,
        provenance=TRUSTED_PROVENANCE,
        format_version=serialized.format_version,
        stored_library_versions=versions,
        current_library_versions=versions,
    )
    if not isinstance(loaded, dict) or loaded.get("features") != selected_feature_names:
        raise RuntimeError("serialized artifact feature-order round trip failed")
    loaded_model = loaded.get("model")
    if loaded_model is None or not hasattr(loaded_model, "predict_proba"):
        raise RuntimeError("serialized artifact model round trip failed")
    sample = dataset.features[np.ix_(rows[: min(32, len(rows))], columns)]
    expected = np.asarray(deployable_model.predict_proba(sample), dtype=np.float64)
    actual = np.asarray(loaded_model.predict_proba(sample), dtype=np.float64)
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise RuntimeError("serialized artifact prediction parity failed")
    session.add(specification)
    await session.flush()
    session.add(
        MLModelArtifactRecord(
            id=uuid4(),
            strategy_specification_id=specification.id,
            trusted_provenance=TRUSTED_PROVENANCE,
            format_version=ARTIFACT_FORMAT_VERSION,
            compression=serialized.compression,
            byte_length=serialized.byte_length,
            sha256=serialized.sha256,
            engine=evaluation.engine,
            library_versions=versions,
            feature_schema=list(selected_feature_names),
            training_identity={
                "start": dataset.metadata.requested_start.isoformat(),
                "end_exclusive": dataset.metadata.requested_end.isoformat(),
                "dataset_fingerprint": dataset.metadata.fingerprint,
            },
            artifact=serialized.payload,
            load_validated_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
        )
    )
    await session.commit()
