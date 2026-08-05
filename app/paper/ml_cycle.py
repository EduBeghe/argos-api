import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.ml.dataset import build_live_ml_feature_vector
from app.ml.gaps import ns_to_datetime
from app.ml.protected.artifacts import ArtifactIntegrityError, load_trusted_artifact
from app.ml.protected.contracts import DECISION_CADENCE_SECONDS
from app.ml.protected.models import MLModelArtifactRecord, MLStrategySpecificationRecord
from app.ml.reader import read_live_ml_candles
from app.ml.screening.engines import library_versions
from app.paper.calculation import PaperExecutionError
from app.paper.config import VERSION_1, quantize_financial
from app.paper.cycle_orchestration import CycleResult
from app.paper.decision_cycles import (
    MAX_EVIDENCE_AGE,
    DecisionCycleError,
    DecisionCycleStatus,
    claim_cycle,
    get_cycle_by_identity,
    validate_cycle_start,
)
from app.paper.decision_evidence import MLDecisionSnapshot, persist_ml_decision
from app.paper.deployment_models import PaperDeploymentRecord
from app.paper.execution import execute_decision
from app.paper.models import NoTrade
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradeRecord,
    get_open_position,
    get_run,
)
from app.strategies.baseline_long_only import StrategyAction


async def run_ml_instrument_cycle(
    *,
    session: AsyncSession,
    deployment_id: UUID,
    instrument: Instrument,
    boundary: datetime,
    execution_time: datetime,
) -> CycleResult:
    validate_cycle_start(boundary=boundary, execution_time=execution_time)
    deployment = await session.scalar(
        select(PaperDeploymentRecord)
        .where(PaperDeploymentRecord.id == deployment_id)
        .with_for_update()
    )
    if deployment is None or deployment.lifecycle != "active":
        raise DecisionCycleError("ML deployment is not active")
    if deployment.strategy_kind != "ml_fixed_horizon_long_only":
        raise DecisionCycleError("deployment is not an ML Strategy")
    if (
        deployment.current_run_id is None
        or deployment.strategy_specification_id is None
    ):
        raise DecisionCycleError("ML deployment lineage is incomplete")
    run = await get_run(session, run_id=deployment.current_run_id)
    if run is None:
        raise PaperExecutionError("ML deployment Paper run not found")
    specification = await session.get(
        MLStrategySpecificationRecord, deployment.strategy_specification_id
    )
    artifact = await session.scalar(
        select(MLModelArtifactRecord).where(
            MLModelArtifactRecord.strategy_specification_id
            == deployment.strategy_specification_id
        )
    )
    if specification is None or artifact is None:
        raise DecisionCycleError("ML Strategy specification or artifact is missing")
    validate_ml_decision_cadence(specification.specification)
    cycle = await claim_cycle(
        session=session,
        run_id=run.id,
        instrument=instrument,
        boundary=boundary,
        started_at=execution_time,
        strategy_name=specification.strategy_name,
        strategy_version=specification.strategy_version,
        configuration_version=VERSION_1.version,
    )
    if cycle is None:
        existing = await get_cycle_by_identity(
            session=session,
            run_id=run.id,
            instrument=instrument,
            boundary=boundary,
            strategy_name=specification.strategy_name,
            strategy_version=specification.strategy_version,
            configuration_version=VERSION_1.version,
        )
        if existing is None or existing.status != DecisionCycleStatus.COMPLETED.value:
            raise DecisionCycleError("ML Decision Cycle is already being processed")
        return _cycle_result(existing, skipped=True)
    columns = await read_live_ml_candles(
        session=session, instrument=instrument, boundary=boundary
    )
    live = build_live_ml_feature_vector(columns)
    source = live.source_index
    source_close = ns_to_datetime(int(columns.close_time_ns[source]))
    if source_close > boundary or boundary - source_close > MAX_EVIDENCE_AGE:
        raise DecisionCycleError("eligible ML candle evidence is stale or not closed")
    feature_indices = _feature_indices(live.feature_names, artifact.feature_schema)
    selected = live.values[np.asarray(feature_indices, dtype=np.int64)]
    loaded = load_trusted_artifact(
        artifact.artifact,
        expected_sha256=artifact.sha256,
        provenance=artifact.trusted_provenance,
        format_version=artifact.format_version,
        stored_library_versions=artifact.library_versions,
        current_library_versions=library_versions(),
    )
    if not isinstance(loaded, dict) or loaded.get("features") != tuple(
        artifact.feature_schema
    ):
        raise ArtifactIntegrityError("artifact feature order is incompatible")
    model = loaded.get("model")
    if (
        model is None
        or not hasattr(model, "predict_proba")
        or not hasattr(model, "classes_")
    ):
        raise ArtifactIntegrityError("artifact model does not support classification")
    probabilities = np.asarray(
        model.predict_proba(selected.reshape(1, -1)), dtype=np.float64
    )
    classes = np.asarray(model.classes_, dtype=np.int64)
    positive = np.flatnonzero(classes == 1)
    if probabilities.shape != (1, len(classes)) or len(positive) != 1:
        raise ArtifactIntegrityError("artifact class probabilities are incompatible")
    score = quantize_financial(Decimal(str(2 * probabilities[0, int(positive[0])] - 1)))
    threshold = _decimal_specification(
        specification.specification, "signed_score_threshold"
    )
    horizon = timedelta(
        seconds=_integer_specification(
            specification.specification, "future_duration_seconds"
        )
    )
    position = await get_open_position(
        session,
        run_id=run.id,
        asset=instrument.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=instrument.interval,
        lock=True,
    )
    snapshot = _snapshot(
        deployment=deployment,
        specification=specification,
        artifact=artifact,
        instrument=instrument,
        boundary=boundary,
        source_open=ns_to_datetime(int(columns.open_time_ns[source])),
        source_price=Decimal(str(columns.close_price[source])),
        score=score,
        threshold=threshold,
        feature_values=selected,
        position=position,
        horizon=horizon,
    )
    identity = await persist_ml_decision(session, snapshot)
    execution = await execute_decision(
        session=session,
        decision=snapshot,
        executed_at=execution_time,
        run_id=run.id,
        decision_identity_id=identity.id,
    )
    if not isinstance(execution, (PaperTradeRecord, NoTrade)):
        raise DecisionCycleError("unexpected ML Paper execution result")
    cycle.source_score_id = None
    cycle.decision_snapshot_id = None
    cycle.ml_decision_snapshot_id = snapshot.id
    cycle.decision_identity_id = identity.id
    cycle.decision_kind = "ml"
    cycle.paper_trade_id = (
        execution.id if isinstance(execution, PaperTradeRecord) else None
    )
    cycle.result = snapshot.action.value
    cycle.completed_at = execution_time
    cycle.status = DecisionCycleStatus.COMPLETED.value
    await session.flush()
    return _cycle_result(cycle, skipped=False)


def _snapshot(
    *,
    deployment: PaperDeploymentRecord,
    specification: MLStrategySpecificationRecord,
    artifact: MLModelArtifactRecord,
    instrument: Instrument,
    boundary: datetime,
    source_open: datetime,
    source_price: Decimal,
    score: Decimal,
    threshold: Decimal,
    feature_values: np.ndarray[tuple[int], np.dtype[np.float32]],
    position: PaperPositionRecord | None,
    horizon: timedelta,
) -> MLDecisionSnapshot:
    scheduled_exit = position.opened_at + horizon if position is not None else None
    if (
        position is not None
        and scheduled_exit is not None
        and boundary >= scheduled_exit
    ):
        action = StrategyAction.SELL_ALL
        rule = "ml_fixed_horizon_elapsed"
        target, percentage = None, Decimal("100")
    elif position is not None:
        action = StrategyAction.NO_ACTION
        rule = "ml_open_position_retained"
        target = percentage = None
    elif score >= threshold:
        action = StrategyAction.OPEN_POSITION
        rule = "ml_entry_threshold_met"
        target, percentage = Decimal("100"), None
    else:
        action = StrategyAction.NO_ACTION
        rule = "ml_entry_threshold_not_met"
        target = percentage = None
    delay = (
        max(0, int((boundary - scheduled_exit).total_seconds()))
        if scheduled_exit
        else 0
    )
    return MLDecisionSnapshot(
        deployment_id=deployment.id,
        strategy_specification_id=specification.id,
        model_artifact_id=artifact.id,
        asset=instrument.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=instrument.interval,
        decision_timestamp=boundary,
        source_candle_open_time=source_open,
        source_market_price=source_price,
        signed_prediction=score,
        entry_threshold=threshold,
        model_artifact_sha256=artifact.sha256,
        feature_schema=tuple(artifact.feature_schema),
        feature_vector_sha256=hashlib.sha256(feature_values.tobytes()).hexdigest(),
        action=action,
        requested_target_portfolio_percentage=target,
        requested_position_percentage=percentage,
        applied_rule=rule,
        explanation=f"Frozen ML Strategy rule: {rule}",
        position_is_open=position is not None,
        scheduled_exit_at=scheduled_exit,
        delayed_after_pause=delay > 0,
        execution_delay_seconds=delay,
    )


def _feature_indices(
    all_names: tuple[str, ...], selected: list[str]
) -> tuple[int, ...]:
    positions = {name: index for index, name in enumerate(all_names)}
    try:
        return tuple(positions[name] for name in selected)
    except KeyError as error:
        raise ArtifactIntegrityError(
            "artifact references an unknown feature"
        ) from error


def _decimal_specification(payload: dict[str, object], key: str) -> Decimal:
    try:
        value = Decimal(str(payload[key]))
    except (KeyError, ValueError) as error:
        raise DecisionCycleError(
            f"ML Strategy specification has invalid {key}"
        ) from error
    if not value.is_finite():
        raise DecisionCycleError(f"ML Strategy specification has invalid {key}")
    if key == "signed_score_threshold" and not Decimal(0) <= value <= Decimal(1):
        raise DecisionCycleError(f"ML Strategy specification has invalid {key}")
    return value


def _integer_specification(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DecisionCycleError(f"ML Strategy specification has invalid {key}")
    return value


def validate_ml_decision_cadence(payload: dict[str, object]) -> None:
    cadence = _integer_specification(payload, "decision_cadence_seconds")
    if cadence != DECISION_CADENCE_SECONDS:
        raise DecisionCycleError(
            "ML Strategy specification has unsupported decision cadence"
        )


def _cycle_result(cycle: object, *, skipped: bool) -> CycleResult:
    from app.paper.decision_cycles import PaperDecisionCycleRecord

    if not isinstance(cycle, PaperDecisionCycleRecord):
        raise DecisionCycleError("ML Decision Cycle record is invalid")
    if (
        cycle.ml_decision_snapshot_id is None
        or cycle.decision_identity_id is None
        or cycle.result is None
    ):
        raise DecisionCycleError("completed ML Decision Cycle is incomplete")
    return CycleResult(
        cycle.id,
        cycle.paper_run_id,
        cycle.scheduled_boundary,
        None,
        cycle.ml_decision_snapshot_id,
        cycle.decision_identity_id,
        "ml",
        cycle.paper_trade_id,
        cycle.result,
        skipped,
    )
