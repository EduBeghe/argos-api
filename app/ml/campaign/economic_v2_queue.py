from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.market.catalog import InstrumentCapability, resolve_instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.loading import load_ml_qualification_dataset
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)
from app.ml.refinement.persistence import completed_run_is_complete
from app.ml.refinement.planning import planned_evaluation_from_persisted_values

from .contracts import (
    DIAGNOSTIC_QUALIFICATION_PROVENANCE,
    ECONOMIC_V2_STRATEGY_POLICY_VERSION,
)
from .economic_v2_identity import (
    identical_payload,
    qualification_identity,
    qualification_identity_payload,
)
from .economic_v2_policy import strategy_policy_grid
from .economic_v2_qualification import calculate_qualification
from .models import (
    MLEconomicQualificationCampaignRecord,
    MLEconomicQualificationRecord,
    MLSearchCampaignFamilyRecord,
)

logger = logging.getLogger(__name__)


class QualificationLeaseLostError(RuntimeError):
    pass


class QualificationCancelledError(RuntimeError):
    pass


async def heartbeat_qualification(
    session: AsyncSession, *, qualification_id: UUID, worker_id: str
) -> bool:
    result = await session.execute(
        update(MLEconomicQualificationRecord)
        .where(
            MLEconomicQualificationRecord.id == qualification_id,
            MLEconomicQualificationRecord.status == "running",
            MLEconomicQualificationRecord.worker_id == worker_id,
            MLEconomicQualificationRecord.cancellation_requested_at.is_(None),
        )
        .values(heartbeat_at=datetime.now(UTC))
    )
    await session.commit()
    return int(result.rowcount) == 1  # type: ignore[attr-defined]


async def _cancel_owned(
    session: AsyncSession, *, qualification_id: UUID, worker_id: str
) -> bool:
    result = await session.execute(
        update(MLEconomicQualificationRecord)
        .where(
            MLEconomicQualificationRecord.id == qualification_id,
            MLEconomicQualificationRecord.status == "running",
            MLEconomicQualificationRecord.worker_id == worker_id,
            MLEconomicQualificationRecord.cancellation_requested_at.is_not(None),
        )
        .values(
            status="cancelled",
            completed_at=datetime.now(UTC),
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
        )
    )
    await session.commit()
    return int(result.rowcount) == 1  # type: ignore[attr-defined]


async def claim_next_qualification(
    session: AsyncSession, *, worker_id: str
) -> MLEconomicQualificationRecord | None:
    row = await session.scalar(
        select(MLEconomicQualificationRecord)
        .where(MLEconomicQualificationRecord.status == "queued")
        .order_by(MLEconomicQualificationRecord.created_at)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if row is None:
        await session.rollback()
        return None
    timestamp = datetime.now(UTC)
    row.status = "running"
    row.worker_id = worker_id
    row.claimed_at = timestamp
    row.heartbeat_at = timestamp
    await session.commit()
    return row


async def fail_stale_qualifications(
    session: AsyncSession, *, stale_after: timedelta
) -> int:
    cutoff = datetime.now(UTC) - stale_after
    result = await session.execute(
        update(MLEconomicQualificationRecord)
        .where(
            MLEconomicQualificationRecord.status == "running",
            MLEconomicQualificationRecord.heartbeat_at < cutoff,
        )
        .values(
            status="queued",
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
            failure_type=None,
            failure_message=None,
        )
    )
    await session.commit()
    return int(result.rowcount)  # type: ignore[attr-defined]


async def process_qualification(
    session: AsyncSession,
    *,
    record: MLEconomicQualificationRecord,
    worker_id: str,
    heartbeat_interval_seconds: float,
) -> None:
    if heartbeat_interval_seconds <= 0:
        raise ValueError("qualification heartbeat interval must be positive")
    qualification_id = record.id
    campaign_id = record.campaign_id
    family_id = record.family_id
    provenance = record.provenance
    evidence_end_exclusive = record.evidence_end_exclusive
    candidate_fingerprint = record.candidate_fingerprint
    policy_fingerprint = record.policy_fingerprint
    deterministic_seed = int(str(record.frozen_inputs["deterministic_seed"]))
    session_bind = session.bind
    if record.status != "running" or record.worker_id != worker_id:
        raise QualificationLeaseLostError("worker does not own qualification lease")
    if record.cancellation_requested_at is not None:
        await _cancel_owned(
            session, qualification_id=qualification_id, worker_id=worker_id
        )
        return
    family = await session.get(MLSearchCampaignFamilyRecord, record.family_id)
    evaluation_id = record.frozen_inputs.get("evaluation_id")
    try:
        parsed_evaluation_id = UUID(str(evaluation_id))
    except ValueError as error:
        raise RuntimeError("qualification evaluation identity is invalid") from error
    evaluation = await session.get(MLRefinementEvaluationRecord, parsed_evaluation_id)
    run = (
        await session.get(MLRefinementRunRecord, evaluation.run_id)
        if evaluation is not None
        else None
    )
    association = await session.get(
        MLEconomicQualificationCampaignRecord,
        (record.campaign_id, record.id),
    )
    if (
        family is None
        or evaluation is None
        or run is None
        or run.status != "completed"
        or not await completed_run_is_complete(session, run)
        or family.best_evaluation_id != evaluation.id
        or association is None
        or association.family_id != family.id
        or evaluation.run_id != run.id
        or run.source_screening_run_id != family.source_screening_run_id
        or run.dataset_fingerprint != family.dataset_fingerprint
        or evaluation.fingerprint
        != str(record.frozen_inputs.get("evaluation_fingerprint"))
        or evaluation.engine != family.engine
        or evaluation.objective != family.objective
        or evaluation.future_duration_seconds != family.future_duration_seconds
        or record.evidence_end_exclusive != datetime(2025, 1, 1, tzinfo=UTC)
        or (
            record.provenance == DIAGNOSTIC_QUALIFICATION_PROVENANCE
            and record.temporal_manifest_fingerprint
            != str(record.frozen_inputs.get("temporal_manifest_fingerprint"))
        )
    ):
        raise RuntimeError("qualification_identity_mismatch")
    candidate = planned_evaluation_from_persisted_values(
        fingerprint=evaluation.fingerprint,
        engine=evaluation.engine,
        future_duration_seconds=evaluation.future_duration_seconds,
        objective=evaluation.objective,
        hyperparameters=evaluation.hyperparameters,
        feature_policy=evaluation.feature_policy,
    )
    policy_payload = record.policy_identity
    if (
        int(str(policy_payload.get("strategy_policy_version", 0)))
        != ECONOMIC_V2_STRATEGY_POLICY_VERSION
    ):
        raise RuntimeError("qualification strategy-policy version is incompatible")
    requested_identity = (
        str(policy_payload["policy_fingerprint"]),
        Decimal(str(policy_payload["confidence_fraction"])),
        Decimal(str(policy_payload["holding_multiplier"])),
        str(policy_payload["regime_policy"]),
        int(str(policy_payload["holding_candles"])),
    )
    policy = next(
        (
            item
            for item in strategy_policy_grid(evaluation.future_duration_seconds)
            if (
                item.fingerprint,
                item.confidence_fraction,
                item.holding_multiplier,
                item.regime_policy,
                item.holding_candles,
            )
            == requested_identity
        ),
        None,
    )
    if policy is None:
        raise RuntimeError("qualification_policy_identity_mismatch")
    rebuilt_identity = qualification_identity_payload(
        family=family,
        evaluation=evaluation,
        run=run,
        policy=record.policy_identity,
        provenance=record.provenance,
    )
    if (
        not identical_payload(record.frozen_inputs, rebuilt_identity)
        or qualification_identity(rebuilt_identity) != record.candidate_fingerprint
        or record.policy_fingerprint != policy.fingerprint
    ):
        raise RuntimeError("qualification_identity_mismatch")
    instrument = resolve_instrument(
        asset=Asset(family.asset),
        provider=Provider(family.provider),
        market_type=MarketType(family.market_type),
        interval=Interval(family.interval),
        capability=InstrumentCapability.SCORING,
    ).instrument
    if not await heartbeat_qualification(
        session, qualification_id=qualification_id, worker_id=worker_id
    ):
        if await _cancel_owned(
            session, qualification_id=qualification_id, worker_id=worker_id
        ):
            return
        raise QualificationLeaseLostError("worker lost qualification lease")
    logger.info(
        "Economic v2 qualification started",
        extra={
            "structured_fields": {
                "qualification_id": str(qualification_id),
                "campaign_id": str(campaign_id),
                "family_id": str(family_id),
                "provenance": provenance,
                "candidate_fingerprint": candidate_fingerprint,
                "policy_fingerprint": policy_fingerprint,
                "read_start": "2020-01-01T00:00:00+00:00",
                "read_end_exclusive": "2025-01-01T00:00:00+00:00",
            }
        },
    )
    stopped = asyncio.Event()
    lease_error: Exception | None = None
    work_error: Exception | None = None
    result = None
    factory = async_sessionmaker(session_bind, expire_on_commit=False)

    async def maintain_lease() -> None:
        nonlocal lease_error
        try:
            while not stopped.is_set():
                try:
                    await asyncio.wait_for(
                        stopped.wait(), timeout=heartbeat_interval_seconds
                    )
                except TimeoutError:
                    async with factory() as heartbeat_session:
                        if not await heartbeat_qualification(
                            heartbeat_session,
                            qualification_id=qualification_id,
                            worker_id=worker_id,
                        ):
                            raise QualificationLeaseLostError(
                                "worker lost qualification lease"
                            ) from None
        except Exception as error:
            lease_error = error

    heartbeat_task = asyncio.create_task(maintain_lease())
    try:
        dataset = await load_ml_qualification_dataset(
            session=session, instrument=instrument
        )
        await session.rollback()
        result = await asyncio.to_thread(
            calculate_qualification,
            dataset=dataset,
            candidate=candidate,
            policy=policy,
            deterministic_seed=deterministic_seed,
        )
    except Exception as error:
        work_error = error
    finally:
        stopped.set()
        await heartbeat_task
    if lease_error is not None:
        if await _cancel_owned(
            session, qualification_id=qualification_id, worker_id=worker_id
        ):
            return
        raise QualificationLeaseLostError(
            "worker lease could not be verified during qualification"
        ) from lease_error
    if work_error is not None:
        raise work_error
    assert result is not None
    final = await session.execute(
        update(MLEconomicQualificationRecord)
        .where(
            MLEconomicQualificationRecord.id == qualification_id,
            MLEconomicQualificationRecord.status == "running",
            MLEconomicQualificationRecord.worker_id == worker_id,
            MLEconomicQualificationRecord.cancellation_requested_at.is_(None),
        )
        .values(
            temporal_manifest=result.temporal_manifest,
            temporal_manifest_fingerprint=result.temporal_manifest_fingerprint,
            scenario_metrics=result.scenario_metrics,
            quarter_returns=[str(item) for item in result.quarter_returns],
            trades=list(result.trades),
            failed_rules=list(result.failed_rules),
            classification="qualified" if not result.failed_rules else "rejected",
            status="completed",
            completed_at=datetime.now(UTC),
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
        )
    )
    if int(final.rowcount) != 1:  # type: ignore[attr-defined]
        await session.rollback()
        if await _cancel_owned(
            session, qualification_id=qualification_id, worker_id=worker_id
        ):
            return
        raise QualificationLeaseLostError("worker lost qualification lease")
    await session.commit()
    logger.info(
        "Economic v2 qualification completed",
        extra={
            "structured_fields": {
                "qualification_id": str(qualification_id),
                "campaign_id": str(campaign_id),
                "family_id": str(family_id),
                "provenance": provenance,
                "classification": "qualified"
                if not result.failed_rules
                else "rejected",
                "failed_rules": len(result.failed_rules),
                "evidence_end_exclusive": str(evidence_end_exclusive),
            }
        },
    )
