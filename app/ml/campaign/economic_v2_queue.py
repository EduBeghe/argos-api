from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import InstrumentCapability, resolve_instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.loading import load_ml_qualification_dataset
from app.ml.refinement.models import MLRefinementEvaluationRecord

from .contracts import (
    ECONOMIC_V2_QUALIFICATION_VERSION,
    ECONOMIC_V2_STRATEGY_POLICY_VERSION,
)
from .economic_v2_policy import strategy_policy_grid
from .economic_v2_qualification import calculate_qualification
from .models import (
    MLEconomicQualificationRecord,
    MLSearchCampaignFamilyRecord,
)

logger = logging.getLogger(__name__)


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
) -> None:
    if record.status != "running" or record.worker_id != worker_id:
        raise RuntimeError("worker does not own qualification lease")
    if record.cancellation_requested_at is not None:
        record.status = "cancelled"
        record.completed_at = datetime.now(UTC)
        record.worker_id = None
        record.claimed_at = None
        record.heartbeat_at = None
        await session.commit()
        return
    family = await session.get(MLSearchCampaignFamilyRecord, record.family_id)
    evaluation_id = record.frozen_inputs.get("evaluation_id")
    try:
        parsed_evaluation_id = UUID(str(evaluation_id))
    except ValueError as error:
        raise RuntimeError("qualification evaluation identity is invalid") from error
    evaluation = await session.get(MLRefinementEvaluationRecord, parsed_evaluation_id)
    if (
        family is None
        or evaluation is None
        or family.best_evaluation_id != evaluation.id
    ):
        raise RuntimeError("qualification frozen candidate ownership is invalid")
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
        raise RuntimeError("qualification strategy-policy identity is malformed")
    instrument = resolve_instrument(
        asset=Asset(family.asset),
        provider=Provider(family.provider),
        market_type=MarketType(family.market_type),
        interval=Interval(family.interval),
        capability=InstrumentCapability.SCORING,
    ).instrument
    logger.info(
        "Economic v2 qualification started",
        extra={
            "structured_fields": {
                "qualification_id": str(record.id),
                "candidate_fingerprint": record.candidate_fingerprint,
                "policy_fingerprint": record.policy_fingerprint,
                "read_start": "2020-01-01T00:00:00+00:00",
                "read_end_exclusive": "2025-01-01T00:00:00+00:00",
            }
        },
    )
    dataset = await load_ml_qualification_dataset(
        session=session, instrument=instrument
    )
    result = calculate_qualification(
        dataset=dataset,
        evaluation=evaluation,
        policy=policy,
        deterministic_seed=int(str(record.frozen_inputs["deterministic_seed"])),
    )
    await session.refresh(record)
    if record.worker_id != worker_id or record.status != "running":
        raise RuntimeError("worker lost qualification lease")
    record.temporal_manifest = result.temporal_manifest
    record.temporal_manifest_fingerprint = result.temporal_manifest_fingerprint
    record.scenario_metrics = result.scenario_metrics
    record.quarter_returns = [str(item) for item in result.quarter_returns]
    record.trades = list(result.trades)
    record.failed_rules = list(result.failed_rules)
    record.classification = "qualified" if not result.failed_rules else "rejected"
    record.status = "completed"
    record.completed_at = datetime.now(UTC)
    record.worker_id = None
    record.claimed_at = None
    record.heartbeat_at = None
    await session.commit()
    logger.info(
        "Economic v2 qualification completed",
        extra={
            "structured_fields": {
                "qualification_id": str(record.id),
                "classification": record.classification,
                "failed_rules": len(result.failed_rules),
            }
        },
    )


def qualification_identity(
    *, evaluation_fingerprint: str, policy_fingerprint: str
) -> str:
    payload = {
        "evaluation_fingerprint": evaluation_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "qualification_version": ECONOMIC_V2_QUALIFICATION_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
