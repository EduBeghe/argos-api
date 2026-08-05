from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import Table

from app.market.enums import Asset, Interval, MarketType, Provider
from app.paper.decision_cycles import PaperDecisionCycleRecord
from app.paper.decision_evidence import (
    MLDecisionSnapshot,
    MLDecisionSnapshotRecord,
    PaperDecisionIdentityRecord,
    persist_baseline_decision_identity,
    persist_ml_decision,
)
from app.paper.execution import execute_decision
from app.paper.models import NoTrade
from app.strategies.baseline_long_only import StrategyAction

NOW = datetime(2026, 8, 4, 12, tzinfo=UTC)


def snapshot(**changes: object) -> MLDecisionSnapshot:
    values: dict[str, object] = {
        "deployment_id": uuid4(),
        "strategy_specification_id": uuid4(),
        "model_artifact_id": uuid4(),
        "asset": Asset.BTC_USDT,
        "provider": Provider.BINANCE,
        "market_type": MarketType.SPOT,
        "interval": Interval.FIVE_MINUTES,
        "decision_timestamp": NOW,
        "source_candle_open_time": NOW - timedelta(minutes=5),
        "source_market_price": Decimal("60000.123456789"),
        "signed_prediction": Decimal("0.42"),
        "entry_threshold": Decimal("0.20"),
        "model_artifact_sha256": "a" * 64,
        "feature_schema": ("ema_20", "rsi_14"),
        "feature_vector_sha256": "b" * 64,
        "action": StrategyAction.OPEN_POSITION,
        "requested_target_portfolio_percentage": Decimal("100"),
        "requested_position_percentage": None,
        "applied_rule": "ml_entry_threshold_met",
        "explanation": "Frozen ML entry threshold was met.",
        "position_is_open": False,
    }
    values.update(changes)
    return MLDecisionSnapshot.model_validate(values)


def test_ml_snapshot_preserves_real_model_and_deployment_evidence() -> None:
    decision = snapshot()
    assert decision.signed_prediction == Decimal("0.42")
    assert decision.entry_threshold == Decimal("0.20")
    assert decision.feature_schema == ("ema_20", "rsi_14")
    assert decision.action is StrategyAction.OPEN_POSITION


@pytest.mark.parametrize(
    "changes",
    [
        {"model_artifact_sha256": "invalid"},
        {"feature_schema": ()},
        {"signed_prediction": Decimal("1.1")},
        {"applied_rule": "entry_threshold_met"},
        {"requested_target_portfolio_percentage": Decimal("15")},
    ],
)
def test_ml_snapshot_rejects_incomplete_or_baseline_evidence(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        snapshot(**changes)


def test_ml_sell_requires_matured_fixed_horizon() -> None:
    decision = snapshot(
        action=StrategyAction.SELL_ALL,
        requested_target_portfolio_percentage=None,
        requested_position_percentage=Decimal("100"),
        applied_rule="ml_fixed_horizon_elapsed",
        position_is_open=True,
        scheduled_exit_at=NOW - timedelta(minutes=30),
        delayed_after_pause=True,
        execution_delay_seconds=1800,
    )
    assert decision.scheduled_exit_at == NOW - timedelta(minutes=30)
    assert decision.delayed_after_pause is True


def test_ml_no_action_uses_ml_vocabulary() -> None:
    decision = snapshot(
        signed_prediction=Decimal("0.1"),
        action=StrategyAction.NO_ACTION,
        requested_target_portfolio_percentage=None,
        applied_rule="ml_entry_threshold_not_met",
    )
    assert decision.action is StrategyAction.NO_ACTION


def test_common_identity_and_cycle_metadata_enforce_typed_xor() -> None:
    identity_constraints = {
        constraint.name
        for constraint in cast(Table, PaperDecisionIdentityRecord.__table__).constraints
    }
    cycle_constraints = {
        constraint.name
        for constraint in cast(Table, PaperDecisionCycleRecord.__table__).constraints
    }
    assert "ck_paper_decision_identity_xor" in identity_constraints
    assert "uq_paper_decision_identity_baseline" in identity_constraints
    assert "uq_paper_decision_identity_ml" in identity_constraints
    assert "ck_paper_cycles_lifecycle" in cycle_constraints
    assert "fk_paper_cycles_baseline_identity_pair" in cycle_constraints
    assert "fk_paper_cycles_ml_identity_pair" in cycle_constraints
    assert MLDecisionSnapshotRecord.__table__.c.model_artifact_id.nullable is False


@pytest.mark.asyncio
async def test_ml_persistence_stages_snapshot_and_typed_identity_atomically() -> None:
    session = MagicMock()
    session.scalar = AsyncMock(return_value=None)
    session.flush = AsyncMock()
    identity = await persist_ml_decision(session, snapshot())
    records = [call.args[0] for call in session.add.call_args_list]
    assert isinstance(records[0], MLDecisionSnapshotRecord)
    assert isinstance(records[1], PaperDecisionIdentityRecord)
    assert identity.kind == "ml"
    assert identity.ml_snapshot_id == records[0].id
    assert identity.baseline_snapshot_id is None
    session.flush.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_baseline_persistence_uses_same_identity_without_ml_reference() -> None:
    session = MagicMock()
    session.scalar = AsyncMock(return_value=None)
    session.flush = AsyncMock()
    baseline_id = uuid4()
    identity = await persist_baseline_decision_identity(session, baseline_id)
    assert identity.kind == "baseline"
    assert identity.baseline_snapshot_id == baseline_id
    assert identity.ml_snapshot_id is None
    session.flush.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_paper_execution_accepts_persisted_ml_no_action_without_score() -> None:
    decision = snapshot(
        signed_prediction=Decimal("0.1"),
        action=StrategyAction.NO_ACTION,
        requested_target_portfolio_percentage=None,
        applied_rule="ml_entry_threshold_not_met",
    )
    run = SimpleNamespace(
        id=uuid4(),
        started_at=NOW - timedelta(hours=1),
        configuration_version=1,
        account_currency="USDT",
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    )
    persisted = SimpleNamespace(
        asset=decision.asset.value,
        provider=decision.provider.value,
        market_type=decision.market_type.value,
        interval=decision.interval.value,
        action=decision.action.value,
        source_market_price=decision.source_market_price,
    )
    session = AsyncMock()
    session.scalar.side_effect = [run, persisted]
    result = await execute_decision(
        session=session,
        decision=decision,
        executed_at=NOW,
        run_id=run.id,
    )
    assert isinstance(result, NoTrade)
    assert session.scalar.await_count == 2
