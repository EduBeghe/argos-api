from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.generation import (
    DecisionGenerationError,
    PortfolioState,
    generate_decision_from_evidence,
)
from app.strategies.baseline_long_only import StrategyAction
from tests.decisions.factories import make_evidence

pytestmark = pytest.mark.asyncio
DECIDED_AT = datetime(2026, 7, 15, 12, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    (
        "score",
        "position_is_open",
        "action",
        "target_percentage",
        "position_percentage",
    ),
    [
        (
            Decimal("0.60"),
            False,
            StrategyAction.OPEN_POSITION,
            Decimal("15.00"),
            None,
        ),
        (
            Decimal("0.19"),
            True,
            StrategyAction.SELL_ALL,
            None,
            Decimal("100.00"),
        ),
        (Decimal("0.20"), True, StrategyAction.NO_ACTION, None, None),
    ],
)
async def test_generation_preserves_evidence_and_exact_sizing(
    score: Decimal,
    position_is_open: bool,
    action: StrategyAction,
    target_percentage: Decimal | None,
    position_percentage: Decimal | None,
) -> None:
    evidence = make_evidence(overall_score=score)
    session = AsyncMock(spec=AsyncSession)
    with patch(
        "app.decisions.generation.insert_decision_snapshot",
        AsyncMock(return_value=True),
    ):
        result = await generate_decision_from_evidence(
            session=cast(AsyncSession, session),
            evidence=evidence,
            portfolio_state=PortfolioState(position_is_open=position_is_open),
            decision_timestamp=DECIDED_AT,
        )

    snapshot = result.snapshot
    assert snapshot.action is action
    assert snapshot.requested_target_portfolio_percentage == target_percentage
    assert snapshot.requested_position_percentage == position_percentage
    assert snapshot.source_score_id == evidence.score.id
    assert snapshot.source_score_model_name == "baseline_technical"
    assert snapshot.source_score_model_version == 1
    assert snapshot.source_overall_score == score
    assert snapshot.source_market_price == Decimal("65000.123456789012345678")
    assert (result.calculated, result.inserted, result.duplicates) == (1, 1, 0)
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


async def test_duplicate_counter_uses_conflict_result() -> None:
    with patch(
        "app.decisions.generation.insert_decision_snapshot",
        AsyncMock(return_value=False),
    ):
        result = await generate_decision_from_evidence(
            session=cast(AsyncSession, AsyncMock(spec=AsyncSession)),
            evidence=make_evidence(),
            portfolio_state=PortfolioState(position_is_open=False),
            decision_timestamp=DECIDED_AT,
        )

    assert (result.calculated, result.inserted, result.duplicates) == (1, 0, 1)


async def test_incompatible_score_metadata_is_rejected_before_persistence() -> None:
    evidence = make_evidence()
    incompatible = evidence.score.model_copy(update={"model_version": 2})
    insert = AsyncMock(return_value=True)

    with (
        patch("app.decisions.generation.insert_decision_snapshot", insert),
        pytest.raises(DecisionGenerationError, match="incompatible"),
    ):
        await generate_decision_from_evidence(
            session=cast(AsyncSession, AsyncMock(spec=AsyncSession)),
            evidence=evidence.__class__(
                score=incompatible,
                source_market_price=evidence.source_market_price,
            ),
            portfolio_state=PortfolioState(position_is_open=False),
            decision_timestamp=DECIDED_AT,
        )

    insert.assert_not_awaited()
