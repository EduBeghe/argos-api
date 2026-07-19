from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.decisions.generation import (
    PortfolioState,
    generate_decision_from_evidence,
)
from app.decisions.persistence import (
    DecisionSnapshotRecord,
    ScoreEvidence,
    get_latest_decision_snapshot,
    list_decision_snapshots,
)
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.history import HistoryOrder
from app.scores.persistence import MarketScoreRecord, insert_market_scores
from app.strategies.baseline_long_only import StrategyAction
from tests.decisions.factories import make_score

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
DECIDED_AT = datetime(2026, 7, 15, 12, 30, tzinfo=UTC)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        await database_session.execute(delete(DecisionSnapshotRecord))
        await database_session.execute(delete(MarketScoreRecord))
        await database_session.commit()
        yield database_session
        await database_session.rollback()
        await database_session.execute(delete(DecisionSnapshotRecord))
        await database_session.execute(delete(MarketScoreRecord))
        await database_session.commit()
    await engine.dispose()


async def persist_evidence(
    session: AsyncSession,
    *,
    score_value: Decimal,
    candle_offset: int = 0,
) -> ScoreEvidence:
    score = make_score(overall_score=score_value).model_copy(
        update={
            "id": uuid4(),
            "candle_open_time": datetime(2026, 7, 15, 12, tzinfo=UTC)
            + timedelta(minutes=5 * candle_offset),
        }
    )
    assert await insert_market_scores(session, [score]) == 1
    return ScoreEvidence(
        score=score,
        source_market_price=Decimal("65000.123456789012345678"),
    )


@pytest.mark.parametrize(
    ("score_value", "position_is_open", "action", "target", "position"),
    [
        (
            Decimal("0.60"),
            False,
            StrategyAction.OPEN_POSITION,
            Decimal("15.000000000000000000"),
            None,
        ),
        (
            Decimal("0.19"),
            True,
            StrategyAction.SELL_ALL,
            None,
            Decimal("100.000000000000000000"),
        ),
        (Decimal("0.20"), True, StrategyAction.NO_ACTION, None, None),
    ],
)
async def test_snapshot_round_trip_preserves_intent_and_provenance(
    session: AsyncSession,
    score_value: Decimal,
    position_is_open: bool,
    action: StrategyAction,
    target: Decimal | None,
    position: Decimal | None,
) -> None:
    evidence = await persist_evidence(session, score_value=score_value)
    result = await generate_decision_from_evidence(
        session=session,
        evidence=evidence,
        portfolio_state=PortfolioState(position_is_open=position_is_open),
        decision_timestamp=DECIDED_AT,
    )
    await session.commit()

    record = await session.scalar(select(DecisionSnapshotRecord))
    assert record is not None
    assert StrategyAction(record.action) is action
    assert record.requested_target_portfolio_percentage == target
    assert record.requested_position_percentage == position
    assert record.source_score_id == evidence.score.id
    assert record.source_score_model_name == "baseline_technical"
    assert record.source_score_model_version == 1
    assert record.source_overall_score == score_value
    assert record.source_market_price == Decimal("65000.123456789012345678")
    assert result.inserted == 1


async def test_repeated_generation_is_conflict_safe_and_latest_is_ordered(
    session: AsyncSession,
) -> None:
    older = await persist_evidence(session, score_value=Decimal("0.60"))
    first = await generate_decision_from_evidence(
        session=session,
        evidence=older,
        portfolio_state=PortfolioState(position_is_open=False),
        decision_timestamp=DECIDED_AT,
    )
    duplicate = await generate_decision_from_evidence(
        session=session,
        evidence=older,
        portfolio_state=PortfolioState(position_is_open=False),
        decision_timestamp=DECIDED_AT + timedelta(minutes=30),
    )
    newer = await persist_evidence(
        session, score_value=Decimal("0.19"), candle_offset=6
    )
    latest_result = await generate_decision_from_evidence(
        session=session,
        evidence=newer,
        portfolio_state=PortfolioState(position_is_open=True),
        decision_timestamp=DECIDED_AT + timedelta(minutes=30),
    )
    await session.commit()

    latest = await get_latest_decision_snapshot(
        session=session,
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
    )
    assert (first.inserted, duplicate.duplicates, latest_result.inserted) == (1, 1, 1)
    assert latest is not None
    assert latest.source_score_id == newer.score.id
    assert latest.action is StrategyAction.SELL_ALL
    assert len((await session.scalars(select(DecisionSnapshotRecord.id))).all()) == 2
    history = await list_decision_snapshots(
        session=session,
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        run_id=None,
        from_time=DECIDED_AT,
        to_time=DECIDED_AT + timedelta(minutes=30),
        order=HistoryOrder.ASC,
        limit=100,
    )
    assert [item.source_score_id for item in history] == [
        older.score.id,
        newer.score.id,
    ]


async def test_database_rejects_invalid_action_sizing(session: AsyncSession) -> None:
    evidence = await persist_evidence(session, score_value=Decimal("0.60"))
    await session.commit()
    generated = await generate_decision_from_evidence(
        session=session,
        evidence=evidence,
        portfolio_state=PortfolioState(position_is_open=False),
        decision_timestamp=DECIDED_AT,
    )
    await session.rollback()
    invalid = generated.snapshot.model_dump(mode="python")
    invalid["id"] = uuid4()
    invalid["action"] = StrategyAction.SELL_ALL.value

    with pytest.raises(IntegrityError):
        await session.execute(insert(DecisionSnapshotRecord).values(invalid))
        await session.flush()
    await session.rollback()


async def test_caller_rollback_removes_staged_snapshot(session: AsyncSession) -> None:
    evidence = await persist_evidence(session, score_value=Decimal("0.60"))
    await generate_decision_from_evidence(
        session=session,
        evidence=evidence,
        portfolio_state=PortfolioState(position_is_open=False),
        decision_timestamp=DECIDED_AT,
    )
    await session.rollback()

    assert await session.scalar(select(DecisionSnapshotRecord.id)) is None
