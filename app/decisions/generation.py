from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.models import DecisionSnapshot
from app.decisions.persistence import (
    ScoreEvidence,
    get_latest_score_evidence,
    insert_decision_snapshot,
)
from app.market.catalog import enabled_decision_instruments
from app.market.enums import Asset, Interval, MarketType, Provider
from app.strategies.baseline_long_only import (
    CONFIGURATION_VERSION,
    REQUIRED_SCORE_MODEL_NAME,
    REQUIRED_SCORE_MODEL_VERSION,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    evaluate_baseline_long_only,
)


class DecisionGenerationError(ValueError):
    """Raised when persisted evidence cannot support a decision."""


class PortfolioStateSourceUnavailableError(DecisionGenerationError):
    """Raised when no real portfolio-state source is configured."""


@dataclass(frozen=True, slots=True)
class PortfolioState:
    position_is_open: bool


class PortfolioStateSource(Protocol):
    async def get_portfolio_state(
        self,
        *,
        asset: Asset,
        provider: Provider,
        market_type: MarketType,
        interval: Interval,
    ) -> PortfolioState: ...


@dataclass(frozen=True, slots=True)
class DecisionGenerationResult:
    snapshot: DecisionSnapshot
    calculated: int
    inserted: int
    duplicates: int

    def __post_init__(self) -> None:
        if self.calculated != 1:
            raise ValueError("one decision must be calculated")
        if self.inserted + self.duplicates != self.calculated:
            raise ValueError("inserted plus duplicates must equal calculated")


async def generate_market_decision(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    portfolio_state: PortfolioState,
    decision_timestamp: datetime,
) -> DecisionGenerationResult:
    evidence = await get_latest_score_evidence(
        session=session,
        asset=asset,
        provider=provider,
        market_type=market_type,
        interval=interval,
    )
    if evidence is None:
        raise DecisionGenerationError("compatible persisted market score not found")
    return await generate_decision_from_evidence(
        session=session,
        evidence=evidence,
        portfolio_state=portfolio_state,
        decision_timestamp=decision_timestamp,
    )


async def generate_decision_from_evidence(
    *,
    session: AsyncSession,
    evidence: ScoreEvidence,
    portfolio_state: PortfolioState,
    decision_timestamp: datetime,
) -> DecisionGenerationResult:
    score = evidence.score
    if (
        score.model_name != REQUIRED_SCORE_MODEL_NAME
        or score.model_version != REQUIRED_SCORE_MODEL_VERSION
    ):
        raise DecisionGenerationError("persisted score is incompatible with strategy")
    decision = evaluate_baseline_long_only(
        overall_score=score.overall_score,
        position_is_open=portfolio_state.position_is_open,
    )
    snapshot = DecisionSnapshot(
        asset=score.asset,
        provider=score.provider,
        market_type=score.market_type,
        interval=score.interval,
        decision_timestamp=decision_timestamp,
        source_candle_open_time=score.candle_open_time,
        source_market_price=evidence.source_market_price,
        source_score_id=score.id,
        source_score_model_name=score.model_name,
        source_score_model_version=score.model_version,
        source_overall_score=score.overall_score,
        strategy_name=STRATEGY_NAME,
        strategy_version=STRATEGY_VERSION,
        configuration_version=CONFIGURATION_VERSION,
        action=decision.action,
        requested_target_portfolio_percentage=(
            decision.requested_target_portfolio_percentage
        ),
        requested_position_percentage=decision.requested_position_percentage,
        applied_rule=decision.applied_rule,
        explanation=decision.explanation,
        position_is_open=portfolio_state.position_is_open,
    )
    inserted = int(await insert_decision_snapshot(session, snapshot))
    return DecisionGenerationResult(
        snapshot=snapshot,
        calculated=1,
        inserted=inserted,
        duplicates=1 - inserted,
    )


async def generate_all_market_decisions(
    *,
    session: AsyncSession,
    portfolio_state_source: PortfolioStateSource,
    decision_timestamp: datetime,
) -> tuple[DecisionGenerationResult, ...]:
    results: list[DecisionGenerationResult] = []
    for instrument in enabled_decision_instruments():
        portfolio_state = await portfolio_state_source.get_portfolio_state(
            asset=instrument.asset,
            provider=instrument.provider,
            market_type=instrument.market_type,
            interval=instrument.interval,
        )
        results.append(
            await generate_market_decision(
                session=session,
                asset=instrument.asset,
                provider=instrument.provider,
                market_type=instrument.market_type,
                interval=instrument.interval,
                portfolio_state=portfolio_state,
                decision_timestamp=decision_timestamp,
            )
        )
    return tuple(results)
