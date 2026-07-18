from dataclasses import asdict
from datetime import datetime
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.dashboard.queries import (
    get_latest_decisions_batch,
    get_latest_scores_batch,
    get_open_positions_batch,
)
from app.dashboard.schemas import (
    DashboardInstrumentResponse,
    DashboardOverviewResponse,
    DashboardPaperResponse,
    DashboardSystemResponse,
)
from app.decisions.schemas import DecisionSnapshotResponse
from app.market.api import build_market_status
from app.market.catalog import (
    binance_spot_five_minute_ingestion_instruments,
    enabled_decision_instruments,
    enabled_scoring_instruments,
    market_instruments,
)
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.queries import get_market_status_aggregates
from app.market.schemas import MarketInstrumentResponse
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.paper.performance import calculate_performance, get_performance_points
from app.paper.persistence import get_active_run, run_domain
from app.paper.portfolio import value_paper_run
from app.paper.schemas import (
    PaperAccountResponse,
    PaperPerformanceResponse,
    PaperPortfolioResponse,
    PaperPositionResponse,
)
from app.scores.schemas import MarketScoreResponse


class Identity(Protocol):
    @property
    def asset(self) -> Asset: ...

    @property
    def provider(self) -> Provider: ...

    @property
    def market_type(self) -> MarketType: ...

    @property
    def interval(self) -> Interval: ...


def _identity(
    value: Identity,
) -> tuple[Asset, Provider, MarketType, Interval]:
    return (
        value.asset,
        value.provider,
        value.market_type,
        value.interval,
    )


async def build_dashboard_overview(
    *, session: AsyncSession, generated_at: datetime
) -> DashboardOverviewResponse:
    definitions = market_instruments()
    ingestion = binance_spot_five_minute_ingestion_instruments()
    first = ingestion[0]
    statuses = build_market_status(
        aggregates=await get_market_status_aggregates(
            session=session,
            provider=first.provider,
            market_type=first.market_type,
            interval=first.interval,
        ),
        now=generated_at,
    )
    scores = await get_latest_scores_batch(
        session=session, instruments=enabled_scoring_instruments()
    )
    decisions = await get_latest_decisions_batch(
        session=session, instruments=enabled_decision_instruments()
    )
    run = await get_active_run(session, currency=DEFAULT_ACCOUNT_CURRENCY)
    positions = (
        await get_open_positions_batch(
            session=session,
            run_id=run.id,
            instruments=enabled_decision_instruments(),
        )
        if run is not None
        else ()
    )
    ingestion_by_asset = {item.asset: item for item in ingestion}
    status_by_identity = {
        _identity(ingestion_by_asset[item.asset]): item
        for item in statuses
        if item.asset in ingestion_by_asset
    }
    score_by_identity = {_identity(item): item for item in scores}
    decision_by_identity = {_identity(item): item for item in decisions}
    position_by_identity = {
        (item.asset, item.provider, item.market_type, item.interval): item
        for item in positions
    }

    paper = DashboardPaperResponse(run=None, portfolio=None, performance=None)
    if run is not None:
        domain_run = run_domain(run)
        performance = calculate_performance(
            run=domain_run,
            snapshots=await get_performance_points(session, run_id=run.id),
        )
        paper = DashboardPaperResponse(
            run=PaperAccountResponse.from_domain(domain_run),
            portfolio=PaperPortfolioResponse.model_validate(
                await value_paper_run(session=session, run=run, valued_at=generated_at)
            ),
            performance=PaperPerformanceResponse.model_validate(
                {
                    "run_id": run.id,
                    "status": run.status,
                    "account_currency": run.account_currency,
                    **asdict(performance),
                }
            ),
        )

    cards: list[DashboardInstrumentResponse] = []
    for definition in definitions:
        identity = _identity(definition)
        score = score_by_identity.get(identity)
        decision = decision_by_identity.get(identity)
        position = position_by_identity.get(
            (
                definition.asset.value,
                definition.provider.value,
                definition.market_type.value,
                definition.interval.value,
            )
        )
        cards.append(
            DashboardInstrumentResponse(
                instrument=MarketInstrumentResponse.from_definition(definition),
                market_status=(
                    status_by_identity.get(identity)
                    if definition.ingestion_enabled
                    else None
                ),
                latest_score=(
                    MarketScoreResponse.from_domain(score)
                    if definition.scoring_enabled and score
                    else None
                ),
                latest_decision=(
                    DecisionSnapshotResponse.from_domain(decision)
                    if definition.decision_enabled and decision
                    else None
                ),
                open_position=(
                    PaperPositionResponse.model_validate(position)
                    if definition.decision_enabled and position
                    else None
                ),
            )
        )
    fresh = sum(item.is_fresh for item in statuses)
    return DashboardOverviewResponse(
        generated_at=generated_at,
        system=DashboardSystemResponse(
            market_data_fresh=all(item.is_fresh for item in statuses),
            instruments_total=len(definitions),
            instruments_fresh=fresh,
        ),
        paper=paper,
        instruments=cards,
    )

