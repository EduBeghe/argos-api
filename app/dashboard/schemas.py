from datetime import datetime

from pydantic import BaseModel

from app.decisions.schemas import DecisionSnapshotResponse
from app.market.schemas import (
    MarketInstrumentResponse,
    MarketStatusResponse,
)
from app.paper.schemas import (
    PaperAccountResponse,
    PaperPerformanceResponse,
    PaperPortfolioResponse,
    PaperPositionResponse,
)
from app.scores.schemas import MarketScoreResponse


class DashboardSystemResponse(BaseModel):
    market_data_fresh: bool
    instruments_total: int
    instruments_fresh: int


class DashboardPaperResponse(BaseModel):
    run: PaperAccountResponse | None
    portfolio: PaperPortfolioResponse | None
    performance: PaperPerformanceResponse | None


class DashboardInstrumentResponse(BaseModel):
    instrument: MarketInstrumentResponse
    market_status: MarketStatusResponse | None
    latest_score: MarketScoreResponse | None
    latest_decision: DecisionSnapshotResponse | None
    open_position: PaperPositionResponse | None


class DashboardOverviewResponse(BaseModel):
    generated_at: datetime
    system: DashboardSystemResponse
    paper: DashboardPaperResponse
    instruments: list[DashboardInstrumentResponse]

