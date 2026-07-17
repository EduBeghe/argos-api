from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.generation import PortfolioState
from app.market.enums import Asset, Interval, MarketType, Provider
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.paper.persistence import get_active_run, get_open_position


class PaperPortfolioUnavailableError(RuntimeError):
    """Raised when no active persisted paper experiment exists."""


class PersistedPaperPortfolioStateSource:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_portfolio_state(
        self,
        *,
        asset: Asset,
        provider: Provider,
        market_type: MarketType,
        interval: Interval,
    ) -> PortfolioState:
        run = await get_active_run(self._session, currency=DEFAULT_ACCOUNT_CURRENCY)
        if run is None:
            raise PaperPortfolioUnavailableError("active Paper Trading run not found")
        position = await get_open_position(
            self._session,
            run_id=run.id,
            asset=asset,
            provider=provider,
            market_type=market_type,
            interval=interval,
        )
        return PortfolioState(position_is_open=position is not None)
