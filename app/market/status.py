from datetime import UTC, datetime, timedelta

from app.ingestion.time_windows import latest_closed_five_minute_open_time
from app.market.catalog import Instrument
from app.market.enums import Interval
from app.market.queries import MarketStatusAggregate
from app.market.schemas import MarketStatusResponse

_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class UnsupportedFreshnessIntervalError(ValueError):
    """Raised when freshness rules are not defined for a candle interval."""


def expected_latest_closed_open_time(*, interval: Interval, now: datetime) -> datetime:
    """Resolve the latest expected closed candle without implying interval support."""

    if interval is Interval.FIVE_MINUTES:
        return _UNIX_EPOCH + timedelta(
            milliseconds=latest_closed_five_minute_open_time(now)
        )
    raise UnsupportedFreshnessIntervalError(
        f"market freshness is not supported for interval {interval}"
    )


def build_market_statuses(
    *,
    instruments: tuple[Instrument, ...],
    aggregates: tuple[MarketStatusAggregate, ...],
    now: datetime,
) -> dict[Instrument, MarketStatusResponse]:
    """Compose status by complete requested identity using one UTC clock."""

    by_identity = {
        Instrument(
            aggregate.asset,
            aggregate.provider,
            aggregate.market_type,
            aggregate.interval,
        ): aggregate
        for aggregate in aggregates
    }
    statuses: dict[Instrument, MarketStatusResponse] = {}
    for instrument in instruments:
        expected_open_time = expected_latest_closed_open_time(
            interval=instrument.interval, now=now
        )
        aggregate = by_identity.get(instrument)
        if aggregate is None:
            statuses[instrument] = MarketStatusResponse(
                asset=instrument.asset,
                interval=instrument.interval,
                candle_count=0,
                latest_open_time=None,
                latest_close_time=None,
                latest_ingested_at=None,
                age_seconds=None,
                is_fresh=False,
            )
            continue
        age = now - aggregate.latest_close_time
        statuses[instrument] = MarketStatusResponse(
            asset=instrument.asset,
            interval=instrument.interval,
            candle_count=aggregate.candle_count,
            latest_open_time=aggregate.latest_open_time,
            latest_close_time=aggregate.latest_close_time,
            latest_ingested_at=aggregate.latest_ingested_at,
            age_seconds=age.days * 86_400 + age.seconds,
            is_fresh=aggregate.latest_open_time >= expected_open_time,
        )
    return statuses
