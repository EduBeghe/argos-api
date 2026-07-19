from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.persistence import DecisionSnapshotRecord
from app.paper.equity_snapshots import PaperPositionSnapshotRecord
from app.paper.evaluation import (
    BenchmarkCandles,
    ClosedPositionEvidence,
    DecisionEvidence,
    MarketIdentity,
    OpenPositionEvidence,
    TradeEvidence,
    calculate_benchmark,
    comparison,
    instrument_evaluations,
    strategy_evaluation,
    unavailable_strategy,
)
from app.paper.evaluation_queries import (
    benchmark_candles,
    evaluation_closed_positions,
    evaluation_decisions,
    evaluation_trades,
    latest_snapshot_and_count,
)
from app.paper.performance import calculate_performance, get_performance_points
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradeRecord,
    PaperTradingRunRecord,
    run_domain,
)


async def evaluate_paper_run(
    *, session: AsyncSession, run: PaperTradingRunRecord
) -> dict[str, object]:
    snapshot, snapshot_count = await latest_snapshot_and_count(session, run_id=run.id)
    run_section = {
        "run_id": run.id,
        "status": run.status,
        "account_currency": run.account_currency,
        "configuration_version": run.configuration_version,
        "evaluated_from": run.started_at,
        "evaluated_until": snapshot.scheduled_boundary if snapshot else run.started_at,
        "latest_snapshot_id": snapshot.id if snapshot else None,
        "snapshot_count": snapshot_count,
    }
    if snapshot is None:
        strategy = unavailable_strategy(run.initial_cash)
        benchmark = calculate_benchmark(
            initial_equity=run.initial_cash,
            fee_rate=run.fee_rate,
            slippage_rate=run.slippage_rate,
            universe=(),
            candles={},
        )
        return {
            "run": run_section,
            "strategy": strategy,
            "benchmark": benchmark,
            "comparison": comparison(strategy, benchmark),
            "instruments": [],
        }

    boundary = snapshot.scheduled_boundary
    decision_records = await evaluation_decisions(
        session, run_id=run.id, evaluated_until=boundary
    )
    trade_records = await evaluation_trades(
        session, run_id=run.id, evaluated_until=boundary
    )
    closed_records = await evaluation_closed_positions(
        session, run_id=run.id, evaluated_until=boundary
    )
    decisions = tuple(
        DecisionEvidence(_identity(record), record.action)
        for record in decision_records
    )
    trades = tuple(
        TradeEvidence(
            _identity(record),
            record.side,
            record.observed_market_price,
            record.execution_price,
            record.quantity,
            record.fee_amount,
        )
        for record in trade_records
    )
    closed_positions = tuple(
        ClosedPositionEvidence(
            _identity(record),
            _required(record.gross_realized_pnl),
            _required(record.net_realized_pnl),
        )
        for record in closed_records
    )
    open_positions = tuple(
        OpenPositionEvidence(_identity(record), record.unrealized_pnl)
        for record in snapshot.positions
    )
    performance_points = await get_performance_points(
        session, run_id=run.id, evaluated_until=boundary
    )
    performance = calculate_performance(
        run=run_domain(run),
        snapshots=tuple(
            point
            for point in performance_points
            if point.scheduled_boundary <= boundary
        ),
    )
    strategy = strategy_evaluation(
        performance=performance,
        decisions=decisions,
        trades=trades,
        closed_positions=closed_positions,
        open_positions=open_positions,
    )
    universe = tuple(sorted({item.identity for item in decisions}))
    evidence: dict[MarketIdentity, BenchmarkCandles] = {}
    for identity in universe:
        entry, ending = await benchmark_candles(
            session,
            identity=identity,
            started_at=run.started_at,
            evaluated_until=boundary,
        )
        evidence[identity] = BenchmarkCandles(entry, ending)
    benchmark = calculate_benchmark(
        initial_equity=run.initial_cash,
        fee_rate=run.fee_rate,
        slippage_rate=run.slippage_rate,
        universe=universe,
        candles=evidence,
    )
    return {
        "run": run_section,
        "strategy": strategy,
        "benchmark": benchmark,
        "comparison": comparison(strategy, benchmark),
        "instruments": instrument_evaluations(
            decisions=decisions,
            trades=trades,
            closed_positions=closed_positions,
            open_positions=open_positions,
        ),
    }


def _identity(
    record: DecisionSnapshotRecord
    | PaperTradeRecord
    | PaperPositionRecord
    | PaperPositionSnapshotRecord,
) -> MarketIdentity:
    return MarketIdentity(
        asset=record.asset,
        provider=record.provider,
        market_type=record.market_type,
        interval=record.interval,
    )


def _required(value: Decimal | None) -> Decimal:
    if value is None:
        raise ValueError("closed Position is missing persisted realized P&L")
    return value
