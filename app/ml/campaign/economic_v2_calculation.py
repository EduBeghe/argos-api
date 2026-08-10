from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from statistics import median

import numpy as np
from numpy.typing import NDArray

from app.ml.dataset import MLExperimentDataset
from app.ml.gaps import ns_to_datetime
from app.ml.protected.simulation import (
    ProtectedPrediction,
    simulate_buy_and_hold,
    simulate_fixed_horizon_long_only,
)
from app.ml.refinement.contracts import EconomicPolicySearchConfiguration
from app.paper.config import SimulationConfiguration

from .economic_v2_policy import StrategyPolicy, strategy_policy_grid


def _compound(values: NDArray[np.float64]) -> Decimal:
    result = Decimal(1)
    for value in values:
        result *= Decimal(1) + Decimal(str(float(value)))
    return result - Decimal(1)


def _maximum_drawdown(values: NDArray[np.float64]) -> Decimal:
    equity = Decimal(1)
    peak = equity
    drawdown = Decimal(0)
    for value in values:
        equity *= Decimal(1) + Decimal(str(float(value)))
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak)
    return drawdown


def _non_overlapping_rows(
    rows: NDArray[np.int64], boundaries: NDArray[np.int64], holding_seconds: int
) -> NDArray[np.int64]:
    selected: list[int] = []
    next_boundary = -1
    holding_ns = holding_seconds * 1_000_000_000
    for row in rows:
        boundary = int(boundaries[int(row)])
        if boundary >= next_boundary:
            selected.append(int(row))
            next_boundary = boundary + holding_ns
    return np.asarray(selected, dtype=np.int64)


def evaluate_strategy_policy(
    *,
    dataset: MLExperimentDataset,
    training: NDArray[np.int64],
    validation: NDArray[np.int64],
    training_scores: NDArray[np.float64],
    validation_scores: NDArray[np.float64],
    horizon_seconds: int,
    policy: StrategyPolicy,
    fee_rate: Decimal,
    slippage_rate: Decimal,
) -> dict[str, object]:
    cutoff = Decimal(
        str(
            float(
                np.quantile(
                    np.abs(training_scores),
                    1.0 - float(policy.confidence_fraction),
                )
            )
        )
    )
    eligible = validation_scores >= float(cutoff)
    context = dataset.market_context
    volatility_lower: Decimal | None = None
    volatility_upper: Decimal | None = None
    if policy.regime_policy == "trend_aligned":
        eligible &= np.isfinite(context.ema_200[validation])
        eligible &= context.close[validation] > context.ema_200[validation]
    elif policy.regime_policy == "volatility_middle_60":
        training_volatility = context.realized_volatility_12h[training]
        training_volatility = training_volatility[np.isfinite(training_volatility)]
        if not len(training_volatility):
            eligible &= False
        else:
            lower, upper = np.quantile(training_volatility, (0.2, 0.8))
            volatility_lower = Decimal(str(float(lower)))
            volatility_upper = Decimal(str(float(upper)))
            validation_volatility = context.realized_volatility_12h[validation]
            eligible &= np.isfinite(validation_volatility)
            eligible &= validation_volatility >= lower
            eligible &= validation_volatility <= upper
    elif policy.regime_policy != "all_regimes":
        raise ValueError("unsupported economic v2 regime policy")
    candidate_rows = validation[np.flatnonzero(eligible)]
    holding_seconds = policy.holding_candles * 300
    finite = np.isfinite(context.close[validation])
    eligible_rows = {int(item) for item in candidate_rows}
    predictions = tuple(
        ProtectedPrediction(
            ns_to_datetime(int(dataset.boundary_time_ns[int(row)])),
            Decimal(str(float(validation_scores[index])))
            if int(row) in eligible_rows
            else Decimal("-2"),
            Decimal(str(float(context.close[int(row)]))),
        )
        for index, row in enumerate(validation)
        if finite[index]
    )
    if len(predictions) < 2:
        raise ValueError("strategy policy fold has insufficient market evidence")
    results = tuple(
        simulate_fixed_horizon_long_only(
            predictions=predictions,
            instrument=dataset.metadata.instrument,
            threshold=cutoff,
            hold=timedelta(seconds=holding_seconds),
            configuration=SimulationConfiguration(
                fee_rate=fee_rate * multiplier,
                slippage_rate=slippage_rate * multiplier,
            ),
        )
        for multiplier in (Decimal(1), Decimal("1.5"), Decimal(2))
    )
    benchmark = simulate_buy_and_hold(
        predictions=predictions,
        instrument=dataset.metadata.instrument,
        configuration=SimulationConfiguration(
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        ),
    )
    return {
        "policy_fingerprint": policy.fingerprint,
        "strategy_policy_version": policy.version,
        "confidence_fraction": str(policy.confidence_fraction),
        "confidence_cutoff": str(cutoff),
        "holding_multiplier": str(policy.holding_multiplier),
        "holding_candles": policy.holding_candles,
        "regime_policy": policy.regime_policy,
        "volatility_lower": (
            str(volatility_lower) if volatility_lower is not None else None
        ),
        "volatility_upper": (
            str(volatility_upper) if volatility_upper is not None else None
        ),
        "eligible_observations": len(candidate_rows),
        "round_trips": results[0].round_trips,
        "net_return": str(results[0].total_return),
        "buy_and_hold_return": str(benchmark.total_return),
        "maximum_drawdown": str(results[0].maximum_drawdown),
        "return_1_5x": str(results[1].total_return),
        "return_2x": str(results[2].total_return),
        "trades": [
            {
                "entry_at": trade.entry_at.isoformat(),
                "exit_at": trade.exit_at.isoformat(),
                "net_return": str(trade.net_return),
            }
            for trade in results[0].trades
        ],
        "equity_curve": [
            (boundary.isoformat(), str(equity))
            for boundary, equity in results[0].equity_curve
        ],
    }


def evaluate_policy_grid_fold(
    *,
    dataset: MLExperimentDataset,
    training: NDArray[np.int64],
    validation: NDArray[np.int64],
    training_scores: NDArray[np.float64],
    validation_scores: NDArray[np.float64],
    horizon_seconds: int,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    policy_configuration: EconomicPolicySearchConfiguration | None = None,
) -> tuple[dict[str, object], ...]:
    """Evaluate exactly the requested grid or the frozen campaign grid."""

    return tuple(
        evaluate_strategy_policy(
            dataset=dataset,
            training=training,
            validation=validation,
            training_scores=training_scores,
            validation_scores=validation_scores,
            horizon_seconds=horizon_seconds,
            policy=policy,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
        for policy in strategy_policy_grid(horizon_seconds, policy_configuration)
    )


def aggregate_policy_folds(
    folds: tuple[tuple[dict[str, object], ...], ...],
) -> tuple[dict[str, object], ...]:
    """Aggregate fold evidence by complete strategy-policy identity."""

    if not folds:
        return ()
    by_policy: dict[str, list[dict[str, object]]] = {}
    for fold in folds:
        for item in fold:
            by_policy.setdefault(str(item["policy_fingerprint"]), []).append(item)
    aggregates: list[dict[str, object]] = []
    for fingerprint, items in sorted(by_policy.items()):
        returns = tuple(Decimal(str(item["net_return"])) for item in items)
        returns_1_5x = np.asarray(
            [float(Decimal(str(item["return_1_5x"]))) for item in items]
        )
        returns_2x = np.asarray(
            [float(Decimal(str(item["return_2x"]))) for item in items]
        )
        first = items[0]
        aggregates.append(
            {
                "policy_fingerprint": fingerprint,
                "strategy_policy_version": first["strategy_policy_version"],
                "confidence_fraction": first["confidence_fraction"],
                "holding_multiplier": first["holding_multiplier"],
                "holding_candles": first["holding_candles"],
                "regime_policy": first["regime_policy"],
                "eligible_observations": sum(
                    int(str(item["eligible_observations"])) for item in items
                ),
                "round_trips": sum(int(str(item["round_trips"])) for item in items),
                "positive_net_folds": sum(value > 0 for value in returns),
                "folds_beating_buy_and_hold": sum(
                    Decimal(str(item["net_return"]))
                    > Decimal(str(item["buy_and_hold_return"]))
                    for item in items
                ),
                "median_net_return": str(median(returns)),
                "worst_net_return": str(min(returns)),
                "maximum_drawdown": str(
                    max(Decimal(str(item["maximum_drawdown"])) for item in items)
                ),
                "inactive_folds": sum(
                    int(str(item["round_trips"])) == 0 for item in items
                ),
                "compounded_return_1_5x": str(_compound(returns_1_5x)),
                "compounded_return_2x": str(_compound(returns_2x)),
            }
        )
    return tuple(aggregates)
