from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np

from app.ml.contract import DEVELOPMENT_END, QUALIFICATION_END
from app.ml.dataset import MLExperimentDataset
from app.ml.gaps import datetime_to_ns
from app.ml.protected.validation import _candidate
from app.ml.refinement.calculation import _fit_predict, _selected_columns
from app.ml.refinement.models import MLRefinementEvaluationRecord

from .contracts import CAMPAIGN_FEE_RATE, CAMPAIGN_SLIPPAGE_RATE
from .economic_v2_calculation import evaluate_strategy_policy
from .economic_v2_policy import (
    StrategyPolicy,
    enforce_v2_market_boundary,
    failed_qualification_rules,
)


@dataclass(frozen=True, slots=True)
class QualificationResult:
    temporal_manifest: dict[str, object]
    temporal_manifest_fingerprint: str
    scenario_metrics: dict[str, object]
    quarter_returns: tuple[Decimal, Decimal, Decimal, Decimal]
    trades: tuple[dict[str, object], ...]
    failed_rules: tuple[str, ...]


def _compound(values: tuple[Decimal, ...]) -> Decimal:
    result = Decimal(1)
    for value in values:
        result *= Decimal(1) + value
    return result - Decimal(1)


def _period_return(
    equity_curve: tuple[tuple[datetime, Decimal], ...],
    *,
    start: datetime,
    end: datetime,
) -> Decimal:
    before = tuple(value for boundary, value in equity_curve if boundary < start)
    during = tuple(value for boundary, value in equity_curve if start <= boundary < end)
    if not during:
        return Decimal(0)
    starting_equity = before[-1] if before else equity_curve[0][1]
    return during[-1] / starting_equity - Decimal(1)


def calculate_qualification(
    *,
    dataset: MLExperimentDataset,
    evaluation: MLRefinementEvaluationRecord,
    policy: StrategyPolicy,
    deterministic_seed: int,
) -> QualificationResult:
    enforce_v2_market_boundary(
        start=dataset.metadata.requested_start,
        end=dataset.metadata.requested_end,
    )
    if dataset.metadata.requested_end != QUALIFICATION_END:
        raise ValueError("qualification dataset must end exactly at 2025-01-01 UTC")
    candidate = _candidate(evaluation)
    target = dataset.targets.for_duration(
        timedelta(seconds=candidate.future_duration_seconds)
    )
    eligible = dataset.feature_eligible & target.eligible
    development = np.flatnonzero(
        eligible & (dataset.boundary_time_ns < datetime_to_ns(DEVELOPMENT_END))
    ).astype(np.int64)
    qualification = np.flatnonzero(
        eligible
        & (dataset.boundary_time_ns >= datetime_to_ns(DEVELOPMENT_END))
        & (dataset.boundary_time_ns < datetime_to_ns(QUALIFICATION_END))
    ).astype(np.int64)
    if not len(development) or not len(qualification):
        raise ValueError("qualification has incomplete temporal evidence")
    columns = _selected_columns(
        candidate, dataset.features[development], target.values[development]
    )
    training_scores, qualification_scores, _ = _fit_predict(
        candidate,
        dataset.features[np.ix_(development, columns)],
        target.values[development],
        dataset.features[np.ix_(qualification, columns)],
        float(2 * (CAMPAIGN_FEE_RATE + CAMPAIGN_SLIPPAGE_RATE)),
        deterministic_seed,
    )
    evidence = evaluate_strategy_policy(
        dataset=dataset,
        training=development,
        validation=qualification,
        training_scores=training_scores,
        validation_scores=qualification_scores,
        horizon_seconds=candidate.future_duration_seconds,
        policy=policy,
        fee_rate=CAMPAIGN_FEE_RATE,
        slippage_rate=CAMPAIGN_SLIPPAGE_RATE,
    )
    raw_trades = evidence["trades"]
    if not isinstance(raw_trades, list):
        raise ValueError("qualification trade evidence is malformed")
    trades = tuple(item for item in raw_trades if isinstance(item, dict))
    quarter_values: list[Decimal] = []
    for quarter in range(4):
        start_month = quarter * 3 + 1
        start = datetime(2024, start_month, 1, tzinfo=UTC)
        end = (
            datetime(2025, 1, 1, tzinfo=UTC)
            if quarter == 3
            else datetime(2024, start_month + 3, 1, tzinfo=UTC)
        )
        raw_curve = evidence["equity_curve"]
        if not isinstance(raw_curve, list):
            raise ValueError("qualification equity evidence is malformed")
        equity_curve = tuple(
            (datetime.fromisoformat(str(item[0])), Decimal(str(item[1])))
            for item in raw_curve
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
        quarter_values.append(_period_return(equity_curve, start=start, end=end))
    quarters = (
        quarter_values[0],
        quarter_values[1],
        quarter_values[2],
        quarter_values[3],
    )
    base_return = Decimal(str(evidence["net_return"]))
    return_1_5x = Decimal(str(evidence["return_1_5x"]))
    return_2x = Decimal(str(evidence["return_2x"]))
    maximum_drawdown = Decimal(str(evidence["maximum_drawdown"]))
    failed = failed_qualification_rules(
        round_trips=int(str(evidence["round_trips"])),
        base_return=base_return,
        return_1_5x=return_1_5x,
        return_2x=return_2x,
        maximum_drawdown=maximum_drawdown,
        quarter_returns=quarters,
    )
    manifest: dict[str, object] = {
        "development_start": dataset.metadata.requested_start.isoformat(),
        "development_end_exclusive": DEVELOPMENT_END.isoformat(),
        "qualification_start": DEVELOPMENT_END.isoformat(),
        "qualification_end_exclusive": QUALIFICATION_END.isoformat(),
        "development_rows": len(development),
        "qualification_rows": len(qualification),
        "selected_columns": [int(item) for item in columns],
        "dataset_fingerprint": dataset.metadata.fingerprint,
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return QualificationResult(
        manifest,
        fingerprint,
        {
            "base": {"total_net_return": str(base_return)},
            "stress_1_5x": {"total_net_return": str(return_1_5x)},
            "stress_2x": {"total_net_return": str(return_2x)},
            "maximum_drawdown": str(maximum_drawdown),
            "round_trips": int(str(evidence["round_trips"])),
        },
        quarters,
        trades,
        failed,
    )
