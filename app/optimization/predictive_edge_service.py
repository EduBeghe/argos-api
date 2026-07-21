import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.optimization.predictive_edge_memory import (
    PredictiveEdgeAggregateRecord,
    PredictiveEdgeBenchmarkRecord,
    PredictiveEdgeBinAggregateRecord,
    PredictiveEdgeRunRecord,
    PredictiveEdgeWindowMetricRecord,
    PredictiveEdgeWindowRecord,
    StrategyHypothesisRecord,
)
from app.optimization.predictive_edges import (
    PREDICTIVE_EDGE_ALGORITHM_VERSION,
    PredictiveEdgeConfiguration,
    PredictiveEdgeResult,
)
from app.optimization.walk_forward import WalkForwardWindow
from app.scores.calculation import MODEL_NAME, MODEL_VERSION


@dataclass(frozen=True, slots=True)
class PredictiveEdgeExecution:
    run_id: UUID | None
    reused_existing_experiment: bool
    result: PredictiveEdgeResult | None
    metrics_reused: int
    compute_avoided_estimate: int


def predictive_edge_experiment_key(
    *,
    dataset_fingerprint: str,
    instrument: Instrument,
    range_start: datetime,
    range_end: datetime,
    configuration: PredictiveEdgeConfiguration,
    fee_rate: Decimal,
    slippage_rate: Decimal,
) -> str:
    payload = {
        "algorithm_version": PREDICTIVE_EDGE_ALGORITHM_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "instrument": [
            instrument.asset.value,
            instrument.provider.value,
            instrument.market_type.value,
            instrument.interval.value,
        ],
        "range": [range_start.isoformat(), range_end.isoformat()],
        "predictors": "baseline_features_and_baseline_technical_v1",
        "score_model": [MODEL_NAME, MODEL_VERSION],
        "configuration": asdict(configuration),
        "fee_rate": str(fee_rate),
        "slippage_rate": str(slippage_rate),
    }
    encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


async def find_reusable_predictive_edge_run(
    session: AsyncSession, experiment_key: str
) -> PredictiveEdgeExecution | None:
    record = await session.scalar(
        select(PredictiveEdgeRunRecord).where(
            PredictiveEdgeRunRecord.experiment_key == experiment_key,
            PredictiveEdgeRunRecord.status == "completed",
        )
    )
    if record is None:
        return None
    await session.execute(
        update(PredictiveEdgeRunRecord)
        .where(PredictiveEdgeRunRecord.id == record.id)
        .values(reuse_count=PredictiveEdgeRunRecord.reuse_count + 1)
    )
    await session.commit()
    return PredictiveEdgeExecution(
        record.id,
        True,
        None,
        record.aggregate_metrics_computed,
        record.compute_avoided_estimate,
    )


async def persist_predictive_edge_result(
    *,
    session: AsyncSession,
    experiment_key: str,
    instrument: Instrument,
    range_start: datetime,
    range_end: datetime,
    dataset_fingerprint: str,
    windows: tuple[WalkForwardWindow, ...],
    result: PredictiveEdgeResult,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    started_at: datetime,
) -> PredictiveEdgeExecution:
    run_id = uuid4()
    configuration = result.configuration
    neutral = configuration.neutral_threshold
    if neutral is None:
        neutral = Decimal(2) * (fee_rate + slippage_rate)
    run = PredictiveEdgeRunRecord(
        id=run_id,
        experiment_key=experiment_key,
        status="completed",
        asset=instrument.asset.value,
        provider=instrument.provider.value,
        market_type=instrument.market_type.value,
        interval=instrument.interval.value,
        dataset_fingerprint=dataset_fingerprint,
        range_start=range_start,
        range_end=range_end,
        algorithm_version=PREDICTIVE_EDGE_ALGORITHM_VERSION,
        score_model_name=MODEL_NAME,
        score_model_version=MODEL_VERSION,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        neutral_threshold=neutral,
        decision_interval_seconds=int(configuration.decision_interval.total_seconds()),
        training_days=configuration.windows.training_days,
        validation_days=configuration.windows.validation_days,
        test_days=configuration.windows.test_days,
        step_days=configuration.windows.step_days,
        maximum_windows=configuration.windows.maximum_windows,
        bin_count=configuration.bins,
        minimum_observations=configuration.thresholds.minimum_observations,
        minimum_windows=configuration.thresholds.minimum_windows,
        minimum_stable_direction_ratio=configuration.thresholds.minimum_stable_direction_ratio,
        minimum_absolute_correlation=configuration.thresholds.minimum_absolute_correlation,
        minimum_directional_hit_rate=configuration.thresholds.minimum_directional_hit_rate,
        windows_generated=len(windows),
        aggregate_metrics_computed=len(result.window_edges),
        metrics_reused=0,
        compute_avoided_estimate=0,
        reuse_count=0,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        failed_at=None,
        failure_reason=None,
    )
    session.add(run)
    window_ids: dict[int, UUID] = {}
    for window in windows:
        identifier = uuid4()
        window_ids[window.ordinal] = identifier
        session.add(
            PredictiveEdgeWindowRecord(
                id=identifier,
                run_id=run_id,
                ordinal=window.ordinal,
                training_from=window.training_from,
                training_to=window.training_to,
                validation_from=window.validation_from,
                validation_to=window.validation_to,
                test_from=window.test_from,
                test_to=window.test_to,
                usable_observations=window.decision_count,
                selection_fingerprint=window.selection_fingerprint,
                evidence_fingerprint=window.dataset_fingerprint,
            )
        )
    for stability_edge in result.stability_edges:
        session.add(
            PredictiveEdgeAggregateRecord(
                id=uuid4(),
                run_id=run_id,
                predictor_name=stability_edge.predictor.name,
                predictor_version=stability_edge.predictor.version,
                predictor_source=stability_edge.predictor.source,
                direction_convention=stability_edge.predictor.direction_convention,
                explanation=stability_edge.predictor.explanation,
                horizon_seconds=int(stability_edge.horizon.total_seconds()),
                condition=stability_edge.condition.value,
                windows_evaluated=stability_edge.windows_evaluated,
                sufficient_windows=stability_edge.sufficient_windows,
                total_usable_observations=stability_edge.total_usable_observations,
                stable_direction_ratio=stability_edge.stable_direction_ratio,
                weighted_mean_correlation=stability_edge.weighted_mean_correlation,
                median_correlation=stability_edge.median_correlation,
                worst_correlation=stability_edge.worst_correlation,
                mean_directional_hit_rate=stability_edge.mean_directional_hit_rate,
                mean_positive_net_ratio=stability_edge.mean_positive_net_ratio,
                mean_net_return=stability_edge.mean_net_return,
                correlation_dispersion=stability_edge.correlation_dispersion,
                classification=stability_edge.classification.value,
            )
        )
    for edge in result.window_edges:
        metrics = edge.metrics
        metric_id = uuid4()
        session.add(
            PredictiveEdgeWindowMetricRecord(
                id=metric_id,
                run_id=run_id,
                window_id=window_ids[edge.window_ordinal],
                predictor_name=edge.predictor.name,
                horizon_seconds=int(edge.horizon.total_seconds()),
                condition=edge.condition.value,
                usable_count=metrics.usable_count,
                missing_count=metrics.missing_count,
                correlation=metrics.correlation,
                directional_hit_rate=metrics.directional_hit_rate,
                mean_gross_return=metrics.mean_gross_return,
                mean_net_return=metrics.mean_net_return,
                positive_net_ratio=metrics.positive_net_ratio,
                standard_error=metrics.standard_error,
                evidence_status=metrics.evidence_status.value,
                insufficient_reason=metrics.insufficient_reason,
                training_bin_bounds=",".join(
                    str(value) for value in edge.training_bin_bounds
                ),
                volatility_threshold=edge.volatility_threshold,
                evidence_fingerprint=edge.fingerprint,
            )
        )
        for bin_value in edge.bins:
            session.add(
                PredictiveEdgeBinAggregateRecord(
                    id=uuid4(),
                    window_metric_id=metric_id,
                    ordinal=bin_value.ordinal,
                    lower_bound=bin_value.lower_bound,
                    upper_bound=bin_value.upper_bound,
                    observation_count=bin_value.observation_count,
                    mean_forward_return=bin_value.mean_forward_return,
                    median_forward_return=bin_value.median_forward_return,
                    up_count=bin_value.up_count,
                    down_count=bin_value.down_count,
                    neutral_count=bin_value.neutral_count,
                    positive_net_ratio=bin_value.positive_net_ratio,
                    mean_net_return=bin_value.mean_net_return,
                    evidence_status=bin_value.evidence_status.value,
                )
            )
    for value in result.hypotheses:
        session.add(
            StrategyHypothesisRecord(
                id=uuid4(),
                hypothesis_id=value.hypothesis_id,
                run_id=run_id,
                predictor_name=value.predictor_name,
                predictor_version=value.predictor_version,
                direction=value.direction,
                horizon_seconds=int(value.horizon.total_seconds()),
                condition=value.condition.value,
                expected_gross_return=value.expected_gross_return,
                expected_net_return=value.expected_net_return,
                directional_hit_rate=value.directional_hit_rate,
                stability_ratio=value.stability_ratio,
                evaluation_interval_seconds=int(
                    value.minimum_evaluation_interval.total_seconds()
                ),
                holding_period_from_seconds=int(
                    value.minimum_holding_period_from.total_seconds()
                ),
                holding_period_to_seconds=int(
                    value.minimum_holding_period_to.total_seconds()
                ),
                evidence_status=value.evidence_status.value,
                rationale=value.rationale,
                limitations=value.limitations,
            )
        )
    for benchmark in result.benchmarks:
        session.add(
            PredictiveEdgeBenchmarkRecord(
                id=uuid4(),
                run_id=run_id,
                horizon_seconds=int(benchmark.horizon.total_seconds()),
                benchmark=benchmark.benchmark,
                observation_count=benchmark.observation_count,
                directional_hit_rate=benchmark.directional_hit_rate,
                mean_return=benchmark.mean_return,
            )
        )
    await session.commit()
    return PredictiveEdgeExecution(run_id, False, result, 0, 0)

