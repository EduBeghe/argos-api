import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.optimization.predictive_edge_engine_v4 import (
    PhaseWindowEvidence,
    PredictiveEdgeV4Result,
    TrainingWindowEvidence,
    decode_v4_cache_value,
    encode_v4_cache_value,
)
from app.optimization.predictive_edge_engine_v41 import (
    ECONOMIC_MARGIN,
    MINIMUM_HIT_IMPROVEMENT,
    PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
    PredictiveEdgeV41Result,
)
from app.optimization.predictive_edge_engine_v41 import (
    PhaseEvidence as V41PhaseEvidence,
)
from app.optimization.predictive_edge_memory import (
    PredictiveEdgeAggregateRecord,
    PredictiveEdgeBenchmarkRecord,
    PredictiveEdgeBinAggregateRecord,
    PredictiveEdgeNearMissRecord,
    PredictiveEdgeRunDiagnosticRecord,
    PredictiveEdgeRunRecord,
    PredictiveEdgeSensitivityRecord,
    PredictiveEdgeV4CacheRecord,
    PredictiveEdgeWindowMetricRecord,
    PredictiveEdgeWindowRecord,
    StrategyHypothesisRecord,
)
from app.optimization.predictive_edge_v4 import (
    DISCOVERY_BIN_COUNTS,
    PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
    RejectionReason,
)
from app.optimization.predictive_edges import (
    PREDICTIVE_EDGE_ALGORITHM_VERSION,
    PREDICTOR_VERSION,
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


def _v41_evidence(value: V41PhaseEvidence | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "observations": value.observations,
        "gross_movement": _decimal_text(value.gross_movement),
        "aligned_gross_movement": _decimal_text(value.aligned_gross_movement),
        "net_movement": _decimal_text(value.net_movement),
        "directional_hit_rate": _decimal_text(value.directional_hit_rate),
        "benchmark": _decimal_text(value.benchmark),
        "benchmark_improvement": _decimal_text(value.benchmark_improvement),
        "standard_error": _decimal_text(value.standard_error),
        "stable": value.stable,
        "economic": value.economic,
        "passes": value.passes,
    }


def _v41_failed_requirements(value: V41PhaseEvidence) -> list[str]:
    failed: list[str] = []
    if not value.economic:
        failed.append("movement_does_not_cover_costs")
    if not value.stable:
        failed.append("unstable_direction")
    if (value.benchmark_improvement or Decimal(0)) < MINIMUM_HIT_IMPROVEMENT:
        failed.append("benchmark_improvement_below_threshold")
    return failed


def _v41_missing_distance(value: V41PhaseEvidence) -> Decimal:
    economic = (
        max(Decimal(0), ECONOMIC_MARGIN - (value.net_movement or Decimal(0)))
        / ECONOMIC_MARGIN
    )
    stability = max(
        Decimal(0), Decimal("0.50") - (value.directional_hit_rate or Decimal(0))
    ) / Decimal("0.50")
    benchmark = (
        max(
            Decimal(0),
            MINIMUM_HIT_IMPROVEMENT - (value.benchmark_improvement or Decimal(0)),
        )
        / MINIMUM_HIT_IMPROVEMENT
    )
    return economic + stability + benchmark


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
            PredictiveEdgeRunRecord.status.in_(("completed", "insufficient_history")),
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
        status=result.conclusion.value,
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
        minimum_hit_rate_improvement_over_benchmark=(
            configuration.thresholds.minimum_hit_rate_improvement_over_benchmark
        ),
        windows_available=result.windows_available,
        windows_required=result.windows_required,
        conclusion_reason=result.reason,
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
    # Establish the parent row before SQLAlchemy batches the relational evidence.
    await session.flush()
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
                semantic_kind=stability_edge.predictor.semantic_kind.value,
                direction_convention=stability_edge.predictor.direction_convention,
                explanation=stability_edge.predictor.explanation,
                horizon_seconds=int(stability_edge.horizon.total_seconds()),
                condition=stability_edge.condition.value,
                direction_mapping=stability_edge.direction_mapping.value,
                bin_ordinal=stability_edge.bin_ordinal,
                lower_bound=stability_edge.lower_bound,
                upper_bound=stability_edge.upper_bound,
                suggested_action=stability_edge.suggested_action.value,
                supporting_bin_ordinals=list(stability_edge.supporting_bin_ordinals),
                windows_evaluated=stability_edge.windows_evaluated,
                windows_selected_for_test=stability_edge.windows_selected_for_test,
                windows_confirmed=stability_edge.windows_confirmed,
                total_usable_observations=stability_edge.total_usable_observations,
                stable_direction_ratio=stability_edge.stable_direction_ratio,
                weighted_mean_correlation=stability_edge.weighted_mean_correlation,
                median_correlation=stability_edge.median_correlation,
                worst_correlation=stability_edge.worst_correlation,
                mean_directional_hit_rate=stability_edge.mean_directional_hit_rate,
                mean_movement_after_friction_ratio=(
                    stability_edge.mean_movement_after_friction_ratio
                ),
                mean_net_movement_magnitude_after_friction=(
                    stability_edge.mean_net_movement_magnitude_after_friction
                ),
                mean_strategy_aligned_gross_return=(
                    stability_edge.mean_strategy_aligned_gross_return
                ),
                mean_strategy_aligned_net_return=(
                    stability_edge.mean_strategy_aligned_net_return
                ),
                mean_expected_avoided_downside=(
                    stability_edge.mean_expected_avoided_downside
                ),
                correlation_dispersion=stability_edge.correlation_dispersion,
                classification=stability_edge.classification.value,
            )
        )
    await session.flush()
    for edge in result.window_edges:
        validation = edge.validation_metrics
        test = edge.test_metrics
        metric_id = uuid4()
        session.add(
            PredictiveEdgeWindowMetricRecord(
                id=metric_id,
                run_id=run_id,
                window_id=window_ids[edge.window_ordinal],
                predictor_name=edge.predictor.name,
                semantic_kind=edge.predictor.semantic_kind.value,
                horizon_seconds=int(edge.horizon.total_seconds()),
                condition=edge.condition.value,
                training_center=edge.training_center,
                direction_mapping=edge.direction_mapping.value,
                validation_status=edge.validation_status.value,
                validation_direction=edge.validation_direction.value,
                validation_usable_count=validation.usable_count,
                validation_correlation=validation.correlation,
                validation_directional_hit_rate=validation.directional_hit_rate,
                validation_mean_strategy_aligned_net_return=(
                    validation.mean_strategy_aligned_net_return
                ),
                selected_for_test=edge.selected_for_test,
                selection_reason=edge.selection_reason,
                selected_bin_ordinals=list(edge.selected_bin_ordinals),
                test_confirmation_status=edge.test_confirmation_status.value,
                test_usable_count=test.usable_count,
                test_missing_count=test.missing_count,
                test_correlation=test.correlation,
                test_directional_hit_rate=test.directional_hit_rate,
                mean_forward_gross_return=test.mean_forward_gross_return,
                mean_absolute_movement=test.mean_absolute_movement,
                mean_net_movement_magnitude_after_friction=(
                    test.mean_net_movement_magnitude_after_friction
                ),
                positive_movement_after_friction_ratio=(
                    test.positive_movement_after_friction_ratio
                ),
                mean_strategy_aligned_gross_return=(
                    test.mean_strategy_aligned_gross_return
                ),
                mean_strategy_aligned_net_return=(
                    test.mean_strategy_aligned_net_return
                ),
                mean_expected_avoided_downside=test.mean_expected_avoided_downside,
                test_standard_error=test.standard_error,
                test_evidence_status=test.evidence_status.value,
                test_insufficient_reason=test.insufficient_reason,
                training_bin_bounds=",".join(
                    str(value) for value in edge.training_bin_bounds
                ),
                volatility_threshold=edge.volatility_threshold,
                evidence_fingerprint=edge.fingerprint,
            )
        )
        await session.flush()
        selections = {item.ordinal: item for item in edge.bin_selections}
        for bin_value in edge.bins:
            selection = selections[bin_value.ordinal]
            session.add(
                PredictiveEdgeBinAggregateRecord(
                    id=uuid4(),
                    window_metric_id=metric_id,
                    ordinal=bin_value.ordinal,
                    lower_bound=bin_value.lower_bound,
                    upper_bound=bin_value.upper_bound,
                    observation_count=bin_value.observation_count,
                    mean_forward_gross_return=bin_value.mean_forward_gross_return,
                    median_forward_gross_return=bin_value.median_forward_gross_return,
                    up_count=bin_value.up_count,
                    down_count=bin_value.down_count,
                    neutral_count=bin_value.neutral_count,
                    positive_movement_after_friction_ratio=(
                        bin_value.positive_movement_after_friction_ratio
                    ),
                    mean_net_movement_magnitude_after_friction=(
                        bin_value.mean_net_movement_magnitude_after_friction
                    ),
                    evidence_status=bin_value.evidence_status.value,
                    suggested_action=selection.suggested_action.value,
                    validation_observation_count=(
                        selection.validation_observation_count
                    ),
                    validation_gross_return=selection.validation_gross_return,
                    validation_long_aligned_net_return=(
                        selection.validation_long_aligned_net_return
                    ),
                    validation_avoided_downside=(selection.validation_avoided_downside),
                    validation_directional_hit_rate=(
                        selection.validation_directional_hit_rate
                    ),
                    selected_for_test=selection.selected_for_test,
                    test_observation_count=selection.test_observation_count,
                    test_confirmation_status=(selection.test_confirmation_status.value),
                    test_gross_return=selection.test_gross_return,
                    test_long_aligned_net_return=(
                        selection.test_long_aligned_net_return
                    ),
                    test_avoided_downside=selection.test_avoided_downside,
                    test_directional_hit_rate=selection.test_directional_hit_rate,
                    rejection_reason=selection.rejection_reason,
                )
            )
    for value in result.hypotheses:
        hypothesis_id = hashlib.sha256(
            "|".join(
                (
                    instrument.asset.value,
                    instrument.provider.value,
                    instrument.market_type.value,
                    instrument.interval.value,
                    value.hypothesis_id,
                )
            ).encode()
        ).hexdigest()
        session.add(
            StrategyHypothesisRecord(
                id=uuid4(),
                hypothesis_id=hypothesis_id,
                run_id=run_id,
                predictor_name=value.predictor_name,
                predictor_version=value.predictor_version,
                hypothesis_contract_version=value.hypothesis_contract_version,
                direction_mapping=value.direction_mapping.value,
                bin_ordinal=value.bin_ordinal,
                lower_bound=value.lower_bound,
                upper_bound=value.upper_bound,
                suggested_action=value.suggested_action.value,
                horizon_seconds=int(value.horizon.total_seconds()),
                condition=value.condition.value,
                supporting_bin_ordinals=list(value.supporting_bin_ordinals),
                expected_gross_return=value.expected_gross_return,
                expected_net_return=value.expected_net_return,
                expected_avoided_loss=value.expected_avoided_loss,
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
                mean_forward_gross_return=benchmark.mean_forward_gross_return,
            )
        )
    await session.commit()
    return PredictiveEdgeExecution(run_id, False, result, 0, 0)


def predictive_edge_v4_experiment_key(
    *,
    dataset_fingerprint: str,
    instrument: Instrument,
    range_start: datetime,
    range_end: datetime,
    fee_rate: Decimal,
    slippage_rate: Decimal,
) -> str:
    """Build an identity that can never alias a legacy v3 run."""

    payload = {
        "engine": "predictive_edge_discovery_v4",
        "contract_version": PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "instrument": [
            instrument.asset.value,
            instrument.provider.value,
            instrument.market_type.value,
            instrument.interval.value,
        ],
        "range": [range_start.isoformat(), range_end.isoformat()],
        "fee_rate": str(fee_rate),
        "slippage_rate": str(slippage_rate),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def predictive_edge_v4_cache_namespace(
    *, instrument: Instrument, fee_rate: Decimal, slippage_rate: Decimal
) -> str:
    """Identify reusable v4 evidence without binding it to a complete dataset."""

    payload = {
        "engine": "predictive_edge_discovery_v4_stage_cache",
        "contract_version": PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
        "instrument": [
            instrument.asset.value,
            instrument.provider.value,
            instrument.market_type.value,
            instrument.interval.value,
        ],
        "predictor_version": PREDICTOR_VERSION,
        "fee_rate": str(fee_rate),
        "slippage_rate": str(slippage_rate),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def predictive_edge_v41_cache_namespace(
    *, instrument: Instrument, fee_rate: Decimal, slippage_rate: Decimal
) -> str:
    payload = {
        "engine": "predictive_edge_discovery_v41_stage_cache",
        "contract_version": PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
        "instrument": [
            instrument.asset.value,
            instrument.provider.value,
            instrument.market_type.value,
            instrument.interval.value,
        ],
        "fee_rate": str(fee_rate),
        "slippage_rate": str(slippage_rate),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


async def persist_predictive_edge_v4_result(
    *,
    session: AsyncSession,
    experiment_key: str,
    cache_namespace: str,
    instrument: Instrument,
    range_start: datetime,
    range_end: datetime,
    dataset_fingerprint: str,
    windows: tuple[WalkForwardWindow, ...],
    result: PredictiveEdgeV4Result,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    started_at: datetime,
    regime_result: PredictiveEdgeV41Result | None = None,
    regime_cache_namespace: str | None = None,
) -> PredictiveEdgeExecution:
    """Persist the real v4 result without reconstructing synthetic configuration."""

    run_id = uuid4()
    training_days = max(1, (windows[0].training_to - windows[0].training_from).days)
    validation_days = max(
        1, (windows[0].validation_to - windows[0].validation_from).days
    )
    test_days = max(1, (windows[0].test_to - windows[0].test_from).days)
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
        algorithm_version=PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
        score_model_name=MODEL_NAME,
        score_model_version=MODEL_VERSION,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        neutral_threshold=Decimal(0),
        decision_interval_seconds=0,
        training_days=training_days,
        validation_days=validation_days,
        test_days=test_days,
        step_days=max(1, (windows[1].training_from - windows[0].training_from).days)
        if len(windows) > 1
        else validation_days,
        maximum_windows=len(windows),
        bin_count=min(DISCOVERY_BIN_COUNTS),
        minimum_observations=0,
        minimum_windows=0,
        minimum_stable_direction_ratio=Decimal(0),
        minimum_absolute_correlation=Decimal(0),
        minimum_directional_hit_rate=Decimal(0),
        minimum_hit_rate_improvement_over_benchmark=Decimal(0),
        windows_available=len(windows),
        windows_required=0,
        conclusion_reason=result.outcome_explanation,
        windows_generated=len(windows),
        aggregate_metrics_computed=(
            result.training_counters.computed
            + result.validation_counters.computed
            + result.test_counters.computed
        ),
        metrics_reused=(
            result.training_counters.reused
            + result.validation_counters.reused
            + result.test_counters.reused
        ),
        compute_avoided_estimate=(
            result.training_counters.reused
            + result.validation_counters.reused
            + result.test_counters.reused
        ),
        reuse_count=0,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        failed_at=None,
        failure_reason=None,
    )
    session.add(run)
    await session.flush()
    session.add(
        PredictiveEdgeRunDiagnosticRecord(
            id=uuid4(),
            run_id=run_id,
            contract_version=PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
            effective_parameters=[
                {
                    "name": item.name,
                    "value": item.value,
                    "source": item.source.value,
                    "rationale": item.rationale,
                    "contract_version": item.contract_version,
                }
                for item in result.effective_parameters
            ],
            rejection_counts_by_reason=result.rejection_counts,
            rejection_counts_by_predictor=result.rejection_counts_by_predictor,
            rejection_counts_by_horizon=result.rejection_counts_by_horizon,
            rejection_counts_by_condition=result.rejection_counts_by_condition,
            rejection_counts_by_neutral_policy=(
                result.rejection_counts_by_neutral_policy
            ),
            rejection_counts_by_cadence=result.rejection_counts_by_cadence,
            strongest_observed_signal=result.strongest_observed_signal,
            bottleneck_stage=result.bottleneck_stage,
            diagnostic_conclusion=result.conclusion.value,
            outcome_explanation=result.outcome_explanation,
            stage_timings={name: str(value) for name, value in result.timings.items()},
            computed_counts={
                "cheap_relationships": result.training_counters.computed,
                "validation_configurations": result.validation_counters.computed,
                "test_configurations": result.test_counters.computed,
            },
            reused_counts={
                "cheap_relationships": result.training_counters.reused,
                "validation_configurations": result.validation_counters.reused,
                "test_configurations": result.test_counters.reused,
            },
            regime_conditioned_conclusion=(
                regime_result.conclusion.value if regime_result is not None else None
            ),
            regime_conditioned_stage_counters=(
                {
                    "a1": asdict(regime_result.stage_a1),
                    "a2": asdict(regime_result.stage_a2),
                    "a3": asdict(regime_result.stage_a3),
                    "validation": asdict(regime_result.stage_b),
                    "test": asdict(regime_result.stage_c),
                }
                if regime_result is not None
                else None
            ),
            regime_conditioned_summary=(
                {
                    "contract_version": PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
                    "confirmed_interaction_count": len(
                        regime_result.confirmed_candidates
                    ),
                    "actionable_near_miss_count": len(
                        regime_result.actionable_near_misses
                    ),
                    "diagnostic_near_miss_count": len(
                        regime_result.diagnostic_near_misses
                    ),
                    "hypotheses_generated": len(regime_result.hypotheses),
                    "primitive_counters": asdict(regime_result.primitive_counters),
                }
                if regime_result is not None
                else None
            ),
        )
    )
    validation_by_fingerprint = {
        item.configuration.fingerprint: item
        for item in result.validation_configurations
    }
    for near_miss in result.near_misses:
        validation = validation_by_fingerprint[near_miss.diagnostic.fingerprint]
        configuration = validation.configuration
        session.add(
            PredictiveEdgeNearMissRecord(
                id=uuid4(),
                run_id=run_id,
                rank=near_miss.rank,
                category_rank=near_miss.rank,
                engine="v4",
                engine_contract_version=PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
                edge_fingerprint=configuration.fingerprint,
                predictor_name=configuration.predictor.name,
                cadence_seconds=int(configuration.cadence.total_seconds()),
                horizon_seconds=int(configuration.horizon.total_seconds()),
                bin_count=configuration.bins,
                condition=configuration.condition.value,
                neutral_policy=configuration.neutral_policy.value,
                profile=configuration.profile.value,
                classification=near_miss.diagnostic.classification,
                primary_rejection_reason=(
                    near_miss.diagnostic.primary_rejection_reason.value
                    if near_miss.diagnostic.primary_rejection_reason is not None
                    else RejectionReason.VALIDATION_FAILED.value
                ),
                passed_requirements=[
                    item.value for item in near_miss.diagnostic.passed_requirements
                ],
                failed_requirements=[
                    item.value for item in near_miss.diagnostic.failed_requirements
                ],
                requirement_margins=[
                    {
                        "requirement": margin.requirement.value,
                        "observed_value": str(margin.observed_value),
                        "required_value": str(margin.required_value),
                        "absolute_margin": str(margin.absolute_margin),
                        "relative_margin": (
                            str(margin.relative_margin)
                            if margin.relative_margin is not None
                            else None
                        ),
                        "passed": margin.passed,
                    }
                    for margin in near_miss.diagnostic.margins
                ],
                usable_observations=near_miss.diagnostic.usable_observations,
                stable_direction_ratio=(near_miss.diagnostic.stable_direction_ratio),
                expected_net_movement=(near_miss.diagnostic.net_expected_movement),
                normalized_missing_distance=(near_miss.normalized_missing_distance),
            )
        )
    if regime_result is not None:
        next_rank = len(result.near_misses) + 1
        for category_rank, candidate in enumerate(
            sorted(
                regime_result.actionable_near_misses,
                key=lambda item: (
                    _v41_missing_distance(item.validation),
                    item.relationship.fingerprint,
                ),
            ),
            start=1,
        ):
            relationship = candidate.relationship
            failed = _v41_failed_requirements(candidate.validation)
            session.add(
                PredictiveEdgeNearMissRecord(
                    id=uuid4(),
                    run_id=run_id,
                    rank=next_rank,
                    category_rank=category_rank,
                    engine="v4.1",
                    engine_contract_version=PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
                    kind="actionable",
                    edge_fingerprint=relationship.fingerprint,
                    predictor_name=relationship.directional_predictor,
                    directional_predictor=relationship.directional_predictor,
                    directional_predictor_version=relationship.directional_version,
                    context_predictor=relationship.context_predictor,
                    context_predictor_version=relationship.context_version,
                    context_regime=relationship.regime.value,
                    selected_bin=relationship.selected_bin,
                    directional_lower_bound=relationship.directional_lower_bound,
                    directional_upper_bound=relationship.directional_upper_bound,
                    context_lower_bound=relationship.frozen_regime.lower_bound,
                    context_upper_bound=relationship.frozen_regime.upper_bound,
                    direction_mapping=relationship.direction_mapping.value,
                    suggested_action=relationship.action.value,
                    cadence_seconds=int(relationship.cadence.total_seconds()),
                    horizon_seconds=int(relationship.horizon.total_seconds()),
                    bin_count=3,
                    condition="all",
                    neutral_policy="training_noise",
                    profile=relationship.profile,
                    classification="regime_conditioned_near_miss",
                    primary_rejection_reason=(
                        candidate.rejection_reason or "validation_failed"
                    ),
                    passed_requirements=[],
                    failed_requirements=failed,
                    requirement_margins=[],
                    usable_observations=candidate.validation.observations,
                    stable_direction_ratio=(
                        candidate.validation.directional_hit_rate or Decimal(0)
                    ),
                    expected_net_movement=candidate.validation.net_movement,
                    expected_gross_movement=candidate.validation.gross_movement,
                    expected_aligned_movement=(
                        candidate.validation.aligned_gross_movement
                    ),
                    directional_hit_rate=candidate.validation.directional_hit_rate,
                    benchmark=candidate.validation.benchmark,
                    benchmark_improvement=candidate.validation.benchmark_improvement,
                    observations=candidate.validation.observations,
                    normalized_missing_distance=_v41_missing_distance(
                        candidate.validation
                    ),
                    training_evidence=_v41_evidence(relationship.training),
                    directional_baseline_evidence=_v41_evidence(
                        relationship.directional_baseline
                    ),
                    validation_evidence=_v41_evidence(candidate.validation),
                    test_evidence=_v41_evidence(candidate.test),
                    context_improvement=relationship.context_improvement,
                    usable_window_count=candidate.validation.observations,
                    confirmed_window_count=(
                        candidate.test.observations
                        if candidate.confirmed and candidate.test
                        else 0
                    ),
                )
            )
            next_rank += 1
        for category_rank, context_predictor in enumerate(
            regime_result.diagnostic_near_misses, start=1
        ):
            context_version = next(
                (
                    item.context_version
                    for item in regime_result.interactions
                    if item.context_predictor == context_predictor
                ),
                1,
            )
            session.add(
                PredictiveEdgeNearMissRecord(
                    id=uuid4(),
                    run_id=run_id,
                    rank=next_rank,
                    category_rank=category_rank,
                    engine="v4.1",
                    engine_contract_version=PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
                    kind="diagnostic",
                    edge_fingerprint=hashlib.sha256(
                        f"v4.1:diagnostic:{context_predictor}".encode()
                    ).hexdigest(),
                    predictor_name=context_predictor,
                    context_predictor=context_predictor,
                    context_predictor_version=context_version,
                    cadence_seconds=0,
                    horizon_seconds=0,
                    bin_count=0,
                    condition="all",
                    neutral_policy="training_noise",
                    profile="regime_conditioned_v41",
                    classification="diagnostic_context_predictor",
                    primary_rejection_reason="non_directional_predictor",
                    passed_requirements=[],
                    failed_requirements=["non_directional_predictor"],
                    requirement_margins=[],
                    usable_observations=0,
                    stable_direction_ratio=Decimal(0),
                    expected_net_movement=None,
                    normalized_missing_distance=Decimal(0),
                )
            )
            next_rank += 1
    for sensitivity in result.sensitivity:
        session.add(
            PredictiveEdgeSensitivityRecord(
                id=uuid4(),
                run_id=run_id,
                edge_fingerprint=sensitivity.edge_fingerprint,
                threshold_name=sensitivity.threshold_name.value,
                current_value=sensitivity.current_value,
                nearest_passing_value=sensitivity.nearest_passing_value,
                candidate_would_advance=sensitivity.candidate_would_advance,
                other_requirements_still_failed=[
                    item.value for item in sensitivity.other_requirements_still_failed
                ],
            )
        )
    hypotheses = {
        item.hypothesis_id: item
        for item in (
            *result.hypotheses,
            *(regime_result.hypotheses if regime_result is not None else ()),
        )
    }
    for hypothesis in (hypotheses[key] for key in sorted(hypotheses)):
        session.add(
            StrategyHypothesisRecord(
                id=uuid4(),
                hypothesis_id=hypothesis.hypothesis_id,
                run_id=run_id,
                predictor_name=hypothesis.predictor_name,
                predictor_version=hypothesis.predictor_version,
                hypothesis_contract_version=hypothesis.hypothesis_contract_version,
                direction_mapping=hypothesis.direction_mapping.value,
                bin_ordinal=hypothesis.bin_ordinal,
                lower_bound=hypothesis.lower_bound,
                upper_bound=hypothesis.upper_bound,
                suggested_action=hypothesis.suggested_action.value,
                horizon_seconds=int(hypothesis.horizon.total_seconds()),
                condition=hypothesis.condition.value,
                supporting_bin_ordinals=list(hypothesis.supporting_bin_ordinals),
                expected_gross_return=hypothesis.expected_gross_return,
                expected_net_return=hypothesis.expected_net_return,
                expected_avoided_loss=hypothesis.expected_avoided_loss,
                directional_hit_rate=hypothesis.directional_hit_rate,
                stability_ratio=hypothesis.stability_ratio,
                evaluation_interval_seconds=int(
                    hypothesis.minimum_evaluation_interval.total_seconds()
                ),
                holding_period_from_seconds=int(
                    hypothesis.minimum_holding_period_from.total_seconds()
                ),
                holding_period_to_seconds=int(
                    hypothesis.minimum_holding_period_to.total_seconds()
                ),
                evidence_status=hypothesis.evidence_status.value,
                rationale=hypothesis.rationale,
                limitations=hypothesis.limitations,
                context_predictor_name=hypothesis.context_predictor_name,
                context_predictor_version=hypothesis.context_predictor_version,
                context_regime=hypothesis.context_regime,
                context_lower_bound=hypothesis.context_lower_bound,
                context_upper_bound=hypothesis.context_upper_bound,
            )
        )
    for cache_key, cache_value in result.cache_writes.items():
        payload = encode_v4_cache_value(cache_value)
        if not isinstance(payload, dict):
            raise TypeError("v4 stage cache payload must be an object")
        if isinstance(cache_value, TrainingWindowEvidence):
            kind = "training"
            cadence = timedelta(0)
            horizon = timedelta(0)
            bins = None
            condition = None
            neutral_policy = None
            profile = None
        elif isinstance(cache_value, PhaseWindowEvidence):
            kind = cache_value.phase
            cadence = timedelta(0)
            horizon = timedelta(0)
            bins = None
            condition = None
            neutral_policy = None
            profile = None
        else:
            raise TypeError(f"unsupported v4 cache stage: {type(cache_value).__name__}")
        statement = (
            postgresql_insert(PredictiveEdgeV4CacheRecord)
            .values(
                id=uuid4(),
                cache_key=cache_key,
                cache_namespace=cache_namespace,
                status="completed",
                evidence_kind=kind,
                contract_version=PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
                dataset_fingerprint=dataset_fingerprint,
                cadence_seconds=int(cadence.total_seconds()),
                horizon_seconds=int(horizon.total_seconds()),
                bin_count=bins,
                condition=condition,
                neutral_policy=neutral_policy,
                profile=profile,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                payload=payload,
                created_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(index_elements=["cache_key"])
        )
        await session.execute(statement)
    if regime_result is not None and regime_cache_namespace is not None:
        for cache_key, evidence in regime_result.cache_writes.items():
            payload = {
                "observations": evidence.observations,
                "gross_movement": _decimal_text(evidence.gross_movement),
                "aligned_gross_movement": _decimal_text(
                    evidence.aligned_gross_movement
                ),
                "net_movement": _decimal_text(evidence.net_movement),
                "directional_hit_rate": _decimal_text(evidence.directional_hit_rate),
                "benchmark": _decimal_text(evidence.benchmark),
                "benchmark_improvement": _decimal_text(evidence.benchmark_improvement),
                "standard_error": _decimal_text(evidence.standard_error),
                "stable": evidence.stable,
                "economic": evidence.economic,
                "passes": evidence.passes,
            }
            await session.execute(
                postgresql_insert(PredictiveEdgeV4CacheRecord)
                .values(
                    id=uuid4(),
                    cache_key=cache_key,
                    cache_namespace=regime_cache_namespace,
                    status="completed",
                    evidence_kind="regime_conditioned_phase",
                    contract_version=PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
                    dataset_fingerprint=dataset_fingerprint,
                    cadence_seconds=0,
                    horizon_seconds=0,
                    bin_count=None,
                    condition=None,
                    neutral_policy=None,
                    profile="regime_conditioned_v41",
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    payload=payload,
                    created_at=datetime.now(UTC),
                )
                .on_conflict_do_nothing(index_elements=["cache_key"])
            )
    await session.commit()
    return PredictiveEdgeExecution(run_id, False, None, 0, 0)


async def load_predictive_edge_v4_cache(
    session: AsyncSession, *, cache_namespace: str
) -> dict[str, object]:
    """Load only completed evidence for one exact v4 experiment identity."""

    rows = (
        await session.scalars(
            select(PredictiveEdgeV4CacheRecord).where(
                PredictiveEdgeV4CacheRecord.cache_namespace == cache_namespace,
                PredictiveEdgeV4CacheRecord.status == "completed",
                PredictiveEdgeV4CacheRecord.contract_version
                == PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
            )
        )
    ).all()
    result: dict[str, object] = {}
    for row in rows:
        try:
            result[row.cache_key] = decode_v4_cache_value(row.payload)
        except (TypeError, ValueError):
            # A malformed or incomplete cache row is never reusable.
            continue
    return result


async def persist_predictive_edge_v41_addendum(
    *,
    session: AsyncSession,
    run_id: UUID,
    result: PredictiveEdgeV41Result,
) -> None:
    """Attach immutable v4.1 observability to a reusable pre-v4.1 v4 run."""

    counters = {
        "a1": asdict(result.stage_a1),
        "a2": asdict(result.stage_a2),
        "a3": asdict(result.stage_a3),
        "validation": asdict(result.stage_b),
        "test": asdict(result.stage_c),
    }
    summary: dict[str, object] = {
        "contract_version": PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
        "confirmed_interaction_count": len(result.confirmed_candidates),
        "actionable_near_miss_count": len(result.actionable_near_misses),
        "diagnostic_near_miss_count": len(result.diagnostic_near_misses),
        "hypotheses_generated": len(result.hypotheses),
        "primitive_counters": asdict(result.primitive_counters),
    }
    await session.execute(
        update(PredictiveEdgeRunDiagnosticRecord)
        .where(PredictiveEdgeRunDiagnosticRecord.run_id == run_id)
        .values(
            regime_conditioned_conclusion=result.conclusion.value,
            regime_conditioned_stage_counters=counters,
            regime_conditioned_summary=summary,
        )
    )
    await session.execute(
        delete(PredictiveEdgeNearMissRecord).where(
            PredictiveEdgeNearMissRecord.run_id == run_id,
            PredictiveEdgeNearMissRecord.engine == "v4.1",
        )
    )
    maximum_rank = await session.scalar(
        select(PredictiveEdgeNearMissRecord.rank)
        .where(PredictiveEdgeNearMissRecord.run_id == run_id)
        .order_by(PredictiveEdgeNearMissRecord.rank.desc())
        .limit(1)
    )
    next_rank = (maximum_rank or 0) + 1
    for category_rank, candidate in enumerate(
        sorted(
            result.actionable_near_misses,
            key=lambda item: (
                _v41_missing_distance(item.validation),
                item.relationship.fingerprint,
            ),
        ),
        start=1,
    ):
        relationship = candidate.relationship
        session.add(
            PredictiveEdgeNearMissRecord(
                id=uuid4(),
                run_id=run_id,
                rank=next_rank,
                category_rank=category_rank,
                engine="v4.1",
                engine_contract_version=PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
                kind="actionable",
                edge_fingerprint=relationship.fingerprint,
                predictor_name=relationship.directional_predictor,
                directional_predictor=relationship.directional_predictor,
                directional_predictor_version=relationship.directional_version,
                context_predictor=relationship.context_predictor,
                context_predictor_version=relationship.context_version,
                context_regime=relationship.regime.value,
                selected_bin=relationship.selected_bin,
                directional_lower_bound=relationship.directional_lower_bound,
                directional_upper_bound=relationship.directional_upper_bound,
                context_lower_bound=relationship.frozen_regime.lower_bound,
                context_upper_bound=relationship.frozen_regime.upper_bound,
                direction_mapping=relationship.direction_mapping.value,
                suggested_action=relationship.action.value,
                cadence_seconds=int(relationship.cadence.total_seconds()),
                horizon_seconds=int(relationship.horizon.total_seconds()),
                bin_count=3,
                condition="all",
                neutral_policy="training_noise",
                profile=relationship.profile,
                classification="regime_conditioned_near_miss",
                primary_rejection_reason=candidate.rejection_reason
                or "validation_failed",
                passed_requirements=[],
                failed_requirements=_v41_failed_requirements(candidate.validation),
                requirement_margins=[],
                usable_observations=candidate.validation.observations,
                stable_direction_ratio=candidate.validation.directional_hit_rate
                or Decimal(0),
                expected_net_movement=candidate.validation.net_movement,
                expected_gross_movement=candidate.validation.gross_movement,
                expected_aligned_movement=candidate.validation.aligned_gross_movement,
                directional_hit_rate=candidate.validation.directional_hit_rate,
                benchmark=candidate.validation.benchmark,
                benchmark_improvement=candidate.validation.benchmark_improvement,
                observations=candidate.validation.observations,
                normalized_missing_distance=_v41_missing_distance(candidate.validation),
                training_evidence=_v41_evidence(relationship.training),
                directional_baseline_evidence=_v41_evidence(
                    relationship.directional_baseline
                ),
                validation_evidence=_v41_evidence(candidate.validation),
                test_evidence=_v41_evidence(candidate.test),
                context_improvement=relationship.context_improvement,
                usable_window_count=candidate.validation.observations,
                confirmed_window_count=(
                    candidate.test.observations
                    if candidate.confirmed and candidate.test
                    else 0
                ),
            )
        )
        next_rank += 1
    for category_rank, context_predictor in enumerate(
        result.diagnostic_near_misses, start=1
    ):
        context_version = next(
            (
                item.context_version
                for item in result.interactions
                if item.context_predictor == context_predictor
            ),
            1,
        )
        session.add(
            PredictiveEdgeNearMissRecord(
                id=uuid4(),
                run_id=run_id,
                rank=next_rank,
                category_rank=category_rank,
                engine="v4.1",
                engine_contract_version=PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
                kind="diagnostic",
                edge_fingerprint=hashlib.sha256(
                    f"v4.1:diagnostic:{context_predictor}".encode()
                ).hexdigest(),
                predictor_name=context_predictor,
                context_predictor=context_predictor,
                context_predictor_version=context_version,
                cadence_seconds=0,
                horizon_seconds=0,
                bin_count=0,
                condition="all",
                neutral_policy="training_noise",
                profile="regime_conditioned_v41",
                classification="diagnostic_context_predictor",
                primary_rejection_reason="non_directional_predictor",
                passed_requirements=[],
                failed_requirements=["non_directional_predictor"],
                requirement_margins=[],
                usable_observations=0,
                stable_direction_ratio=Decimal(0),
                expected_net_movement=None,
                normalized_missing_distance=Decimal(0),
            )
        )
        next_rank += 1
    await session.commit()


async def load_predictive_edge_v41_cache(
    session: AsyncSession, *, cache_namespace: str
) -> dict[str, V41PhaseEvidence]:
    rows = (
        await session.scalars(
            select(PredictiveEdgeV4CacheRecord).where(
                PredictiveEdgeV4CacheRecord.cache_namespace == cache_namespace,
                PredictiveEdgeV4CacheRecord.status == "completed",
                PredictiveEdgeV4CacheRecord.contract_version
                == PREDICTIVE_EDGE_V41_CONTRACT_VERSION,
            )
        )
    ).all()
    result: dict[str, V41PhaseEvidence] = {}
    for row in rows:
        try:
            payload = row.payload
            result[row.cache_key] = V41PhaseEvidence(
                observations=int(str(payload["observations"])),
                gross_movement=_decimal(payload.get("gross_movement")),
                aligned_gross_movement=_decimal(payload.get("aligned_gross_movement")),
                net_movement=_decimal(payload.get("net_movement")),
                directional_hit_rate=_decimal(payload.get("directional_hit_rate")),
                benchmark=_decimal(payload.get("benchmark")),
                benchmark_improvement=_decimal(payload.get("benchmark_improvement")),
                standard_error=_decimal(payload.get("standard_error")),
                stable=bool(payload["stable"]),
                economic=bool(payload["economic"]),
                passes=bool(payload["passes"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _decimal_text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _decimal(value: object) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None
