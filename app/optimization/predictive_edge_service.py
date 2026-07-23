import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import select, update
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
        ÛŞ¶¶‰Ëkºwµç}¹”°(€€€€¤(€€€Í•ÍÍ¥½¸¹…‘¡ÉÕ¸¤(€€€…İ…¥ĞÍ•ÍÍ¥½¸¹™±ÕÍ  ¤(€€€Í•ÍÍ¥½¸¹…‘ (€€€€€€€AÉ•‘¥Ñ¥Ù•‘•IÕ¹¥…¹½ÍÑ¥I•½É (€€€€€€€€€€€¥õÕÕ¥Ğ ¤°(€€€€€€€€€€€ÉÕ¹}¥õÉÕ¹}¥°(€€€€€€€€€€€½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸õAI%Q%Y}}XÑ}=9QIQ}YIM%=8°(€€€€€€€€€€€•™™•Ñ¥Ù•}Á…É…µ•Ñ•ÉÌõl(€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰¹…µ”ˆè¥Ñ•´¹¹…µ”°(€€€€€€€€€€€€€€€€€€€€‰Ù…±Õ”ˆè¥Ñ•´¹Ù…±Õ”°(€€€€€€€€€€€€€€€€€€€€‰Í½ÕÉ”ˆè¥Ñ•´¹Í½ÕÉ”¹Ù…±Õ”°(€€€€€€€€€€€€€€€€€€€€‰É…Ñ¥½¹…±”ˆè¥Ñ•´¹É…Ñ¥½¹…±”°(€€€€€€€€€€€€€€€€€€€€‰½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸ˆè¥Ñ•´¹½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€€€€™½È¥Ñ•´¥¸É•ÍÕ±Ğ¹•™™•Ñ¥Ù•}Á…É…µ•Ñ•ÉÌ(€€€€€€€€€€€t°(€€€€€€€€€€€É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}É•…Í½¸õÉ•ÍÕ±Ğ¹É•©•Ñ¥½¹}½Õ¹ÑÌ°(€€€€€€€€€€€É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}ÁÉ•‘¥Ñ½ÈõÉ•ÍÕ±Ğ¹É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}ÁÉ•‘¥Ñ½È°(€€€€€€€€€€€É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}¡½É¥é½¸õÉ•ÍÕ±Ğ¹É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}¡½É¥é½¸°(€€€€€€€€€€€É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}½¹‘¥Ñ¥½¸õÉ•ÍÕ±Ğ¹É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}½¹‘¥Ñ¥½¸°(€€€€€€€€€€€É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}¹•ÕÑÉ…±}Á½±¥äô (€€€€€€€€€€€€€€€É•ÍÕ±Ğ¹É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}¹•ÕÑÉ…±}Á½±¥ä(€€€€€€€€€€€€¤°(€€€€€€€€€€€É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}…‘•¹”õÉ•ÍÕ±Ğ¹É•©•Ñ¥½¹}½Õ¹ÑÍ}‰å}…‘•¹”°(€€€€€€€€€€€ÍÑÉ½¹•ÍÑ}½‰Í•ÉÙ•‘}Í¥¹…°õÉ•ÍÕ±Ğ¹ÍÑÉ½¹•ÍÑ}½‰Í•ÉÙ•‘}Í¥¹…°°(€€€€€€€€€€€‰½ÑÑ±•¹•­}ÍÑ…”õÉ•ÍÕ±Ğ¹‰½ÑÑ±•¹•­}ÍÑ…”°(€€€€€€€€€€€‘¥…¹½ÍÑ¥}½¹±ÕÍ¥½¸õÉ•ÍÕ±Ğ¹½¹±ÕÍ¥½¸¹Ù…±Õ”°(€€€€€€€€€€€½ÕÑ½µ•}•áÁ±…¹…Ñ¥½¸õÉ•ÍÕ±Ğ¹½ÕÑ½µ•}•áÁ±…¹…Ñ¥½¸°(€€€€€€€€€€€ÍÑ…•}Ñ¥µ¥¹Ìõí¹…µ”èÍÑÈ¡Ù…±Õ”¤™½È¹…µ”°Ù…±Õ”¥¸É•ÍÕ±Ğ¹Ñ¥µ¥¹Ì¹¥Ñ•µÌ ¥ô°(€€€€€€€€€€€½µÁÕÑ•‘}½Õ¹ÑÌõì(€€€€€€€€€€€€€€€€‰¡•…Á}É•±…Ñ¥½¹Í¡¥ÁÌˆèÉ•ÍÕ±Ğ¹ÑÉ…¥¹¥¹}½Õ¹Ñ•ÉÌ¹½µÁÕÑ•°(€€€€€€€€€€€€€€€€‰Ù…±¥‘…Ñ¥½¹}½¹™¥ÕÉ…Ñ¥½¹ÌˆèÉ•ÍÕ±Ğ¹Ù…±¥‘…Ñ¥½¹}½Õ¹Ñ•ÉÌ¹½µÁÕÑ•°(€€€€€€€€€€€€€€€€‰Ñ•ÍÑ}½¹™¥ÕÉ…Ñ¥½¹ÌˆèÉ•ÍÕ±Ğ¹Ñ•ÍÑ}½Õ¹Ñ•ÉÌ¹½µÁÕÑ•°(€€€€€€€€€€€ô°(€€€€€€€€€€€É•ÕÍ•‘}½Õ¹ÑÌõì(€€€€€€€€€€€€€€€€‰¡•…Á}É•±…Ñ¥½¹Í¡¥ÁÌˆèÉ•ÍÕ±Ğ¹ÑÉ…¥¹¥¹}½Õ¹Ñ•ÉÌ¹É•ÕÍ•°(€€€€€€€€€€€€€€€€‰Ù…±¥‘…Ñ¥½¹}½¹™¥ÕÉ…Ñ¥½¹ÌˆèÉ•ÍÕ±Ğ¹Ù…±¥‘…Ñ¥½¹}½Õ¹Ñ•ÉÌ¹É•ÕÍ•°(€€€€€€€€€€€€€€€€‰Ñ•ÍÑ}½¹™¥ÕÉ…Ñ¥½¹ÌˆèÉ•ÍÕ±Ğ¹Ñ•ÍÑ}½Õ¹Ñ•ÉÌ¹É•ÕÍ•°(€€€€€€€€€€€ô°(€€€€€€€€€€€É•¥µ•}½¹‘¥Ñ¥½¹•‘}½¹±ÕÍ¥½¸ô (€€€€€€€€€€€€€€€É•¥µ•}É•ÍÕ±Ğ¹½¹±ÕÍ¥½¸¹Ù…±Õ”¥˜É•¥µ•}É•ÍÕ±Ğ¥Ì¹½Ğ9½¹”•±Í”9½¹”(€€€€€€€€€€€€¤°(€€€€€€€€€€€É•¥µ•}½¹‘¥Ñ¥½¹•‘}ÍÑ…•}½Õ¹Ñ•ÉÌô (€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰„Äˆè…Í‘¥Ğ¡É•¥µ•}É•ÍÕ±Ğ¹ÍÑ…•}„Ä¤°(€€€€€€€€€€€€€€€€€€€€‰„Èˆè…Í‘¥Ğ¡É•¥µ•}É•ÍÕ±Ğ¹ÍÑ…•}„È¤°(€€€€€€€€€€€€€€€€€€€€‰„Ìˆè…Í‘¥Ğ¡É•¥µ•}É•ÍÕ±Ğ¹ÍÑ…•}„Ì¤°(€€€€€€€€€€€€€€€€€€€€‰Ù…±¥‘…Ñ¥½¸ˆè…Í‘¥Ğ¡É•¥µ•}É•ÍÕ±Ğ¹ÍÑ…•}ˆ¤°(€€€€€€€€€€€€€€€€€€€€‰Ñ•ÍĞˆè…Í‘¥Ğ¡É•¥µ•}É•ÍÕ±Ğ¹ÍÑ…•}Œ¤°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€€€€¥˜É•¥µ•}É•ÍÕ±Ğ¥Ì¹½Ğ9½¹”(€€€€€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€€€€€¤°(€€€€€€€€¤(€€€€¤(€€€Ù…±¥‘…Ñ¥½¹}‰å}™¥¹•ÉÁÉ¥¹Ğ€ôì(€€€€€€€¥Ñ•´¹½¹™¥ÕÉ…Ñ¥½¸¹™¥¹•ÉÁÉ¥¹Ğè¥Ñ•´(€€€€€€€™½È¥Ñ•´¥¸É•ÍÕ±Ğ¹Ù…±¥‘…Ñ¥½¹}½¹™¥ÕÉ…Ñ¥½¹Ì(€€€ô(€€€™½È¹•…É}µ¥ÍÌ¥¸É•ÍÕ±Ğ¹¹•…É}µ¥ÍÍ•Ìè(€€€€€€€Ù…±¥‘…Ñ¥½¸€ôÙ…±¥‘…Ñ¥½¹}‰å}™¥¹•ÉÁÉ¥¹Ñm¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹™¥¹•ÉÁÉ¥¹Ñt(€€€€€€€½¹™¥ÕÉ…Ñ¥½¸€ôÙ…±¥‘…Ñ¥½¸¹½¹™¥ÕÉ…Ñ¥½¸(€€€€€€€Í•ÍÍ¥½¸¹…‘ (€€€€€€€€€€€AÉ•‘¥Ñ¥Ù•‘•9•…É5¥ÍÍI•½É (€€€€€€€€€€€€€€€¥õÕÕ¥Ğ ¤°(€€€€€€€€€€€€€€€ÉÕ¹}¥õÉÕ¹}¥°(€€€€€€€€€€€€€€€É…¹¬õ¹•…É}µ¥ÍÌ¹É…¹¬°(€€€€€€€€€€€€€€€•‘•}™¥¹•ÉÁÉ¥¹Ğõ½¹™¥ÕÉ…Ñ¥½¸¹™¥¹•ÉÁÉ¥¹Ğ°(€€€€€€€€€€€€€€€ÁÉ•‘¥Ñ½É}¹…µ”õ½¹™¥ÕÉ…Ñ¥½¸¹ÁÉ•‘¥Ñ½È¹¹…µ”°(€€€€€€€€€€€€€€€…‘•¹•}Í•½¹‘Ìõ¥¹Ğ¡½¹™¥ÕÉ…Ñ¥½¸¹…‘•¹”¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤°(€€€€€€€€€€€€€€€¡½É¥é½¹}Í•½¹‘Ìõ¥¹Ğ¡½¹™¥ÕÉ…Ñ¥½¸¹¡½É¥é½¸¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤°(€€€€€€€€€€€€€€€‰¥¹}½Õ¹Ğõ½¹™¥ÕÉ…Ñ¥½¸¹‰¥¹Ì°(€€€€€€€€€€€€€€€½¹‘¥Ñ¥½¸õ½¹™¥ÕÉ…Ñ¥½¸¹½¹‘¥Ñ¥½¸¹Ù…±Õ”°(€€€€€€€€€€€€€€€¹•ÕÑÉ…±}Á½±¥äõ½¹™¥ÕÉ…Ñ¥½¸¹¹•ÕÑÉ…±}Á½±¥ä¹Ù…±Õ”°(€€€€€€€€€€€€€€€ÁÉ½™¥±”õ½¹™¥ÕÉ…Ñ¥½¸¹ÁÉ½™¥±”¹Ù…±Õ”°(€€€€€€€€€€€€€€€±…ÍÍ¥™¥…Ñ¥½¸õ¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹±…ÍÍ¥™¥…Ñ¥½¸°(€€€€€€€€€€€€€€€ÁÉ¥µ…Éå}É•©•Ñ¥½¹}É•…Í½¸ô (€€€€€€€€€€€€€€€€€€€¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹ÁÉ¥µ…Éå}É•©•Ñ¥½¹}É•…Í½¸¹Ù…±Õ”(€€€€€€€€€€€€€€€€€€€¥˜¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹ÁÉ¥µ…Éå}É•©•Ñ¥½¹}É•…Í½¸¥Ì¹½Ğ9½¹”(€€€€€€€€€€€€€€€€€€€•±Í”I•©•Ñ¥½¹I•…Í½¸¹Y1%Q%=9}%1¹Ù…±Õ”(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€Á…ÍÍ•‘}É•ÅÕ¥É•µ•¹ÑÌõl(€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Ù…±Õ”™½È¥Ñ•´¥¸¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹Á…ÍÍ•‘}É•ÅÕ¥É•µ•¹ÑÌ(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€€€€€™…¥±•‘}É•ÅÕ¥É•µ•¹ÑÌõl(€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Ù…±Õ”™½È¥Ñ•´¥¸¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹™…¥±•‘}É•ÅÕ¥É•µ•¹ÑÌ(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€€€€€É•ÅÕ¥É•µ•¹Ñ}µ…É¥¹Ìõl(€€€€€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€€€€€‰É•ÅÕ¥É•µ•¹Ğˆèµ…É¥¸¹É•ÅÕ¥É•µ•¹Ğ¹Ù…±Õ”°(€€€€€€€€€€€€€€€€€€€€€€€€‰½‰Í•ÉÙ•‘}Ù…±Õ”ˆèÍÑÈ¡µ…É¥¸¹½‰Í•ÉÙ•‘}Ù…±Õ”¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰É•ÅÕ¥É•‘}Ù…±Õ”ˆèÍÑÈ¡µ…É¥¸¹É•ÅÕ¥É•‘}Ù…±Õ”¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰…‰Í½±ÕÑ•}µ…É¥¸ˆèÍÑÈ¡µ…É¥¸¹…‰Í½±ÕÑ•}µ…É¥¸¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰É•±…Ñ¥Ù•}µ…É¥¸ˆè€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€ÍÑÈ¡µ…É¥¸¹É•±…Ñ¥Ù•}µ…É¥¸¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€¥˜µ…É¥¸¹É•±…Ñ¥Ù•}µ…É¥¸¥Ì¹½Ğ9½¹”(€€€€€€€€€€€€€€€€€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€€€€€€‰Á…ÍÍ•ˆèµ…É¥¸¹Á…ÍÍ•°(€€€€€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€€€€€€€€™½Èµ…É¥¸¥¸¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹µ…É¥¹Ì(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€€€€€ÕÍ…‰±•}½‰Í•ÉÙ…Ñ¥½¹Ìõ¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹ÕÍ…‰±•}½‰Í•ÉÙ…Ñ¥½¹Ì°(€€€€€€€€€€€€€€€ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼ô¡¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹ÍÑ…‰±•}‘¥É•Ñ¥½¹}É…Ñ¥¼¤°(€€€€€€€€€€€€€€€•áÁ•Ñ•‘}¹•Ñ}µ½Ù•µ•¹Ğô¡¹•…É}µ¥ÍÌ¹‘¥…¹½ÍÑ¥Œ¹¹•Ñ}•áÁ•Ñ•‘}µ½Ù•µ•¹Ğ¤°(€€€€€€€€€€€€€€€¹½Éµ…±¥é•‘}µ¥ÍÍ¥¹}‘¥ÍÑ…¹”ô¡¹•…É}µ¥ÍÌ¹¹½Éµ…±¥é•‘}µ¥ÍÍ¥¹}‘¥ÍÑ…¹”¤°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€™½ÈÍ•¹Í¥Ñ¥Ù¥Ñä¥¸É•ÍÕ±Ğ¹Í•¹Í¥Ñ¥Ù¥Ñäè(€€€€€€€Í•ÍÍ¥½¸¹…‘ (€€€€€€€€€€€AÉ•‘¥Ñ¥Ù•‘•M•¹Í¥Ñ¥Ù¥ÑåI•½É (€€€€€€€€€€€€€€€¥õÕÕ¥Ğ ¤°(€€€€€€€€€€€€€€€ÉÕ¹}¥õÉÕ¹}¥°(€€€€€€€€€€€€€€€•‘•}™¥¹•ÉÁÉ¥¹ĞõÍ•¹Í¥Ñ¥Ù¥Ñä¹•‘•}™¥¹•ÉÁÉ¥¹Ğ°(€€€€€€€€€€€€€€€Ñ¡É•Í¡½±‘}¹…µ”õÍ•¹Í¥Ñ¥Ù¥Ñä¹Ñ¡É•Í¡½±‘}¹…µ”¹Ù…±Õ”°(€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}Ù…±Õ”õÍ•¹Í¥Ñ¥Ù¥Ñä¹ÕÉÉ•¹Ñ}Ù…±Õ”°(€€€€€€€€€€€€€€€¹•…É•ÍÑ}Á…ÍÍ¥¹}Ù…±Õ”õÍ•¹Í¥Ñ¥Ù¥Ñä¹¹•…É•ÍÑ}Á…ÍÍ¥¹}Ù…±Õ”°(€€€€€€€€€€€€€€€…¹‘¥‘…Ñ•}İ½Õ±‘}…‘Ù…¹”õÍ•¹Í¥Ñ¥Ù¥Ñä¹…¹‘¥‘…Ñ•}İ½Õ±‘}…‘Ù…¹”°(€€€€€€€€€€€€€€€½Ñ¡•É}É•ÅÕ¥É•µ•¹ÑÍ}ÍÑ¥±±}™…¥±•õl(€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Ù…±Õ”™½È¥Ñ•´¥¸Í•¹Í¥Ñ¥Ù¥Ñä¹½Ñ¡•É}É•ÅÕ¥É•µ•¹ÑÍ}ÍÑ¥±±}™…¥±•(€€€€€€€€€€€€€€€t°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€¡åÁ½Ñ¡•Í•Ì€ôì(€€€€€€€¥Ñ•´¹¡åÁ½Ñ¡•Í¥Í}¥è¥Ñ•´(€€€€€€€™½È¥Ñ•´¥¸€ (€€€€€€€€€€€€©É•ÍÕ±Ğ¹¡åÁ½Ñ¡•Í•Ì°(€€€€€€€€€€€€¨¡É•¥µ•}É•ÍÕ±Ğ¹¡åÁ½Ñ¡•Í•Ì¥˜É•¥µ•}É•ÍÕ±Ğ¥Ì¹½Ğ9½¹”•±Í”€ ¤¤°(€€€€€€€€¤(€€€ô(€€€™½È¡åÁ½Ñ¡•Í¥Ì¥¸€¡¡åÁ½Ñ¡•Í•Ím­•åt™½È­•ä¥¸Í½ÉÑ•¡¡åÁ½Ñ¡•Í•Ì¤¤è(€€€€€€€Í•ÍÍ¥½¸¹…‘ (€€€€€€€€€€€MÑÉ…Ñ•å!åÁ½Ñ¡•Í¥ÍI•½É (€€€€€€€€€€€€€€€¥õÕÕ¥Ğ ¤°(€€€€€€€€€€€€€€€¡åÁ½Ñ¡•Í¥Í}¥õ¡åÁ½Ñ¡•Í¥Ì¹¡åÁ½Ñ¡•Í¥Í}¥°(€€€€€€€€€€€€€€€ÉÕ¹}¥õÉÕ¹}¥°(€€€€€€€€€€€€€€€ÁÉ•‘¥Ñ½É}¹…µ”õ¡åÁ½Ñ¡•Í¥Ì¹ÁÉ•‘¥Ñ½É}¹…µ”°(€€€€€€€€€€€€€€€ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸õ¡åÁ½Ñ¡•Í¥Ì¹ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸°(€€€€€€€€€€€€€€€¡åÁ½Ñ¡•Í¥Í}½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸õ¡åÁ½Ñ¡•Í¥Ì¹¡åÁ½Ñ¡•Í¥Í}½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸°(€€€€€€€€€€€€€€€‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œõ¡åÁ½Ñ¡•Í¥Ì¹‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ¹Ù…±Õ”°(€€€€€€€€€€€€€€€‰¥¹}½É‘¥¹…°õ¡åÁ½Ñ¡•Í¥Ì¹‰¥¹}½É‘¥¹…°°(€€€€€€€€€€€€€€€±½İ•É}‰½Õ¹õ¡åÁ½Ñ¡•Í¥Ì¹±½İ•É}‰½Õ¹°(€€€€€€€€€€€€€€€ÕÁÁ•É}‰½Õ¹õ¡åÁ½Ñ¡•Í¥Ì¹ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€€€€€ÍÕ•ÍÑ•‘}…Ñ¥½¸õ¡åÁ½Ñ¡•Í¥Ì¹ÍÕ•ÍÑ•‘}…Ñ¥½¸¹Ù…±Õ”°(€€€€€€€€€€€€€€€¡½É¥é½¹}Í•½¹‘Ìõ¥¹Ğ¡¡åÁ½Ñ¡•Í¥Ì¹¡½É¥é½¸¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤°(€€€€€€€€€€€€€€€½¹‘¥Ñ¥½¸õ¡åÁ½Ñ¡•Í¥Ì¹½¹‘¥Ñ¥½¸¹Ù…±Õ”°(€€€€€€€€€€€€€€€ÍÕÁÁ½ÉÑ¥¹}‰¥¹}½É‘¥¹…±Ìõ±¥ÍĞ¡¡åÁ½Ñ¡•Í¥Ì¹ÍÕÁÁ½ÉÑ¥¹}‰¥¹}½É‘¥¹…±Ì¤°(€€€€€€€€€€€€€€€•áÁ•Ñ•‘}É½ÍÍ}É•ÑÕÉ¸õ¡åÁ½Ñ¡•Í¥Ì¹•áÁ•Ñ•‘}É½ÍÍ}É•ÑÕÉ¸°(€€€€€€€€€€€€€€€•áÁ•Ñ•‘}¹•Ñ}É•ÑÕÉ¸õ¡åÁ½Ñ¡•Í¥Ì¹•áÁ•Ñ•‘}¹•Ñ}É•ÑÕÉ¸°(€€€€€€€€€€€€€€€•áÁ•Ñ•‘}…Ù½¥‘•‘}±½ÍÌõ¡åÁ½Ñ¡•Í¥Ì¹•áÁ•Ñ•‘}…Ù½¥‘•‘}±½ÍÌ°(€€€€€€€€€€€€€€€‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”õ¡åÁ½Ñ¡•Í¥Ì¹‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”°(€€€€€€€€€€€€€€€ÍÑ…‰¥±¥Ñå}É…Ñ¥¼õ¡åÁ½Ñ¡•Í¥Ì¹ÍÑ…‰¥±¥Ñå}É…Ñ¥¼°(€€€€€€€€€€€€€€€•Ù…±Õ…Ñ¥½¹}¥¹Ñ•ÉÙ…±}Í•½¹‘Ìõ¥¹Ğ (€€€€€€€€€€€€€€€€€€€¡åÁ½Ñ¡•Í¥Ì¹µ¥¹¥µÕµ}•Ù…±Õ…Ñ¥½¹}¥¹Ñ•ÉÙ…°¹Ñ½Ñ…±}Í•½¹‘Ì ¤(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€¡½±‘¥¹}Á•É¥½‘}™É½µ}Í•½¹‘Ìõ¥¹Ğ (€€€€€€€€€€€€€€€€€€€¡åÁ½Ñ¡•Í¥Ì¹µ¥¹¥µÕµ}¡½±‘¥¹}Á•É¥½‘}™É½´¹Ñ½Ñ…±}Í•½¹‘Ì ¤(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€¡½±‘¥¹}Á•É¥½‘}Ñ½}Í•½¹‘Ìõ¥¹Ğ (€€€€€€€€€€€€€€€€€€€¡åÁ½Ñ¡•Í¥Ì¹µ¥¹¥µÕµ}¡½±‘¥¹}Á•É¥½‘}Ñ¼¹Ñ½Ñ…±}Í•½¹‘Ì ¤(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€•Ù¥‘•¹•}ÍÑ…ÑÕÌõ¡åÁ½Ñ¡•Í¥Ì¹•Ù¥‘•¹•}ÍÑ…ÑÕÌ¹Ù…±Õ”°(€€€€€€€€€€€€€€€É…Ñ¥½¹…±”õ¡åÁ½Ñ¡•Í¥Ì¹É…Ñ¥½¹…±”°(€€€€€€€€€€€€€€€±¥µ¥Ñ…Ñ¥½¹Ìõ¡åÁ½Ñ¡•Í¥Ì¹±¥µ¥Ñ…Ñ¥½¹Ì°(€€€€€€€€€€€€€€€½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}¹…µ”õ¡åÁ½Ñ¡•Í¥Ì¹½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}¹…µ”°(€€€€€€€€€€€€€€€½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸õ¡åÁ½Ñ¡•Í¥Ì¹½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸°(€€€€€€€€€€€€€€€½¹Ñ•áÑ}É•¥µ”õ¡åÁ½Ñ¡•Í¥Ì¹½¹Ñ•áÑ}É•¥µ”°(€€€€€€€€€€€€€€€½¹Ñ•áÑ}±½İ•É}‰½Õ¹õ¡åÁ½Ñ¡•Í¥Ì¹½¹Ñ•áÑ}±½İ•É}‰½Õ¹°(€€€€€€€€€€€€€€€½¹Ñ•áÑ}ÕÁÁ•É}‰½Õ¹õ¡åÁ½Ñ¡•Í¥Ì¹½¹Ñ•áÑ}ÕÁÁ•É}‰½Õ¹°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€™½È…¡•}­•ä°…¡•}Ù…±Õ”¥¸É•ÍÕ±Ğ¹…¡•}İÉ¥Ñ•Ì¹¥Ñ•µÌ ¤è(€€€€€€€Á…å±½…€ô•¹½‘•}ØÑ}…¡•}Ù…±Õ”¡…¡•}Ù…±Õ”¤(€€€€€€€¥˜¹½Ğ¥Í¥¹ÍÑ…¹”¡Á…å±½…°‘¥Ğ¤è(€€€€€€€€€€€É…¥Í”QåÁ•ÉÉ½È ‰ØĞÍÑ…”…¡”Á…å±½…µÕÍĞ‰”…¸½‰©•Ğˆ¤(€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡…¡•}Ù…±Õ”°QÉ…¥¹¥¹]¥¹‘½İÙ¥‘•¹”¤è(€€€€€€€€€€€­¥¹€ô€‰ÑÉ…¥¹¥¹œˆ(€€€€€€€€€€€…‘•¹”€ôÑ¥µ•‘•±Ñ„ À¤(€€€€€€€€€€€¡½É¥é½¸€ôÑ¥µ•‘•±Ñ„ À¤(€€€€€€€€€€€‰¥¹Ì€ô9½¹”(€€€€€€€€€€€½¹‘¥Ñ¥½¸€ô9½¹”(€€€€€€€€€€€¹•ÕÑÉ…±}Á½±¥ä€ô9½¹”(€€€€€€€€€€€ÁÉ½™¥±”€ô9½¹”(€€€€€€€•±¥˜¥Í¥¹ÍÑ…¹”¡…¡•}Ù…±Õ”°A¡…Í•]¥¹‘½İÙ¥‘•¹”¤è(€€€€€€€€€€€­¥¹€ô…¡•}Ù…±Õ”¹Á¡…Í”(€€€€€€€€€€€…‘•¹”€ôÑ¥µ•‘•±Ñ„ À¤(€€€€€€€€€€€¡½É¥é½¸€ôÑ¥µ•‘•±Ñ„ À¤(€€€€€€€€€€€‰¥¹Ì€ô9½¹”(€€€€€€€€€€€½¹‘¥Ñ¥½¸€ô9½¹”(€€€€€€€€€€€¹•ÕÑÉ…±}Á½±¥ä€ô9½¹”(€€€€€€€€€€€ÁÉ½™¥±”€ô9½¹”(€€€€€€€•±Í”è(€€€€€€€€€€€É…¥Í”QåÁ•ÉÉ½È¡˜‰Õ¹ÍÕÁÁ½ÉÑ•ØĞ…¡”ÍÑ…”èíÑåÁ”¡…¡•}Ù…±Õ”¤¹}}¹…µ•}}ôˆ¤(€€€€€€€ÍÑ…Ñ•µ•¹Ğ€ô€ (€€€€€€€€€€€Á½ÍÑÉ•ÍÅ±}¥¹Í•ÉĞ¡AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¤(€€€€€€€€€€€€¹Ù…±Õ•Ì (€€€€€€€€€€€€€€€¥õÕÕ¥Ğ ¤°(€€€€€€€€€€€€€€€…¡•}­•äõ…¡•}­•ä°(€€€€€€€€€€€€€€€…¡•}¹…µ•ÍÁ…”õ…¡•}¹…µ•ÍÁ…”°(€€€€€€€€€€€€€€€ÍÑ…ÑÕÌô‰½µÁ±•Ñ•ˆ°(€€€€€€€€€€€€€€€•Ù¥‘•¹•}­¥¹õ­¥¹°(€€€€€€€€€€€€€€€½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸õAI%Q%Y}}XÑ}=9QIQ}YIM%=8°(€€€€€€€€€€€€€€€‘…Ñ…Í•Ñ}™¥¹•ÉÁÉ¥¹Ğõ‘…Ñ…Í•Ñ}™¥¹•ÉÁÉ¥¹Ğ°(€€€€€€€€€€€€€€€…‘•¹•}Í•½¹‘Ìõ¥¹Ğ¡…‘•¹”¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤°(€€€€€€€€€€€€€€€¡½É¥é½¹}Í•½¹‘Ìõ¥¹Ğ¡¡½É¥é½¸¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤°(€€€€€€€€€€€€€€€‰¥¹}½Õ¹Ğõ‰¥¹Ì°(€€€€€€€€€€€€€€€½¹‘¥Ñ¥½¸õ½¹‘¥Ñ¥½¸°(€€€€€€€€€€€€€€€¹•ÕÑÉ…±}Á½±¥äõ¹•ÕÑÉ…±}Á½±¥ä°(€€€€€€€€€€€€€€€ÁÉ½™¥±”õÁÉ½™¥±”°(€€€€€€€€€€€€€€€™••}É…Ñ”õ™••}É…Ñ”°(€€€€€€€€€€€€€€€Í±¥ÁÁ…•}É…Ñ”õÍ±¥ÁÁ…•}É…Ñ”°(€€€€€€€€€€€€€€€Á…å±½…õÁ…å±½…°(€€€€€€€€€€€€€€€É•…Ñ•‘}…Ğõ‘…Ñ•Ñ¥µ”¹¹½Ü¡UQ¤°(€€€€€€€€€€€€¤(€€€€€€€€€€€€¹½¹}½¹™±¥Ñ}‘½}¹½Ñ¡¥¹œ¡¥¹‘•á}•±•µ•¹ÑÌõl‰…¡•}­•ä‰t¤(€€€€€€€€¤(€€€€€€€…İ…¥ĞÍ•ÍÍ¥½¸¹•á•ÕÑ”¡ÍÑ…Ñ•µ•¹Ğ¤(€€€¥˜É•¥µ•}É•ÍÕ±Ğ¥Ì¹½Ğ9½¹”…¹É•¥µ•}…¡•}¹…µ•ÍÁ…”¥Ì¹½Ğ9½¹”è(€€€€€€€™½È…¡•}­•ä°•Ù¥‘•¹”¥¸É•¥µ•}É•ÍÕ±Ğ¹…¡•}İÉ¥Ñ•Ì¹¥Ñ•µÌ ¤è(€€€€€€€€€€€Á…å±½…€ôì(€€€€€€€€€€€€€€€€‰½‰Í•ÉÙ…Ñ¥½¹Ìˆè•Ù¥‘•¹”¹½‰Í•ÉÙ…Ñ¥½¹Ì°(€€€€€€€€€€€€€€€€‰É½ÍÍ}µ½Ù•µ•¹Ğˆè}‘•¥µ…±}Ñ•áĞ¡•Ù¥‘•¹”¹É½ÍÍ}µ½Ù•µ•¹Ğ¤°(€€€€€€€€€€€€€€€€‰…±¥¹•‘}É½ÍÍ}µ½Ù•µ•¹Ğˆè}‘•¥µ…±}Ñ•áĞ (€€€€€€€€€€€€€€€€€€€•Ù¥‘•¹”¹…±¥¹•‘}É½ÍÍ}µ½Ù•µ•¹Ğ(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€‰¹•Ñ}µ½Ù•µ•¹Ğˆè}‘•¥µ…±}Ñ•áĞ¡•Ù¥‘•¹”¹¹•Ñ}µ½Ù•µ•¹Ğ¤°(€€€€€€€€€€€€€€€€‰‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”ˆè}‘•¥µ…±}Ñ•áĞ¡•Ù¥‘•¹”¹‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”¤°(€€€€€€€€€€€€€€€€‰‰•¹¡µ…É¬ˆè}‘•¥µ…±}Ñ•áĞ¡•Ù¥‘•¹”¹‰•¹¡µ…É¬¤°(€€€€€€€€€€€€€€€€‰‰•¹¡µ…É­}¥µÁÉ½Ù•µ•¹Ğˆè}‘•¥µ…±}Ñ•áĞ¡•Ù¥‘•¹”¹‰•¹¡µ…É­}¥µÁÉ½Ù•µ•¹Ğ¤°(€€€€€€€€€€€€€€€€‰ÍÑ…¹‘…É‘}•ÉÉ½Èˆè}‘•¥µ…±}Ñ•áĞ¡•Ù¥‘•¹”¹ÍÑ…¹‘…É‘}•ÉÉ½È¤°(€€€€€€€€€€€€€€€€‰ÍÑ…‰±”ˆè•Ù¥‘•¹”¹ÍÑ…‰±”°(€€€€€€€€€€€€€€€€‰•½¹½µ¥Œˆè•Ù¥‘•¹”¹•½¹½µ¥Œ°(€€€€€€€€€€€€€€€€‰Á…ÍÍ•Ìˆè•Ù¥‘•¹”¹Á…ÍÍ•Ì°(€€€€€€€€€€€ô(€€€€€€€€€€€…İ…¥ĞÍ•ÍÍ¥½¸¹•á•ÕÑ” (€€€€€€€€€€€€€€€Á½ÍÑÉ•ÍÅ±}¥¹Í•ÉĞ¡AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¤(€€€€€€€€€€€€€€€€¹Ù…±Õ•Ì (€€€€€€€€€€€€€€€€€€€¥õÕÕ¥Ğ ¤°(€€€€€€€€€€€€€€€€€€€…¡•}­•äõ…¡•}­•ä°(€€€€€€€€€€€€€€€€€€€…¡•}¹…µ•ÍÁ…”õÉ•¥µ•}…¡•}¹…µ•ÍÁ…”°(€€€€€€€€€€€€€€€€€€€ÍÑ…ÑÕÌô‰½µÁ±•Ñ•ˆ°(€€€€€€€€€€€€€€€€€€€•Ù¥‘•¹•}­¥¹ô‰É•¥µ•}½¹‘¥Ñ¥½¹•‘}Á¡…Í”ˆ°(€€€€€€€€€€€€€€€€€€€½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸õAI%Q%Y}}XĞÅ}=9QIQ}YIM%=8°(€€€€€€€€€€€€€€€€€€€‘…Ñ…Í•Ñ}™¥¹•ÉÁÉ¥¹Ğõ‘…Ñ…Í•Ñ}™¥¹•ÉÁÉ¥¹Ğ°(€€€€€€€€€€€€€€€€€€€…‘•¹•}Í•½¹‘ÌôÀ°(€€€€€€€€€€€€€€€€€€€¡½É¥é½¹}Í•½¹‘ÌôÀ°(€€€€€€€€€€€€€€€€€€€‰¥¹}½Õ¹Ğõ9½¹”°(€€€€€€€€€€€€€€€€€€€½¹‘¥Ñ¥½¸õ9½¹”°(€€€€€€€€€€€€€€€€€€€¹•ÕÑÉ…±}Á½±¥äõ9½¹”°(€€€€€€€€€€€€€€€€€€€ÁÉ½™¥±”ô‰É•¥µ•}½¹‘¥Ñ¥½¹•‘}ØĞÄˆ°(€€€€€€€€€€€€€€€€€€€™••}É…Ñ”õ™••}É…Ñ”°(€€€€€€€€€€€€€€€€€€€Í±¥ÁÁ…•}É…Ñ”õÍ±¥ÁÁ…•}É…Ñ”°(€€€€€€€€€€€€€€€€€€€Á…å±½…õÁ…å±½…°(€€€€€€€€€€€€€€€€€€€É•…Ñ•‘}…Ğõ‘…Ñ•Ñ¥µ”¹¹½Ü¡UQ¤°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€¹½¹}½¹™±¥Ñ}‘½}¹½Ñ¡¥¹œ¡¥¹‘•á}•±•µ•¹ÑÌõl‰…¡•}­•ä‰t¤(€€€€€€€€€€€€¤(€€€…İ…¥ĞÍ•ÍÍ¥½¸¹½µµ¥Ğ ¤(€€€É•ÑÕÉ¸AÉ•‘¥Ñ¥Ù•‘•á•ÕÑ¥½¸¡ÉÕ¹}¥°…±Í”°9½¹”°€À°€À¤(()…Íå¹Œ‘•˜±½…‘}ÁÉ•‘¥Ñ¥Ù•}•‘•}ØÑ}…¡” (€€€Í•ÍÍ¥½¸èÍå¹M•ÍÍ¥½¸°€¨°…¡•}¹…µ•ÍÁ…”èÍÑÈ(¤€´ø‘¥ÑmÍÑÈ°½‰©•Ñtè(€€€€ˆˆ‰1½…½¹±ä½µÁ±•Ñ••Ù¥‘•¹”™½È½¹”•á…ĞØĞ•áÁ•É¥µ•¹Ğ¥‘•¹Ñ¥Ñä¸ˆˆˆ((€€€É½İÌ€ô€ (€€€€€€€…İ…¥ĞÍ•ÍÍ¥½¸¹Í…±…ÉÌ (€€€€€€€€€€€Í•±•Ğ¡AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¤¹İ¡•É” (€€€€€€€€€€€€€€€AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¹…¡•}¹…µ•ÍÁ…”€ôô…¡•}¹…µ•ÍÁ…”°(€€€€€€€€€€€€€€€AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¹ÍÑ…ÑÕÌ€ôô€‰½µÁ±•Ñ•ˆ°(€€€€€€€€€€€€€€€AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¹½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸(€€€€€€€€€€€€€€€€ôôAI%Q%Y}}XÑ}=9QIQ}YIM%=8°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€€¤¹…±° ¤(€€€É•ÍÕ±Ğè‘¥ÑmÍÑÈ°½‰©•Ñt€ôíô(€€€™½ÈÉ½Ü¥¸É½İÌè(€€€€€€€ÑÉäè(€€€€€€€€€€€É•ÍÕ±ÑmÉ½Ü¹…¡•}­•åt€ô‘•½‘•}ØÑ}…¡•}Ù…±Õ”¡É½Ü¹Á…å±½…¤(€€€€€€€•á•ÁĞ€¡QåÁ•ÉÉ½È°Y…±Õ•ÉÉ½È¤è(€€€€€€€€€€€€Œµ…±™½Éµ•½È¥¹½µÁ±•Ñ”…¡”É½Ü¥Ì¹•Ù•ÈÉ•ÕÍ…‰±”¸(€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€É•ÑÕÉ¸É•ÍÕ±Ğ(()…Íå¹Œ‘•˜±½…‘}ÁÉ•‘¥Ñ¥Ù•}•‘•}ØĞÅ}…¡” (€€€Í•ÍÍ¥½¸èÍå¹M•ÍÍ¥½¸°€¨°…¡•}¹…µ•ÍÁ…”èÍÑÈ(¤€´ø‘¥ÑmÍÑÈ°XĞÅA¡…Í•Ù¥‘•¹•tè(€€€É½İÌ€ô€ (€€€€€€€…İ…¥ĞÍ•ÍÍ¥½¸¹Í…±…ÉÌ (€€€€€€€€€€€Í•±•Ğ¡AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¤¹İ¡•É” (€€€€€€€€€€€€€€€AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¹…¡•}¹…µ•ÍÁ…”€ôô…¡•}¹…µ•ÍÁ…”°(€€€€€€€€€€€€€€€AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¹ÍÑ…ÑÕÌ€ôô€‰½µÁ±•Ñ•ˆ°(€€€€€€€€€€€€€€€AÉ•‘¥Ñ¥Ù•‘•XÑ…¡•I•½É¹½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸(€€€€€€€€€€€€€€€€ôôAI%Q%Y}}XĞÅ}=9QIQ}YIM%=8°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€€¤¹…±° ¤(€€€É•ÍÕ±Ğè‘¥ÑmÍÑÈ°XĞÅA¡…Í•Ù¥‘•¹•t€ôíô(€€€™½ÈÉ½Ü¥¸É½İÌè(€€€€€€€ÑÉäè(€€€€€€€€€€€Á…å±½…€ôÉ½Ü¹Á…å±½…(€€€€€€€€€€€É•ÍÕ±ÑmÉ½Ü¹…¡•}­•åt€ôXĞÅA¡…Í•Ù¥‘•¹” (€€€€€€€€€€€€€€€½‰Í•ÉÙ…Ñ¥½¹Ìõ¥¹Ğ¡ÍÑÈ¡Á…å±½…‘l‰½‰Í•ÉÙ…Ñ¥½¹Ì‰t¤¤°(€€€€€€€€€€€€€€€É½ÍÍ}µ½Ù•µ•¹Ğõ}‘•¥µ…°¡Á…å±½…¹•Ğ ‰É½ÍÍ}µ½Ù•µ•¹Ğˆ¤¤°(€€€€€€€€€€€€€€€…±¥¹•‘}É½ÍÍ}µ½Ù•µ•¹Ğõ}‘•¥µ…°¡Á…å±½…¹•Ğ ‰…±¥¹•‘}É½ÍÍ}µ½Ù•µ•¹Ğˆ¤¤°(€€€€€€€€€€€€€€€¹•Ñ}µ½Ù•µ•¹Ğõ}‘•¥µ…°¡Á…å±½…¹•Ğ ‰¹•Ñ}µ½Ù•µ•¹Ğˆ¤¤°(€€€€€€€€€€€€€€€‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”õ}‘•¥µ…°¡Á…å±½…¹•Ğ ‰‘¥É•Ñ¥½¹…±}¡¥Ñ}É…Ñ”ˆ¤¤°(€€€€€€€€€€€€€€€‰•¹¡µ…É¬õ}‘•¥µ…°¡Á…å±½…¹•Ğ ‰‰•¹¡µ…É¬ˆ¤¤°(€€€€€€€€€€€€€€€‰•¹¡µ…É­}¥µÁÉ½Ù•µ•¹Ğõ}‘•¥µ…°¡Á…å±½…¹•Ğ ‰‰•¹¡µ…É­}¥µÁÉ½Ù•µ•¹Ğˆ¤¤°(€€€€€€€€€€€€€€€ÍÑ…¹‘…É‘}•ÉÉ½Èõ}‘•¥µ…°¡Á…å±½…¹•Ğ ‰ÍÑ…¹‘…É‘}•ÉÉ½Èˆ¤¤°(€€€€€€€€€€€€€€€ÍÑ…‰±”õ‰½½°¡Á…å±½…‘l‰ÍÑ…‰±”‰t¤°(€€€€€€€€€€€€€€€•½¹½µ¥Œõ‰½½°¡Á…å±½…‘l‰•½¹½µ¥Œ‰t¤°(€€€€€€€€€€€€€€€Á…ÍÍ•Ìõ‰½½°¡Á…å±½…‘l‰Á…ÍÍ•Ì‰t¤°(€€€€€€€€€€€€¤(€€€€€€€•á•ÁĞ€¡-•åÉÉ½È°QåÁ•ÉÉ½È°Y…±Õ•ÉÉ½È¤è(€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€É•ÑÕÉ¸É•ÍÕ±Ğ(()‘•˜}‘•¥µ…±}Ñ•áĞ¡Ù…±Õ”è•¥µ…°ğ9½¹”¤€´øÍÑÈğ9½¹”è(€€€É•ÑÕÉ¸ÍÑÈ¡Ù…±Õ”¤¥˜Ù…±Õ”¥Ì¹½Ğ9½¹”•±Í”9½¹”(()‘•˜}‘•¥µ…°¡Ù…±Õ”è½‰©•Ğ¤€´ø•¥µ…°ğ9½¹”è(€€€É•ÑÕÉ¸•¥µ…°¡ÍÑÈ¡Ù…±Õ”¤¤¥˜Ù…±Õ”¥Ì¹½Ğ9½¹”•±Í”9½¹”