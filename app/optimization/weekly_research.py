import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.optimization.edge_guided import (
    EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
    EDGE_GUIDED_RANKING_CONTRACT_VERSION,
    EDGE_GUIDED_STRATEGY_CONTRACT_VERSION,
    evaluate_edge_guided_candidates,
    generate_edge_guided_candidates,
)
from app.optimization.edge_guided_memory import (
    EDGE_GUIDED_PERSISTENCE_CONTRACT_VERSION,
    EdgeGuidedExecution,
    edge_guided_experiment_key,
    find_reusable_weekly_research_run,
    load_reusable_edge_guided_stages,
    persist_edge_guided_result,
)
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRevision,
    read_historical_dataset,
    read_historical_dataset_revision,
)
from app.optimization.predictive_edge_engine_v4 import analyze_predictive_edges_v4
from app.optimization.predictive_edge_service import (
    find_reusable_predictive_edge_run,
    load_predictive_edge_v4_cache,
    persist_predictive_edge_v4_result,
    predictive_edge_v4_cache_namespace,
    predictive_edge_v4_experiment_key,
)
from app.optimization.predictive_edge_v4 import (
    PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
)
from app.optimization.predictive_edges import (
    PREDICTOR_VERSION,
    PredictiveEdgeConfiguration,
    build_predictive_windows,
)
from app.paper.config import SimulationConfiguration

WEEKLY_RESEARCH_CONTRACT_VERSION = 1
WEEKLY_RESEARCH_ADVISORY_LOCK = 0x4152474F53524553


class WeeklyResearchError(RuntimeError):
    """Raised when weekly Strategy research cannot complete operationally."""


@dataclass(frozen=True, slots=True)
class WeeklyResearchWork:
    candles_loaded: int = 0
    feature_rows_built: int = 0
    evaluations_computed: int = 0
    evaluations_reused: int = 0
    simulations_avoided: int = 0


@dataclass(frozen=True, slots=True)
class WeeklyInstrumentResearch:
    instrument: Instrument
    reused: bool
    execution: EdgeGuidedExecution
    work: WeeklyResearchWork
    timings: dict[str, Decimal]
    diagnostics: dict[str, object] = field(default_factory=dict)


def weekly_boundary(now: datetime) -> datetime:
    """Return the most recent Sunday 03:17 UTC research boundary."""

    if now.tzinfo is not UTC:
        raise WeeklyResearchError("weekly research time must use UTC")
    sunday = (now.weekday() + 1) % 7
    candidate = (now - timedelta(days=sunday)).replace(
        hour=3, minute=17, second=0, microsecond=0
    )
    return candidate if candidate <= now else candidate - timedelta(days=7)


def weekly_research_reuse_key(
    *,
    identity: HistoricalDatasetIdentity,
    revision: HistoricalDatasetRevision,
    configuration: PredictiveEdgeConfiguration,
    simulation: SimulationConfiguration,
    initial_cash: Decimal,
    maximum_candidates: int,
    boundary: datetime,
) -> str:
    payload = {
        "weekly_research_contract_version": WEEKLY_RESEARCH_CONTRACT_VERSION,
        "predictor_version": PREDICTOR_VERSION,
        "edge_algorithm_version": PREDICTIVE_EDGE_V4_CONTRACT_VERSION,
        "strategy_contract_version": EDGE_GUIDED_STRATEGY_CONTRACT_VERSION,
        "ranking_contract_version": EDGE_GUIDED_RANKING_CONTRACT_VERSION,
        "qualification_contract_version": EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
        "persistence_contract_version": EDGE_GUIDED_PERSISTENCE_CONTRACT_VERSION,
        "instrument": [
            identity.instrument.asset.value,
            identity.instrument.provider.value,
            identity.instrument.market_type.value,
            identity.instrument.interval.value,
        ],
        "range": [identity.range.start.isoformat(), identity.range.end.isoformat()],
        "dataset_revision": revision.fingerprint,
        "configuration": asdict(configuration),
        "simulation": asdict(simulation),
        "initial_cash": str(initial_cash),
        "maximum_candidates": maximum_candidates,
        "weekly_boundary": boundary.isoformat(),
    }
    encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


async def try_weekly_research_lock(session: AsyncSession) -> bool:
    return bool(
        await session.scalar(
            text("SELECT pg_try_advisory_lock(:key)"),
            {"key": WEEKLY_RESEARCH_ADVISORY_LOCK},
        )
    )


async def release_weekly_research_lock(session: AsyncSession) -> None:
    await session.execute(
        text("SELECT pg_advisory_unlock(:key)"),
        {"key": WEEKLY_RESEARCH_ADVISORY_LOCK},
    )


async def execute_weekly_instrument_research(
    *,
    session: AsyncSession,
    identity: HistoricalDatasetIdentity,
    configuration: PredictiveEdgeConfiguration,
    simulation: SimulationConfiguration,
    initial_cash: Decimal,
    maximum_candidates: int,
    boundary: datetime,
    persist: bool,
    force_recompute: bool = False,
) -> WeeklyInstrumentResearch:
    """Execute or reuse one bounded research graph for one catalog instrument."""

    total_started = perf_counter()
    revision_started = perf_counter()
    revision = await read_historical_dataset_revision(
        session=session, identity=identity, now=boundary
    )
    reuse_key = weekly_research_reuse_key(
        identity=identity,
        revision=revision,
        configuration=configuration,
        simulation=simulation,
        initial_cash=initial_cash,
        maximum_candidates=maximum_candidates,
        boundary=boundary,
    )
    revision_elapsed = perf_counter() - revision_started
    if persist and not force_recompute:
        reused = await find_reusable_weekly_research_run(session, reuse_key)
        if reused is not None:
            elapsed = perf_counter() - total_started
            return WeeklyInstrumentResearch(
                identity.instrument,
                True,
                reused,
                WeeklyResearchWork(
                    evaluations_reused=reused.result.evaluated_count,
                    simulations_avoided=reused.compute_avoided_estimate,
                ),
                {
                    "identity_lookup": Decimal(str(revision_elapsed)),
                    "dataset_load": Decimal(0),
                    "feature_matrix": Decimal(0),
                    "edge_analysis": Decimal(0),
                    "candidate_evaluation": Decimal(0),
                    "persistence": Decimal(0),
                    "total": Decimal(str(elapsed)),
                },
            )

    dataset_started = perf_counter()
    dataset = await read_historical_dataset(
        session=session, identity=identity, now=boundary
    )
    dataset_elapsed = perf_counter() - dataset_started
    matrix_started = perf_counter()
    matrix = build_historical_feature_matrix(dataset)
    matrix_elapsed = perf_counter() - matrix_started
    windows = build_predictive_windows(
        dataset=dataset, matrix=matrix, configuration=configuration
    )
    edge_key = predictive_edge_v4_experiment_key(
        dataset_fingerprint=dataset.fingerprint,
        instrument=identity.instrument,
        range_start=identity.range.start,
        range_end=identity.range.end,
        fee_rate=simulation.fee_rate,
        slippage_rate=simulation.slippage_rate,
    )
    edge_cache_namespace = predictive_edge_v4_cache_namespace(
        instrument=identity.instrument,
        fee_rate=simulation.fee_rate,
        slippage_rate=simulation.slippage_rate,
    )
    edge_started = perf_counter()
    edge_execution = (
        await find_reusable_predictive_edge_run(session, edge_key)
        if persist and not force_recompute
        else None
    )
    stage_cache = (
        await load_predictive_edge_v4_cache(
            session, cache_namespace=edge_cache_namespace
        )
        if persist and not force_recompute
        else {}
    )
    edge_result = analyze_predictive_edges_v4(
        matrix=matrix,
        windows=windows,
        simulation=simulation,
        cache_namespace=edge_cache_namespace,
        stage_cache=stage_cache,
        force_recompute=force_recompute,
    )
    if persist and force_recompute:
        edge_execution = await find_reusable_predictive_edge_run(session, edge_key)
    if persist and (edge_execution is None or edge_execution.run_id is None):
        edge_execution = await persist_predictive_edge_v4_result(
            session=session,
            experiment_key=edge_key,
            cache_namespace=edge_cache_namespace,
            instrument=identity.instrument,
            range_start=identity.range.start,
            range_end=identity.range.end,
            dataset_fingerprint=dataset.fingerprint,
            windows=windows,
            result=edge_result,
            fee_rate=simulation.fee_rate,
            slippage_rate=simulation.slippage_rate,
            started_at=boundary,
        )
    edge_elapsed = perf_counter() - edge_started
    candidate_started = perf_counter()
    generated_candidates = generate_edge_guided_candidates(
        hypotheses=edge_result.hypotheses,
        simulation=simulation,
        decision_interval=configuration.decision_interval,
        maximum_candidates=maximum_candidates,
    )
    reusable_stages = (
        {}
        if force_recompute
        else await load_reusable_edge_guided_stages(
            session,
            tuple(item.candidate_fingerprint for item in generated_candidates),
            windows,
        )
    )
    result = evaluate_edge_guided_candidates(
        dataset=dataset,
        feature_matrix=matrix,
        hypotheses=edge_result.hypotheses,
        initial_cash=initial_cash,
        simulation=simulation,
        decision_interval=configuration.decision_interval,
        maximum_candidates=maximum_candidates,
        windows=windows,
        thresholds=configuration.thresholds,
        reusable_stage_evidence=reusable_stages,
    )
    candidate_elapsed = perf_counter() - candidate_started
    result = replace(
        result,
        funnel={
            "cheap_relationships_generated": edge_result.training_counters.generated,
            "cheap_relationships_computed": edge_result.training_counters.computed,
            "cheap_relationships_reused": edge_result.training_counters.reused,
            "cheap_relationships_retained": edge_result.training_counters.retained,
            "validation_configurations_computed": (
                edge_result.validation_counters.computed
            ),
            "validation_configurations_reused": (
                edge_result.validation_counters.reused
            ),
            "validation_configurations_retained": (
                edge_result.validation_counters.retained
            ),
            "test_configurations_computed": edge_result.test_counters.computed,
            "test_configurations_reused": edge_result.test_counters.reused,
            "confirmed_edges": len(edge_result.confirmed_edges),
            "diagnostic_only_edges": len(edge_result.diagnostic_only_edges),
            "near_misses": len(edge_result.near_misses),
            **result.funnel,
        },
        timings={
            "identity_lookup": Decimal(str(revision_elapsed)),
            "dataset_load": Decimal(str(dataset_elapsed)),
            "feature_matrix": Decimal(str(matrix_elapsed)),
            "edge_analysis": Decimal(str(edge_elapsed)),
            "candidate_evaluation": Decimal(str(candidate_elapsed)),
            "persistence": Decimal(0),
            "total": Decimal(str(perf_counter() - total_started)),
        },
    )
    persistence_started = perf_counter()
    if persist:
        if edge_execution is None or edge_execution.run_id is None:
            raise WeeklyResearchError("predictive evidence was not persisted")
        experiment_key = edge_guided_experiment_key(
            predictive_edge_run_id=edge_execution.run_id,
            dataset_fingerprint=dataset.fingerprint,
            instrument=identity.instrument,
            range_start=identity.range.start,
            range_end=identity.range.end,
            windows=configuration.windows,
            initial_cash=initial_cash,
            simulation=simulation,
            decision_interval=configuration.decision_interval,
            maximum_candidates=maximum_candidates,
            hypothesis_fingerprints=tuple(
                item.hypothesis_id for item in edge_result.hypotheses
            ),
        )
        execution = await persist_edge_guided_result(
            session=session,
            experiment_key=experiment_key,
            predictive_edge_run_id=edge_execution.run_id,
            instrument=identity.instrument,
            range_start=identity.range.start,
            range_end=identity.range.end,
            initial_cash=initial_cash,
            simulation=simulation,
            decision_interval=configuration.decision_interval,
            started_at=boundary,
            result=result,
            reuse_key=reuse_key,
            dataset_revision=asdict(revision),
        )
    else:
        if session.new or session.dirty or session.deleted:
            raise WeeklyResearchError("non-persisted research mutated the session")
        execution = EdgeGuidedExecution(
            None,
            reuse_key,
            False,
            result,
            0,
            result.evaluated_count,
            0,
        )
    persistence_elapsed = perf_counter() - persistence_started
    total_elapsed = perf_counter() - total_started
    return WeeklyInstrumentResearch(
        identity.instrument,
        False,
        execution,
        WeeklyResearchWork(
            candles_loaded=dataset.actual_candle_count,
            feature_rows_built=len(matrix.rows),
            evaluations_computed=result.evaluated_count,
        ),
        {
            "identity_lookup": Decimal(str(revision_elapsed)),
            "dataset_load": Decimal(str(dataset_elapsed)),
            "feature_matrix": Decimal(str(matrix_elapsed)),
            "edge_analysis": Decimal(str(edge_elapsed)),
            "candidate_evaluation": Decimal(str(candidate_elapsed)),
            "persistence": Decimal(str(persistence_elapsed)),
            "total": Decimal(str(total_elapsed)),
        },
        {
            "rejection_counts": edge_result.rejection_counts,
            "bottleneck_stage": edge_result.bottleneck_stage,
            "strongest_observed_signal": edge_result.strongest_observed_signal,
            "diagnostic_conclusion": edge_result.conclusion.value,
            "outcome_explanation": edge_result.outcome_explanation,
        },
    )
