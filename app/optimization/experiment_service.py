import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from time import perf_counter
from uuid import UUID, uuid4

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.experience import (
    CandidateHistoricalProfile,
    HistoricalEvaluation,
    build_candidate_profiles,
)
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import HistoricalDataset
from app.optimization.memory import (
    StrategyCandidateEvaluationRecord,
    StrategyCandidateRecord,
    StrategyRefinementRunRecord,
    complete_run,
    create_run,
    fail_run,
    find_experiment,
    insert_benchmark_evaluations,
    insert_candidate_evaluations,
    load_candidate_definitions,
    load_compatible_evidence,
    reclaim_abandoned_run,
    upsert_candidates,
)
from app.optimization.planning import (
    DEFAULT_PLANNER_ALLOCATION,
    REFINEMENT_PLANNER_VERSION,
    CandidateConfiguration,
    HistoricalCandidate,
    PlannerAllocation,
    SearchPlan,
    build_search_plan,
)
from app.optimization.refinement import (
    BenchmarkSegmentResult,
    CandidateSegmentResult,
    HistoricalRefinementResult,
    RefinementBudget,
    RefinementCandidate,
    RefinementSegment,
    candidate_for,
    canonical_candidate_parameters,
    generate_joint_candidate_pool,
    refine_historical_strategy,
    split_feature_matrix,
)
from app.optimization.simulation import (
    DEFAULT_DECISION_INTERVAL,
    SIMULATION_ENGINE_VERSION,
)
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.scores.calculator import BaselineTechnicalParameters
from app.strategies.baseline_long_only import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
    BaselineLongOnlyParameters,
)


class ExperimentMemoryError(RuntimeError):
    """Raised when experiment memory cannot safely complete or reuse a run."""


# A persisted running row acts as a lease. A two-hour window accommodates the current
# bounded synchronous refinement while ensuring a killed process cannot block forever.
ABANDONED_EXPERIMENT_TIMEOUT = timedelta(hours=2)


@dataclass(frozen=True, slots=True)
class ReusedExperiment:
    experiment_key: str
    run_id: UUID
    reused_existing_experiment: bool
    result_digest: str
    recommendation_status: str
    recommended_candidate_id: str | None
    evaluations_reused: int
    evaluations_computed: int = 0
    compute_avoided_estimate: int = 0


@dataclass(frozen=True, slots=True)
class ExperimentExecution:
    result: HistoricalRefinementResult | None
    reused: ReusedExperiment | None


async def execute_refinement(
    *,
    session: AsyncSession,
    dataset: HistoricalDataset,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    budget: RefinementBudget,
    persist: bool,
    force_recompute: bool,
    started_at: datetime,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
    planner_allocation: PlannerAllocation = DEFAULT_PLANNER_ALLOCATION,
) -> ExperimentExecution:
    if started_at.tzinfo is not UTC:
        raise ExperimentMemoryError("experiment start and history cutoff must use UTC")
    matrix = build_historical_feature_matrix(dataset)
    _, segments = split_feature_matrix(matrix, decision_interval=decision_interval)
    experiment_key = build_experiment_key(
        dataset=dataset,
        segments=segments,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        budget=budget,
        decision_interval=decision_interval,
        planner_allocation=planner_allocation,
    )
    existing = await find_experiment(session, experiment_key)
    if existing is not None and existing.status == "completed" and not force_recompute:
        payload = existing.result_payload or {}
        return ExperimentExecution(
            None,
            ReusedExperiment(
                experiment_key,
                existing.id,
                True,
                str(payload.get("result_digest", "")),
                existing.recommendation_status or "",
                existing.recommended_candidate_id,
                _stored_int((existing.counts or {}).get("evaluations_computed")),
                0,
                _stored_int((existing.counts or {}).get("evaluations_computed")),
            ),
        )
    run = existing
    if existing is not None and existing.status == "running":
        if (
            not persist
            or started_at - existing.started_at <= ABANDONED_EXPERIMENT_TIMEOUT
        ):
            raise ExperimentMemoryError("an identical experiment is already running")
        run = await reclaim_abandoned_run(
            session,
            run_id=existing.id,
            previous_started_at=existing.started_at,
            stale_before=started_at - ABANDONED_EXPERIMENT_TIMEOUT,
            restarted_at=started_at,
        )
        if run is None:
            await session.rollback()
            raise ExperimentMemoryError(
                "an identical experiment was concurrently reclaimed"
            )
        await session.commit()

    evidence_rows = await load_compatible_evidence(
        session,
        score_model_name=MODEL_NAME,
        score_model_version=MODEL_VERSION,
        strategy_name=STRATEGY_NAME,
        strategy_version=STRATEGY_VERSION,
        simulation_engine_version=SIMULATION_ENGINE_VERSION,
        history_cutoff=started_at,
    )
    historical, profiles = await _historical_candidates(
        session, evidence_rows, dataset.fingerprint
    )
    pool = tuple(_configuration(value) for value in generate_joint_candidate_pool())
    # The exact baseline is selected by identity, independent of grid order.
    baseline = next(
        item
        for item in pool
        if candidate_for(
            item.score_parameters, item.strategy_parameters
        ).is_production_baseline
    )
    plan = (
        await _stored_plan(session, existing, planner_allocation)
        if existing is not None and existing.status == "completed" and force_recompute
        else build_search_plan(
            historical=historical,
            novel_pool=pool,
            max_candidates=budget.max_joint_candidates,
            production_baseline=baseline,
            allocation=planner_allocation,
        )
    )
    if persist and existing is None:
        run = await create_run(
            session,
            _run_values(
                experiment_key,
                dataset,
                segments,
                initial_cash,
                fee_rate,
                slippage_rate,
                budget,
                decision_interval,
                planner_allocation,
                started_at,
            ),
        )
        if run is None:
            await session.rollback()
            concurrent = await find_experiment(session, experiment_key)
            if concurrent is not None and concurrent.status == "completed":
                payload = concurrent.result_payload or {}
                return ExperimentExecution(
                    None,
                    ReusedExperiment(
                        experiment_key,
                        concurrent.id,
                        True,
                        str(payload.get("result_digest", "")),
                        concurrent.recommendation_status or "",
                        concurrent.recommended_candidate_id,
                        _stored_int(
                            (concurrent.counts or {}).get("evaluations_computed")
                        ),
                        0,
                        _stored_int(
                            (concurrent.counts or {}).get("evaluations_computed")
                        ),
                    ),
                )
            raise ExperimentMemoryError("an identical experiment is already running")
        await session.commit()
    elif persist and existing is not None and existing.status == "failed":
        await session.execute(
            update(StrategyRefinementRunRecord)
            .where(StrategyRefinementRunRecord.id == existing.id)
            .values(
                status="running",
                started_at=started_at,
                failed_at=None,
                failure_reason=None,
            )
        )
        await session.commit()

    clock_started = perf_counter()
    try:
        result = refine_historical_strategy(
            dataset=dataset,
            feature_matrix=matrix,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            budget=budget,
            decision_interval=decision_interval,
            search_plan=plan,
        )
        recommended_profile = next(
            (
                item
                for item in profiles
                if result.recommendation is not None
                and item.stable_candidate_id
                == result.recommendation.candidate.stable_candidate_id
            ),
            None,
        )
        influence = _history_explanation(plan, recommended_profile)
        result = replace(
            result,
            experiment_key=experiment_key,
            persisted=persist,
            history_run_count=len({row[1].id for row in evidence_rows}),
            history_cutoff=started_at,
            compatible_profile_count=len(profiles),
            search_plan=plan,
            recommended_historical_profile=recommended_profile,
            history_influence_explanation=influence,
            evidence_classification=(
                recommended_profile.evidence_classification
                if recommended_profile is not None
                else None
            ),
            recommendation_explanation=(
                f"{result.recommendation_explanation} {influence}"
            ),
        )
        digest = _result_digest(result)
        if existing is not None and existing.status == "completed" and force_recompute:
            old_digest = str((existing.result_payload or {}).get("result_digest", ""))
            if old_digest != digest:
                raise ExperimentMemoryError(
                    "force recompute diverged from the stored deterministic result"
                )
            await session.execute(
                update(StrategyRefinementRunRecord)
                .where(StrategyRefinementRunRecord.id == existing.id)
                .values(last_verified_at=started_at)
            )
            await session.commit()
            return ExperimentExecution(
                replace(result, persisted=True),
                None,
            )
        if persist:
            if run is None:
                raise ExperimentMemoryError("persisted refinement run was not claimed")
            evaluated_at = datetime.now(UTC)
            candidates, evaluations = _candidate_rows(result, run.id, evaluated_at)
            await upsert_candidates(session, candidates)
            await insert_candidate_evaluations(session, evaluations)
            await insert_benchmark_evaluations(
                session, _benchmark_rows(result, run.id, evaluated_at)
            )
            await complete_run(
                session,
                run.id,
                {
                    "recommendation_status": result.recommendation_status.value,
                    "recommended_candidate_id": (
                        result.recommendation.candidate.stable_candidate_id
                        if result.recommendation
                        else None
                    ),
                    "recommendation_explanation": result.recommendation_explanation,
                    "counts": _counts(result),
                    "search_plan": _plan_payload(plan),
                    "result_payload": {"result_digest": digest},
                    "elapsed_seconds": Decimal(str(perf_counter() - clock_started)),
                    "completed_at": evaluated_at,
                },
            )
            await session.commit()
        return ExperimentExecution(result, None)
    except Exception as error:
        await session.rollback()
        if persist and run is not None and run.status != "completed":
            await fail_run(
                session,
                run.id,
                failed_at=datetime.now(UTC),
                reason=f"{type(error).__name__}: {error}",
            )
            await session.commit()
        raise


def build_experiment_key(
    *,
    dataset: HistoricalDataset,
    segments: tuple[RefinementSegment, RefinementSegment, RefinementSegment],
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    budget: RefinementBudget,
    decision_interval: timedelta,
    planner_allocation: PlannerAllocation,
) -> str:
    instrument = dataset.identity.instrument
    payload = {
        "dataset_fingerprint": dataset.fingerprint,
        "instrument": [
            instrument.asset.value,
            instrument.provider.value,
            instrument.market_type.value,
            instrument.interval.value,
        ],
        "requested_range": [
            dataset.identity.range.start.isoformat(),
            dataset.identity.range.end.isoformat(),
        ],
        "segments": [asdict(value) for value in segments],
        "score": [MODEL_NAME, MODEL_VERSION],
        "strategy": [STRATEGY_NAME, STRATEGY_VERSION],
        "simulation_engine_version": SIMULATION_ENGINE_VERSION,
        "initial_cash": str(initial_cash),
        "account_currency": DEFAULT_ACCOUNT_CURRENCY,
        "fee_rate": str(fee_rate),
        "slippage_rate": str(slippage_rate),
        "decision_interval_seconds": int(decision_interval.total_seconds()),
        "budget": asdict(budget),
        "planner_version": REFINEMENT_PLANNER_VERSION,
        "planner_allocation": _allocation_payload(planner_allocation),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode()).hexdigest()


async def _historical_candidates(
    session: AsyncSession,
    rows: list[
        tuple[StrategyCandidateEvaluationRecord, StrategyRefinementRunRecord, Decimal]
    ],
    current_fingerprint: str,
) -> tuple[tuple[HistoricalCandidate, ...], tuple[CandidateHistoricalProfile, ...]]:
    evidence = tuple(
        HistoricalEvaluation(
            evaluation.stable_candidate_id,
            str(run.id),
            Instrument(
                Asset(run.asset),
                Provider(run.provider),
                MarketType(run.market_type),
                Interval(run.interval),
            ),
            evaluation.segment,
            evaluation.total_return,
            baseline_return,
            evaluation.max_drawdown,
            evaluation.trade_count,
            evaluation.fees_paid,
            evaluation.eligible,
            evaluation.evaluated_at,
            evaluation.eligible if evaluation.segment == "test" else None,
        )
        for evaluation, run, baseline_return in rows
    )
    profiles = build_candidate_profiles(evidence)
    definitions = await load_candidate_definitions(
        session, tuple(item.stable_candidate_id for item in profiles)
    )
    current_ids = {
        evaluation.stable_candidate_id
        for evaluation, run, _ in rows
        if run.dataset_fingerprint == current_fingerprint
    }
    historical = tuple(
        HistoricalCandidate(
            _configuration_from_record(definitions[profile.stable_candidate_id]),
            profile,
            profile.stable_candidate_id in current_ids,
        )
        for profile in profiles
        if profile.stable_candidate_id in definitions
    )
    return historical, profiles


def _run_values(
    experiment_key: str,
    dataset: HistoricalDataset,
    segments: tuple[RefinementSegment, RefinementSegment, RefinementSegment],
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    budget: RefinementBudget,
    decision_interval: timedelta,
    allocation: PlannerAllocation,
    started_at: datetime,
) -> dict[str, object]:
    training, validation, test = segments
    instrument = dataset.identity.instrument
    return {
        "experiment_key": experiment_key,
        "asset": instrument.asset.value,
        "provider": instrument.provider.value,
        "market_type": instrument.market_type.value,
        "interval": instrument.interval.value,
        "dataset_fingerprint": dataset.fingerprint,
        "requested_from": dataset.identity.range.start,
        "requested_to": dataset.identity.range.end,
        "score_model_name": MODEL_NAME,
        "score_model_version": MODEL_VERSION,
        "strategy_name": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "simulation_engine_version": SIMULATION_ENGINE_VERSION,
        "initial_cash": initial_cash,
        "account_currency": DEFAULT_ACCOUNT_CURRENCY,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "decision_interval_seconds": int(decision_interval.total_seconds()),
        "training_from": training.first_boundary,
        "training_to": training.last_boundary,
        "validation_from": validation.first_boundary,
        "validation_to": validation.last_boundary,
        "test_from": test.first_boundary,
        "test_to": test.last_boundary,
        "refinement_budget": asdict(budget),
        "planner_allocation": _allocation_payload(allocation),
        "search_planner_version": REFINEMENT_PLANNER_VERSION,
        "history_cutoff": started_at,
        "started_at": started_at,
    }


def _candidate_rows(
    result: HistoricalRefinementResult, run_id: UUID, evaluated_at: datetime
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    entries: list[
        tuple[CandidateSegmentResult, str, int | None, bool | None, str | None]
    ] = []
    entries.extend(
        (item, "score_screening", rank, None, None)
        for rank, item in enumerate(result.score_screening_ranking, 1)
    )
    entries.extend(
        (item, "joint_training", rank, None, None)
        for rank, item in enumerate(result.training_ranking, 1)
    )
    entries.extend(
        (
            item,
            "validation",
            rank,
            item.total_return > result.production_baseline.validation.total_return,
            None,
        )
        for rank, item in enumerate(result.validation_results, 1)
    )
    if result.recommendation is not None:
        entries.append(
            (
                result.recommendation.test,
                "test",
                1,
                result.recommendation_status.value.endswith("confirmed"),
                None,
            )
        )
    entries.extend(
        (item.ablated, "controlled_ablation", None, None, item.disabled_signal)
        for item in result.ablation_summary
    )
    candidates = {
        item.candidate.stable_candidate_id: item.candidate for item, *_ in entries
    }
    candidate_rows = [
        _candidate_definition(value, evaluated_at) for value in candidates.values()
    ]
    evaluation_rows: list[dict[str, object]] = [
        {
            "id": uuid4(),
            "refinement_run_id": run_id,
            "stable_candidate_id": item.candidate.stable_candidate_id,
            "phase": phase,
            "segment": item.segment.value,
            "disabled_signal": disabled,
            "total_return": item.total_return,
            "absolute_pnl": item.absolute_pnl,
            "max_drawdown": item.max_drawdown,
            "trade_count": item.trade_count,
            "fees_paid": item.fees_paid,
            "final_equity": item.final_equity,
            "return_over_drawdown": item.return_over_drawdown,
            "rank": rank,
            "eligible": eligible,
            "rejection_reason": None,
            "evaluated_at": evaluated_at,
        }
        for item, phase, rank, eligible, disabled in entries
    ]
    return candidate_rows, evaluation_rows


def _benchmark_rows(
    result: HistoricalRefinementResult, run_id: UUID, evaluated_at: datetime
) -> list[dict[str, object]]:
    values: list[tuple[str, BenchmarkSegmentResult | CandidateSegmentResult]] = []
    for segment in ("training", "validation", "test"):
        values.append(
            ("production_baseline", getattr(result.production_baseline, segment))
        )
        values.append(("buy_and_hold", getattr(result.buy_and_hold, segment)))
    return [
        {
            "id": uuid4(),
            "refinement_run_id": run_id,
            "benchmark": name,
            "segment": item.segment.value,
            "total_return": item.total_return,
            "absolute_pnl": item.absolute_pnl,
            "max_drawdown": item.max_drawdown,
            "trade_count": item.trade_count,
            "fees_paid": item.fees_paid,
            "final_equity": item.final_equity,
            "return_over_drawdown": item.return_over_drawdown,
            "evaluated_at": evaluated_at,
        }
        for name, item in values
    ]


def _candidate_definition(
    candidate: RefinementCandidate, created_at: datetime
) -> dict[str, object]:
    score = _score_payload(candidate.score_parameters)
    strategy = _strategy_payload(candidate.strategy_parameters)
    return {
        "stable_candidate_id": candidate.stable_candidate_id,
        "score_model_name": MODEL_NAME,
        "score_model_version": MODEL_VERSION,
        "strategy_name": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "score_parameters": score,
        "strategy_parameters": strategy,
        "canonical_parameters": {
            "ordered_values": list(
                canonical_candidate_parameters(
                    candidate.score_parameters, candidate.strategy_parameters
                )
            )
        },
        "created_at": created_at,
    }


def _configuration(candidate: RefinementCandidate) -> CandidateConfiguration:
    return CandidateConfiguration(
        candidate.stable_candidate_id,
        candidate.score_parameters,
        candidate.strategy_parameters,
    )


def _configuration_from_record(
    record: StrategyCandidateRecord,
) -> CandidateConfiguration:
    score = record.score_parameters
    strategy = record.strategy_parameters
    return CandidateConfiguration(
        record.stable_candidate_id,
        BaselineTechnicalParameters(
            Decimal(str(score["trend_weight"])),
            Decimal(str(score["rsi_weight"])),
            Decimal(str(score["return_1_weight"])),
            Decimal(str(score["return_12_weight"])),
            Decimal(str(score["volume_confirmation_weight"])),
        ),
        BaselineLongOnlyParameters(
            Decimal(str(strategy["entry_threshold"])),
            Decimal(str(strategy["exit_threshold"])),
            Decimal(str(strategy["open_position_target_percentage"])),
        ),
    )


def _score_payload(value: BaselineTechnicalParameters) -> dict[str, str]:
    return {
        name: str(getattr(value, name))
        for name in (
            "trend_weight",
            "rsi_weight",
            "return_1_weight",
            "return_12_weight",
            "volume_confirmation_weight",
        )
    }


def _strategy_payload(value: BaselineLongOnlyParameters) -> dict[str, str]:
    return {
        name: str(getattr(value, name))
        for name in (
            "entry_threshold",
            "exit_threshold",
            "open_position_target_percentage",
        )
    }


def _allocation_payload(value: PlannerAllocation) -> dict[str, object]:
    return {
        "known_rechecks": str(value.known_rechecks),
        "promising_neighborhoods": str(value.promising_neighborhoods),
        "uncertain_rechecks": str(value.uncertain_rechecks),
        "novel_exploration": str(value.novel_exploration),
        "reconsider_rejected_after_runs": value.reconsider_rejected_after_runs,
    }


def _plan_payload(value: SearchPlan | None) -> dict[str, object]:
    return {
        "planner_version": REFINEMENT_PLANNER_VERSION,
        "fallback_without_history": value is None,
        "counts": value.counts if value else {},
        "known_rechecks": (
            [item.stable_candidate_id for item in value.known_rechecks] if value else []
        ),
        "promising_neighborhoods": (
            [item.stable_candidate_id for item in value.promising_neighborhoods]
            if value
            else []
        ),
        "uncertain_rechecks": (
            [item.stable_candidate_id for item in value.uncertain_rechecks]
            if value
            else []
        ),
        "novel_exploration": (
            [item.stable_candidate_id for item in value.novel_exploration]
            if value
            else []
        ),
    }


def _counts(result: HistoricalRefinementResult) -> dict[str, object]:
    return {
        **asdict(result.counts),
        "evaluations_computed": result.evaluations_computed,
        "evaluations_reused": result.evaluations_reused,
        "compute_avoided_estimate": result.compute_avoided_estimate,
    }


def _result_digest(result: HistoricalRefinementResult) -> str:
    payload = {
        "recommendation_status": result.recommendation_status.value,
        "recommended_candidate_id": result.recommendation.candidate.stable_candidate_id
        if result.recommendation
        else None,
        "baseline": [
            str(getattr(result.production_baseline, segment).total_return)
            for segment in ("training", "validation", "test")
        ],
        "buy_and_hold": [
            str(getattr(result.buy_and_hold, segment).total_return)
            for segment in ("training", "validation", "test")
        ],
        "counts": asdict(result.counts),
    }
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _history_explanation(
    plan: SearchPlan | None, profile: CandidateHistoricalProfile | None
) -> str:
    if plan is None:
        return (
            "No compatible completed history was available; the legacy deterministic "
            "search was used."
        )
    counts = plan.counts
    profile_text = (
        f" The recommended candidate is {profile.evidence_classification.value} across "
        f"{profile.compatible_run_count} compatible runs."
        if profile is not None
        else " The recommended candidate has no prior compatible profile."
    )
    return (
        "History influenced this run by rechecking "
        f"{counts['known_rechecks']} known candidates, generating "
        f"{counts['promising_neighborhoods']} promising neighbors, reconsidering "
        f"{counts['uncertain_rechecks']} uncertain candidates, and exploring "
        f"{counts['novel_exploration']} novel candidates."
        f"{profile_text} Historical evidence does not guarantee profitability."
    )


def _stored_int(value: object | None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return 0
    return int(value)


async def _stored_plan(
    session: AsyncSession,
    run: StrategyRefinementRunRecord,
    allocation: PlannerAllocation,
) -> SearchPlan | None:
    payload = run.search_plan or {}
    if payload.get("fallback_without_history") is True:
        return None
    categories = (
        "known_rechecks",
        "promising_neighborhoods",
        "uncertain_rechecks",
        "novel_exploration",
    )
    ids = tuple(
        candidate_id
        for category in categories
        for candidate_id in _stored_list(payload.get(category))
    )
    definitions = await load_candidate_definitions(session, ids)
    if len(definitions) != len(set(ids)):
        raise ExperimentMemoryError(
            "stored search plan has missing candidate definitions"
        )
    values = tuple(
        tuple(
            _configuration_from_record(definitions[candidate_id])
            for candidate_id in _stored_list(payload.get(category))
        )
        for category in categories
    )
    return SearchPlan(
        REFINEMENT_PLANNER_VERSION,
        allocation,
        values[0],
        values[1],
        values[2],
        values[3],
    )


def _stored_list(value: object | None) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ExperimentMemoryError("stored search plan is malformed")
    return tuple(value)
