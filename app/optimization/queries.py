from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import cast
from uuid import UUID

from sqlalchemy import and_, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.optimization.memory import (
    COMPATIBLE_PROFILE_PHASES,
    StrategyBenchmarkEvaluationRecord,
    StrategyCandidateEvaluationRecord,
    StrategyCandidateRecord,
    StrategyRefinementRunRecord,
)
from app.optimization.schemas import CandidateQuery, RefinementRunQuery
from app.optimization.simulation import SIMULATION_ENGINE_VERSION

CANDIDATE_SCAN_BATCH_SIZE = 200


def _run_filters(query: RefinementRunQuery) -> list[ColumnElement[bool]]:
    values: list[ColumnElement[bool]] = []
    for name in ("asset", "provider", "market_type", "interval", "status"):
        value = getattr(query, name)
        if value is not None:
            values.append(getattr(StrategyRefinementRunRecord, name) == value.value)
    if query.recommendation_status is not None:
        values.append(
            StrategyRefinementRunRecord.recommendation_status
            == query.recommendation_status
        )
    if query.started_from is not None:
        values.append(StrategyRefinementRunRecord.started_at >= query.started_from)
    if query.started_to is not None:
        values.append(StrategyRefinementRunRecord.started_at <= query.started_to)
    return values


async def list_refinement_runs(
    session: AsyncSession, query: RefinementRunQuery
) -> tuple[StrategyRefinementRunRecord, ...]:
    statement = (
        select(StrategyRefinementRunRecord)
        .where(*_run_filters(query))
        .order_by(
            StrategyRefinementRunRecord.started_at.desc(),
            StrategyRefinementRunRecord.id.desc(),
        )
        .limit(query.limit)
        .offset(query.offset)
    )
    return tuple((await session.scalars(statement)).all())


async def get_refinement_run(
    session: AsyncSession, run_id: UUID
) -> StrategyRefinementRunRecord | None:
    return await session.get(StrategyRefinementRunRecord, run_id)


async def get_candidate(
    session: AsyncSession, candidate_id: str
) -> StrategyCandidateRecord | None:
    return await session.get(StrategyCandidateRecord, candidate_id)


def _candidate_evidence_filters(
    query: CandidateQuery,
) -> list[ColumnElement[bool]]:
    values: list[ColumnElement[bool]] = [
        StrategyRefinementRunRecord.status == "completed"
    ]
    for name in ("asset", "provider", "market_type", "interval"):
        value = getattr(query, name)
        if value is not None:
            values.append(getattr(StrategyRefinementRunRecord, name) == value.value)
    return values


async def list_candidate_definition_batch(
    session: AsyncSession,
    query: CandidateQuery,
    *,
    after_candidate_id: str | None,
    batch_size: int = CANDIDATE_SCAN_BATCH_SIZE,
) -> tuple[StrategyCandidateRecord, ...]:
    candidate_ids = (
        select(StrategyCandidateEvaluationRecord.stable_candidate_id)
        .join(
            StrategyRefinementRunRecord,
            StrategyRefinementRunRecord.id
            == StrategyCandidateEvaluationRecord.refinement_run_id,
        )
        .where(*_candidate_evidence_filters(query))
        .distinct()
    )
    filters: list[ColumnElement[bool]] = [
        StrategyCandidateRecord.stable_candidate_id.in_(candidate_ids)
    ]
    if after_candidate_id is not None:
        filters.append(StrategyCandidateRecord.stable_candidate_id > after_candidate_id)
    for name in (
        "score_model_name",
        "score_model_version",
        "strategy_name",
        "strategy_version",
    ):
        value = getattr(query, name)
        if value is not None:
            filters.append(getattr(StrategyCandidateRecord, name) == value)
    statement = (
        select(StrategyCandidateRecord)
        .where(*filters)
        .order_by(StrategyCandidateRecord.stable_candidate_id)
        .limit(batch_size)
    )
    return tuple((await session.scalars(statement)).all())


async def load_profile_evidence(
    session: AsyncSession,
    *,
    candidate_ids: tuple[str, ...],
    query: CandidateQuery | None = None,
) -> list[
    tuple[StrategyCandidateEvaluationRecord, StrategyRefinementRunRecord, Decimal]
]:
    if not candidate_ids:
        return []
    baseline = StrategyBenchmarkEvaluationRecord.__table__.alias("profile_baseline")
    filters: list[ColumnElement[bool]] = [
        StrategyCandidateEvaluationRecord.stable_candidate_id.in_(candidate_ids),
        StrategyRefinementRunRecord.status == "completed",
        StrategyRefinementRunRecord.score_model_name
        == StrategyCandidateRecord.score_model_name,
        StrategyRefinementRunRecord.score_model_version
        == StrategyCandidateRecord.score_model_version,
        StrategyRefinementRunRecord.strategy_name
        == StrategyCandidateRecord.strategy_name,
        StrategyRefinementRunRecord.strategy_version
        == StrategyCandidateRecord.strategy_version,
        StrategyRefinementRunRecord.simulation_engine_version
        == SIMULATION_ENGINE_VERSION,
        StrategyCandidateEvaluationRecord.phase.in_(COMPATIBLE_PROFILE_PHASES),
    ]
    if query is not None:
        filters.extend(_candidate_evidence_filters(query)[1:])
    statement = (
        select(
            StrategyCandidateEvaluationRecord,
            StrategyRefinementRunRecord,
            baseline.c.total_return,
        )
        .join(
            StrategyRefinementRunRecord,
            StrategyRefinementRunRecord.id
            == StrategyCandidateEvaluationRecord.refinement_run_id,
        )
        .join(
            StrategyCandidateRecord,
            StrategyCandidateRecord.stable_candidate_id
            == StrategyCandidateEvaluationRecord.stable_candidate_id,
        )
        .join(
            baseline,
            and_(
                baseline.c.refinement_run_id == StrategyRefinementRunRecord.id,
                baseline.c.benchmark == "production_baseline",
                baseline.c.segment == StrategyCandidateEvaluationRecord.segment,
            ),
        )
        .where(*filters)
        .order_by(
            StrategyCandidateEvaluationRecord.evaluated_at,
            StrategyCandidateEvaluationRecord.id,
        )
    )
    rows = (await session.execute(statement)).all()
    return list(
        cast(
            Sequence[
                tuple[
                    StrategyCandidateEvaluationRecord,
                    StrategyRefinementRunRecord,
                    Decimal,
                ]
            ],
            rows,
        )
    )


async def load_run_evaluation_page(
    session: AsyncSession,
    run_id: UUID,
    *,
    after_id: UUID | None,
    limit: int,
) -> tuple[StrategyCandidateEvaluationRecord, ...]:
    filters: list[ColumnElement[bool]] = [
        StrategyCandidateEvaluationRecord.refinement_run_id == run_id
    ]
    if after_id is not None:
        filters.append(StrategyCandidateEvaluationRecord.id > after_id)
    return tuple(
        (
            await session.scalars(
                select(StrategyCandidateEvaluationRecord)
                .where(*filters)
                .order_by(StrategyCandidateEvaluationRecord.id)
                .limit(limit + 1)
            )
        ).all()
    )


async def load_run_evaluation_counts(
    session: AsyncSession, run_id: UUID
) -> tuple[tuple[str, str, int], ...]:
    rows = (
        await session.execute(
            select(
                StrategyCandidateEvaluationRecord.phase,
                StrategyCandidateEvaluationRecord.segment,
                func.count(),
            )
            .where(StrategyCandidateEvaluationRecord.refinement_run_id == run_id)
            .group_by(
                StrategyCandidateEvaluationRecord.phase,
                StrategyCandidateEvaluationRecord.segment,
            )
            .order_by(
                StrategyCandidateEvaluationRecord.phase,
                StrategyCandidateEvaluationRecord.segment,
            )
        )
    ).all()
    return tuple(cast(Sequence[tuple[str, str, int]], rows))


async def load_run_benchmarks(
    session: AsyncSession, run_id: UUID
) -> tuple[StrategyBenchmarkEvaluationRecord, ...]:
    return tuple(
        (
            await session.scalars(
                select(StrategyBenchmarkEvaluationRecord)
                .where(StrategyBenchmarkEvaluationRecord.refinement_run_id == run_id)
                .order_by(
                    StrategyBenchmarkEvaluationRecord.benchmark,
                    StrategyBenchmarkEvaluationRecord.segment,
                )
            )
        ).all()
    )


async def load_candidate_history(
    session: AsyncSession,
    candidate_id: str,
    *,
    limit: int,
    after: tuple[datetime, UUID] | None,
) -> list[
    tuple[StrategyCandidateEvaluationRecord, StrategyRefinementRunRecord, Decimal]
]:
    baseline = StrategyBenchmarkEvaluationRecord.__table__.alias("history_baseline")
    statement = (
        select(
            StrategyCandidateEvaluationRecord,
            StrategyRefinementRunRecord,
            baseline.c.total_return,
        )
        .join(
            StrategyRefinementRunRecord,
            StrategyRefinementRunRecord.id
            == StrategyCandidateEvaluationRecord.refinement_run_id,
        )
        .join(
            baseline,
            and_(
                baseline.c.refinement_run_id == StrategyRefinementRunRecord.id,
                baseline.c.benchmark == "production_baseline",
                baseline.c.segment == StrategyCandidateEvaluationRecord.segment,
            ),
        )
        .where(
            StrategyCandidateEvaluationRecord.stable_candidate_id == candidate_id,
            *(
                (
                    tuple_(
                        StrategyCandidateEvaluationRecord.evaluated_at,
                        StrategyCandidateEvaluationRecord.id,
                    )
                    > after,
                )
                if after is not None
                else ()
            ),
        )
        .order_by(
            StrategyCandidateEvaluationRecord.evaluated_at,
            StrategyCandidateEvaluationRecord.id,
        )
        .limit(limit + 1)
    )
    rows = (await session.execute(statement)).all()
    return list(
        cast(
            Sequence[
                tuple[
                    StrategyCandidateEvaluationRecord,
                    StrategyRefinementRunRecord,
                    Decimal,
                ]
            ],
            rows,
        )
    )
