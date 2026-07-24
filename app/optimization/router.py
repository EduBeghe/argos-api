from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.composite_search import CompositeCandidateStatus
from app.optimization.composite_search_read import (
    MAX_COMPOSITE_PAGE_LIMIT,
    CompositeSearchNotFoundError,
    get_composite_candidate,
    get_composite_search,
    list_composite_candidates,
)
from app.optimization.composite_search_schemas import (
    CompositeCandidateDetailResponse,
    CompositeCandidatePageResponse,
    CompositeSearchSummaryResponse,
)
from app.optimization.edge_guided import EdgeGuidedCandidateStatus
from app.optimization.edge_guided_read import (
    MAX_EDGE_GUIDED_PAGE_LIMIT,
    EdgeGuidedReadNotFoundError,
    get_edge_guided_candidate,
    get_edge_guided_run,
    list_edge_guided_candidates,
    list_edge_guided_runs,
)
from app.optimization.edge_guided_schemas import (
    EdgeGuidedActionFilter,
    EdgeGuidedCandidateDetailResponse,
    EdgeGuidedCandidatePageResponse,
    EdgeGuidedRunPageResponse,
    EdgeGuidedRunResponse,
    EdgeGuidedRunStatus,
)
from app.optimization.predictive_edge_read import (
    MAX_PAGE_LIMIT,
    PredictiveEdgeNotFoundError,
    get_predictive_edge_diagnostics,
    get_predictive_edge_near_miss,
    get_predictive_edge_run,
    list_predictive_edge_near_misses,
    list_predictive_edge_runs,
    list_predictive_edge_windows,
    list_predictive_edges,
    list_strategy_hypotheses,
)
from app.optimization.predictive_edge_schemas import (
    PredictiveEdgeDiagnosticResponse,
    PredictiveEdgeNearMissDetailResponse,
    PredictiveEdgeNearMissPageResponse,
    PredictiveEdgePageResponse,
    PredictiveEdgeRunDetailResponse,
    PredictiveEdgeRunPageResponse,
    PredictiveEdgeWindowPageResponse,
    StrategyHypothesisPageResponse,
)
from app.optimization.predictive_edges import (
    Condition,
    EvidenceStatus,
)
from app.optimization.read_service import (
    OptimizationReadNotFoundError,
    read_candidate,
    read_candidates,
    read_refinement_run,
    read_refinement_runs,
    read_run_evaluations,
)
from app.optimization.schemas import (
    MAX_OPTIMIZATION_PAGE_SIZE,
    CandidateDetailResponse,
    CandidatePageResponse,
    CandidateQuery,
    EvaluationPageResponse,
    RefinementRunDetailResponse,
    RefinementRunQuery,
    RefinementRunSummaryResponse,
    WalkForwardCandidatePageResponse,
    WalkForwardRunDetailResponse,
    WalkForwardRunResponse,
    WalkForwardSelectionPageResponse,
    WalkForwardWindowEvaluationPageResponse,
)
from app.optimization.targeted_refinement import TargetedCandidateStatus
from app.optimization.targeted_refinement_read import (
    MAX_TARGETED_PAGE_LIMIT,
    TargetedRefinementNotFoundError,
    get_targeted_candidate,
    get_targeted_refinement,
    list_targeted_candidates,
)
from app.optimization.targeted_refinement_schemas import (
    TargetedCandidateDetailResponse,
    TargetedCandidatePageResponse,
    TargetedRefinementSummaryResponse,
)
from app.optimization.walk_forward import WalkForwardClassification
from app.optimization.walk_forward_read import (
    WalkForwardReadNotFoundError,
    read_refinement_walk_forward,
    read_walk_forward_candidate_windows,
    read_walk_forward_candidates,
    read_walk_forward_run,
    read_walk_forward_selections,
)

router = APIRouter(prefix="/optimization", tags=["optimization"])
RunQuery = Annotated[RefinementRunQuery, Query()]
CandidatesQuery = Annotated[CandidateQuery, Query()]


@router.get(
    "/predictive-edge-runs/{run_id}/composite-search",
    response_model=CompositeSearchSummaryResponse,
)
async def composite_search(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> CompositeSearchSummaryResponse:
    try:
        return await get_composite_search(session, run_id)
    except CompositeSearchNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/composite-search/candidates",
    response_model=CompositeCandidatePageResponse,
)
async def composite_search_candidates(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    depth: Annotated[int | None, Query(ge=2, le=7)] = None,
    status: Annotated[CompositeCandidateStatus | None, Query()] = None,
    feature_name: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_COMPOSITE_PAGE_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CompositeCandidatePageResponse:
    try:
        return await list_composite_candidates(
            session,
            run_id,
            depth=depth,
            status=status,
            feature_name=feature_name,
            limit=limit,
            offset=offset,
        )
    except CompositeSearchNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/composite-search/candidates/{candidate_id}",
    response_model=CompositeCandidateDetailResponse,
)
async def composite_search_candidate(
    run_id: UUID,
    candidate_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CompositeCandidateDetailResponse:
    try:
        return await get_composite_candidate(session, run_id, candidate_id)
    except CompositeSearchNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/targeted-refinement",
    response_model=TargetedRefinementSummaryResponse,
)
async def targeted_refinement(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> TargetedRefinementSummaryResponse:
    try:
        return await get_targeted_refinement(session, run_id)
    except TargetedRefinementNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/targeted-refinement/candidates",
    response_model=TargetedCandidatePageResponse,
)
async def targeted_refinement_candidates(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    status: Annotated[TargetedCandidateStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_TARGETED_PAGE_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TargetedCandidatePageResponse:
    try:
        return await list_targeted_candidates(
            session, run_id, status=status, limit=limit, offset=offset
        )
    except TargetedRefinementNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/targeted-refinement/candidates/{candidate_id}",
    response_model=TargetedCandidateDetailResponse,
)
async def targeted_refinement_candidate(
    run_id: UUID,
    candidate_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TargetedCandidateDetailResponse:
    try:
        return await get_targeted_candidate(session, run_id, candidate_id)
    except TargetedRefinementNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/edge-guided/runs", response_model=EdgeGuidedRunPageResponse)
async def edge_guided_runs(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    asset: Annotated[Asset | None, Query()] = None,
    provider: Annotated[Provider | None, Query()] = None,
    market_type: Annotated[MarketType | None, Query()] = None,
    interval: Annotated[Interval | None, Query()] = None,
    status: Annotated[EdgeGuidedRunStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_EDGE_GUIDED_PAGE_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EdgeGuidedRunPageResponse:
    return await list_edge_guided_runs(
        session,
        asset=asset,
        provider=provider,
        market_type=market_type,
        interval=interval,
        status=status.value if status else None,
        limit=limit,
        offset=offset,
    )


@router.get("/edge-guided/runs/{run_id}", response_model=EdgeGuidedRunResponse)
async def edge_guided_run(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> EdgeGuidedRunResponse:
    try:
        return await get_edge_guided_run(session, run_id)
    except EdgeGuidedReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/edge-guided/candidates", response_model=EdgeGuidedCandidatePageResponse)
async def edge_guided_candidates(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    run_id: Annotated[UUID | None, Query()] = None,
    asset: Annotated[Asset | None, Query()] = None,
    provider: Annotated[Provider | None, Query()] = None,
    market_type: Annotated[MarketType | None, Query()] = None,
    interval: Annotated[Interval | None, Query()] = None,
    status: Annotated[EdgeGuidedCandidateStatus | None, Query()] = None,
    action: Annotated[EdgeGuidedActionFilter | None, Query()] = None,
    predictor_name: Annotated[str | None, Query()] = None,
    condition: Annotated[Condition | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_EDGE_GUIDED_PAGE_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EdgeGuidedCandidatePageResponse:
    return await list_edge_guided_candidates(
        session,
        run_id=run_id,
        asset=asset,
        provider=provider,
        market_type=market_type,
        interval=interval,
        status=status,
        action=action,
        predictor_name=predictor_name,
        condition=condition,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/edge-guided/candidates/{candidate_evaluation_id}",
    response_model=EdgeGuidedCandidateDetailResponse,
)
async def edge_guided_candidate(
    candidate_evaluation_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> EdgeGuidedCandidateDetailResponse:
    try:
        return await get_edge_guided_candidate(session, candidate_evaluation_id)
    except EdgeGuidedReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/predictive-edge-runs", response_model=PredictiveEdgeRunPageResponse)
async def predictive_edge_runs(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    asset: Annotated[str | None, Query()] = None,
    provider: Annotated[str | None, Query()] = None,
    market_type: Annotated[str | None, Query()] = None,
    interval: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = 50,
    kind: Literal["actionable", "diagnostic"] = "actionable",
) -> PredictiveEdgeRunPageResponse:
    return await list_predictive_edge_runs(
        session,
        asset=asset,
        provider=provider,
        market_type=market_type,
        interval=interval,
        status=status,
        cursor=cursor,
        limit=limit,
    )


@router.get(
    "/predictive-edge-runs/{run_id}", response_model=PredictiveEdgeRunDetailResponse
)
async def predictive_edge_run(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> PredictiveEdgeRunDetailResponse:
    try:
        return await get_predictive_edge_run(session, run_id)
    except PredictiveEdgeNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/edges", response_model=PredictiveEdgePageResponse
)
async def predictive_edges(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    predictor: Annotated[str | None, Query()] = None,
    horizon_seconds: Annotated[int | None, Query(gt=0)] = None,
    condition: Annotated[str | None, Query()] = None,
    classification: Annotated[EvidenceStatus | None, Query()] = None,
    cursor: Annotated[UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = 50,
) -> PredictiveEdgePageResponse:
    try:
        return await list_predictive_edges(
            session,
            run_id=run_id,
            predictor=predictor,
            horizon_seconds=horizon_seconds,
            condition=condition,
            classification=classification,
            cursor=cursor,
            limit=limit,
        )
    except PredictiveEdgeNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/edges/{edge_id}/windows",
    response_model=PredictiveEdgeWindowPageResponse,
)
async def predictive_edge_windows(
    run_id: UUID,
    edge_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    cursor: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = 50,
) -> PredictiveEdgeWindowPageResponse:
    try:
        return await list_predictive_edge_windows(
            session, run_id=run_id, edge_id=edge_id, cursor=cursor, limit=limit
        )
    except PredictiveEdgeNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/hypotheses",
    response_model=StrategyHypothesisPageResponse,
)
async def predictive_edge_hypotheses(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = 50,
) -> StrategyHypothesisPageResponse:
    try:
        return await list_strategy_hypotheses(
            session, run_id=run_id, cursor=cursor, limit=limit
        )
    except PredictiveEdgeNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/diagnostics",
    response_model=PredictiveEdgeDiagnosticResponse,
)
async def predictive_edge_diagnostics(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> PredictiveEdgeDiagnosticResponse:
    try:
        return await get_predictive_edge_diagnostics(session, run_id)
    except PredictiveEdgeNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/near-misses",
    response_model=PredictiveEdgeNearMissPageResponse,
)
async def predictive_edge_near_misses(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    cursor: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = 50,
    kind: Literal["actionable", "diagnostic"] = "actionable",
) -> PredictiveEdgeNearMissPageResponse:
    try:
        return await list_predictive_edge_near_misses(
            session, run_id=run_id, cursor=cursor, limit=limit, kind=kind
        )
    except PredictiveEdgeNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/near-misses/{near_miss_id}",
    response_model=PredictiveEdgeNearMissDetailResponse,
)
async def predictive_edge_near_miss(
    run_id: UUID,
    near_miss_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> PredictiveEdgeNearMissDetailResponse:
    try:
        return await get_predictive_edge_near_miss(
            session, run_id=run_id, near_miss_id=near_miss_id
        )
    except PredictiveEdgeNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/refinement-runs", response_model=list[RefinementRunSummaryResponse])
async def refinement_runs(
    query: RunQuery, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> tuple[RefinementRunSummaryResponse, ...]:
    return await read_refinement_runs(session, query)


@router.get("/refinement-runs/{run_id}", response_model=RefinementRunDetailResponse)
async def refinement_run(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> RefinementRunDetailResponse:
    try:
        return await read_refinement_run(session, run_id)
    except OptimizationReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/refinement-runs/{run_id}/evaluations", response_model=EvaluationPageResponse
)
async def refinement_run_evaluations(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=MAX_OPTIMIZATION_PAGE_SIZE)] = 100,
    cursor: Annotated[str | None, Query()] = None,
) -> EvaluationPageResponse:
    try:
        return await read_run_evaluations(session, run_id, limit=limit, cursor=cursor)
    except OptimizationReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/candidates", response_model=CandidatePageResponse)
async def candidates(
    query: CandidatesQuery, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> CandidatePageResponse:
    return await read_candidates(session, query)


@router.get("/candidates/{stable_candidate_id}", response_model=CandidateDetailResponse)
async def candidate(
    stable_candidate_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=MAX_OPTIMIZATION_PAGE_SIZE)] = 100,
    cursor: Annotated[str | None, Query()] = None,
) -> CandidateDetailResponse:
    try:
        return await read_candidate(
            session, stable_candidate_id, limit=limit, cursor=cursor
        )
    except OptimizationReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get(
    "/refinement-runs/{run_id}/walk-forward",
    response_model=WalkForwardRunResponse,
)
async def refinement_walk_forward(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> WalkForwardRunResponse:
    try:
        return await read_refinement_walk_forward(session, run_id)
    except WalkForwardReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/walk-forward-runs/{run_id}", response_model=WalkForwardRunDetailResponse)
async def walk_forward_run(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> WalkForwardRunDetailResponse:
    try:
        return await read_walk_forward_run(session, run_id)
    except WalkForwardReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/walk-forward-runs/{run_id}/candidates",
    response_model=WalkForwardCandidatePageResponse,
)
async def walk_forward_candidates(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    classification: Annotated[WalkForwardClassification | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_OPTIMIZATION_PAGE_SIZE)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> WalkForwardCandidatePageResponse:
    try:
        return await read_walk_forward_candidates(
            session,
            run_id,
            classification=classification,
            limit=limit,
            cursor=cursor,
        )
    except WalkForwardReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get(
    "/walk-forward-runs/{run_id}/selections",
    response_model=WalkForwardSelectionPageResponse,
)
async def walk_forward_selections(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    selected_for_test: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_OPTIMIZATION_PAGE_SIZE)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> WalkForwardSelectionPageResponse:
    try:
        return await read_walk_forward_selections(
            session,
            run_id,
            selected_for_test=selected_for_test,
            limit=limit,
            cursor=cursor,
        )
    except WalkForwardReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get(
    "/walk-forward-runs/{run_id}/candidates/{candidate_id}/windows",
    response_model=WalkForwardWindowEvaluationPageResponse,
)
async def walk_forward_candidate_windows(
    run_id: UUID,
    candidate_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=MAX_OPTIMIZATION_PAGE_SIZE)] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> WalkForwardWindowEvaluationPageResponse:
    try:
        return await read_walk_forward_candidate_windows(
            session,
            run_id,
            candidate_id,
            limit=limit,
            cursor=cursor,
        )
    except WalkForwardReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
