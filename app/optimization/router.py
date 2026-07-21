from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
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
)

router = APIRouter(prefix="/optimization", tags=["optimization"])
RunQuery = Annotated[RefinementRunQuery, Query()]
CandidatesQuery = Annotated[CandidateQuery, Query()]


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
