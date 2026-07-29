from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.refinement.contract_service import refinement_contract
from app.ml.refinement.contracts import (
    MLRefinementRequest,
    PredictionObjective,
    RefinementClassification,
    RefinementStatus,
)
from app.ml.refinement.persistence import MLRefinementError
from app.ml.refinement.queue import (
    QueueDisposition,
    RefinementQueueConflictError,
    RefinementQueueNotFoundError,
    RefinementQueueValidationError,
    cancel_refinement,
    enqueue_refinement,
    retry_refinement,
)
from app.ml.refinement.read_service import (
    MAX_REFINEMENT_PAGE_LIMIT,
    RefinementReadNotFoundError,
)
from app.ml.refinement.read_service import (
    get_evaluation as get_ml_refinement_evaluation,
)
from app.ml.refinement.read_service import (
    get_run_detail as get_ml_refinement_run,
)
from app.ml.refinement.read_service import (
    list_evaluations as list_ml_refinement_evaluations,
)
from app.ml.refinement.read_service import (
    list_runs as list_ml_refinement_runs,
)
from app.ml.refinement.schemas import (
    RefinementContractResponse,
    RefinementEnqueueResponse,
    RefinementEvaluationDetail,
    RefinementEvaluationPage,
    RefinementRunDetail,
    RefinementRunPage,
    RefinementRunResponse,
)
from app.ml.screening.config import ScreeningEngine
from app.ml.screening.evidence import EvidenceClassification
from app.ml.screening.read_service import (
    MAX_SCREENING_PAGE_LIMIT,
    ScreeningReadNotFoundError,
    get_screening_result_detail,
    get_screening_run_detail,
    latest_completed_screening_run,
    list_screening_results,
    list_screening_runs,
)
from app.ml.screening.schemas import (
    ScreeningResultDetail,
    ScreeningResultPage,
    ScreeningRunDetail,
    ScreeningRunPage,
    ScreeningRunSummary,
)
from app.optimization.composite_search import (
    CompositeCandidateStatus,
    CompositeSelectionTrack,
)
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


@router.get("/ml-refinement-contract", response_model=RefinementContractResponse)
async def ml_refinement_contract() -> RefinementContractResponse:
    return refinement_contract()


@router.post(
    "/ml-refinement-runs",
    response_model=RefinementEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_ml_refinement_run(
    request: MLRefinementRequest,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RefinementEnqueueResponse:
    try:
        result = await enqueue_refinement(session, request)
    except MLRefinementError as error:
        detail = str(error)
        code = 404 if "does not exist" in detail else 409
        raise HTTPException(status_code=code, detail=detail) from error
    except RefinementQueueValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RefinementQueueConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if result.disposition is QueueDisposition.REUSED_COMPLETED:
        response.status_code = status.HTTP_200_OK
    location = f"/optimization/ml-refinement-runs/{result.run.id}"
    response.headers["Location"] = location
    return RefinementEnqueueResponse(
        disposition=result.disposition,
        run=RefinementRunResponse.model_validate(result.run),
        location=location,
    )


@router.post(
    "/ml-refinement-runs/{run_id}/cancel", response_model=RefinementRunResponse
)
async def cancel_ml_refinement_run(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> RefinementRunResponse:
    try:
        record = await cancel_refinement(session, run_id)
    except RefinementQueueNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return RefinementRunResponse.model_validate(record)


@router.post(
    "/ml-refinement-runs/{run_id}/retry",
    response_model=RefinementEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_ml_refinement_run(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> RefinementEnqueueResponse:
    try:
        result = await retry_refinement(session, run_id)
    except RefinementQueueNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RefinementQueueConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    location = f"/optimization/ml-refinement-runs/{run_id}"
    return RefinementEnqueueResponse(
        disposition=result.disposition,
        run=RefinementRunResponse.model_validate(result.run),
        location=location,
    )


@router.get("/ml-refinement-runs", response_model=RefinementRunPage)
async def ml_refinement_runs(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    status: Annotated[RefinementStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_REFINEMENT_PAGE_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RefinementRunPage:
    return await list_ml_refinement_runs(
        session, status=status, limit=limit, offset=offset
    )


@router.get("/ml-refinement-runs/{run_id}", response_model=RefinementRunDetail)
async def ml_refinement_run(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> RefinementRunDetail:
    try:
        return await get_ml_refinement_run(session, run_id)
    except RefinementReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/ml-refinement-runs/{run_id}/evaluations",
    response_model=RefinementEvaluationPage,
)
async def ml_refinement_evaluations(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    engine: Annotated[ScreeningEngine | None, Query()] = None,
    objective: Annotated[PredictionObjective | None, Query()] = None,
    classification: Annotated[RefinementClassification | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_REFINEMENT_PAGE_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RefinementEvaluationPage:
    try:
        return await list_ml_refinement_evaluations(
            session,
            run_id,
            engine=engine,
            objective=objective,
            classification=classification,
            limit=limit,
            offset=offset,
        )
    except RefinementReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/ml-refinement-runs/{run_id}/evaluations/{evaluation_id}",
    response_model=RefinementEvaluationDetail,
)
async def ml_refinement_evaluation(
    run_id: UUID,
    evaluation_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RefinementEvaluationDetail:
    try:
        return await get_ml_refinement_evaluation(session, run_id, evaluation_id)
    except RefinementReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/ml-screening-runs", response_model=ScreeningRunPage)
async def ml_screening_runs(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    asset: Annotated[Asset | None, Query()] = None,
    provider: Annotated[Provider | None, Query()] = None,
    market_type: Annotated[MarketType | None, Query()] = None,
    interval: Annotated[Interval | None, Query()] = None,
    status: Annotated[Literal["running", "completed", "failed"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_SCREENING_PAGE_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ScreeningRunPage:
    return await list_screening_runs(
        session,
        asset=asset,
        provider=provider,
        market_type=market_type,
        interval=interval,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get("/ml-screening-runs/latest", response_model=ScreeningRunSummary)
async def latest_ml_screening_run(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    asset: Annotated[Asset, Query()],
    provider: Annotated[Provider, Query()],
    market_type: Annotated[MarketType, Query()],
    interval: Annotated[Interval, Query()],
) -> ScreeningRunSummary:
    try:
        return await latest_completed_screening_run(
            session,
            asset=asset,
            provider=provider,
            market_type=market_type,
            interval=interval,
        )
    except ScreeningReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/ml-screening-runs/{run_id}", response_model=ScreeningRunDetail)
async def ml_screening_run(
    run_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> ScreeningRunDetail:
    try:
        return await get_screening_run_detail(session, run_id)
    except ScreeningReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/ml-screening-runs/{run_id}/results", response_model=ScreeningResultPage)
async def ml_screening_results(
    run_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    engine: Annotated[ScreeningEngine | None, Query()] = None,
    pareto_retained: Annotated[bool | None, Query()] = None,
    evidence_classification: Annotated[EvidenceClassification | None, Query()] = None,
    minimum_future_duration_seconds: Annotated[int | None, Query(ge=1)] = None,
    maximum_future_duration_seconds: Annotated[int | None, Query(ge=1)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_SCREENING_PAGE_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ScreeningResultPage:
    if (
        minimum_future_duration_seconds is not None
        and maximum_future_duration_seconds is not None
        and maximum_future_duration_seconds < minimum_future_duration_seconds
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "maximum future duration must be greater than or equal to "
                "minimum future duration"
            ),
        )
    try:
        return await list_screening_results(
            session,
            run_id,
            engine=engine,
            pareto_retained=pareto_retained,
            evidence_classification=evidence_classification,
            minimum_future_duration_seconds=minimum_future_duration_seconds,
            maximum_future_duration_seconds=maximum_future_duration_seconds,
            limit=limit,
            offset=offset,
        )
    except ScreeningReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/ml-screening-runs/{run_id}/results/{result_id}",
    response_model=ScreeningResultDetail,
)
async def ml_screening_result(
    run_id: UUID,
    result_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ScreeningResultDetail:
    try:
        return await get_screening_result_detail(session, run_id, result_id)
    except ScreeningReadNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/predictive-edge-runs/{run_id}/composite-search",
    response_model=CompositeSearchSummaryResponse,
)
async def composite_search(
    run_id: UUID, sessi×O4¶‰žËkºwµçQ…Ñ•}¥‘ôˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õQ…É•Ñ•‘…¹‘¥‘…Ñ••Ñ…¥±I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜Ñ…É•Ñ•‘}É•™¥¹•µ•¹Ñ}…¹‘¥‘…Ñ” (€€€ÉÕ¹}¥èUU%°(€€€…¹‘¥‘…Ñ•}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(¤€´øQ…É•Ñ•‘…¹‘¥‘…Ñ••Ñ…¥±I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð•Ñ}Ñ…É•Ñ•‘}…¹‘¥‘…Ñ”¡Í•ÍÍ¥½¸°ÉÕ¹}¥°…¹‘¥‘…Ñ•}¥¤(€€€•á•ÁÐQ…É•Ñ•‘I•™¥¹•µ•¹Ñ9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð ˆ½•‘”µÕ¥‘•½ÉÕ¹Ìˆ°É•ÍÁ½¹Í•}µ½‘•°õ‘•Õ¥‘•‘IÕ¹A…•I•ÍÁ½¹Í”¤)…Íå¹Œ‘•˜•‘•}Õ¥‘•‘}ÉÕ¹Ì (€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€…ÍÍ•Ðè¹¹½Ñ…Ñ•‘mÍÍ•Ðð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÁÉ½Ù¥‘•Èè¹¹½Ñ…Ñ•‘mAÉ½Ù¥‘•Èð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€µ…É­•Ñ}ÑåÁ”è¹¹½Ñ…Ñ•‘m5…É­•ÑQåÁ”ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€¥¹Ñ•ÉÙ…°è¹¹½Ñ…Ñ•‘m%¹Ñ•ÉÙ…°ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÍÑ…ÑÕÌè¹¹½Ñ…Ñ•‘m‘•Õ¥‘•‘IÕ¹MÑ…ÑÕÌð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}}U%}A}1%5%P¥t€ô€ÔÀ°(€€€½™™Í•Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÀ¥t€ô€À°(¤€´ø‘•Õ¥‘•‘IÕ¹A…•I•ÍÁ½¹Í”è(€€€É•ÑÕÉ¸…Ý…¥Ð±¥ÍÑ}•‘•}Õ¥‘•‘}ÉÕ¹Ì (€€€€€€€Í•ÍÍ¥½¸°(€€€€€€€…ÍÍ•Ðõ…ÍÍ•Ð°(€€€€€€€ÁÉ½Ù¥‘•ÈõÁÉ½Ù¥‘•È°(€€€€€€€µ…É­•Ñ}ÑåÁ”õµ…É­•Ñ}ÑåÁ”°(€€€€€€€¥¹Ñ•ÉÙ…°õ¥¹Ñ•ÉÙ…°°(€€€€€€€ÍÑ…ÑÕÌõÍÑ…ÑÕÌ¹Ù…±Õ”¥˜ÍÑ…ÑÕÌ•±Í”9½¹”°(€€€€€€€±¥µ¥Ðõ±¥µ¥Ð°(€€€€€€€½™™Í•Ðõ½™™Í•Ð°(€€€€¤(()É½ÕÑ•È¹•Ð ˆ½•‘”µÕ¥‘•½ÉÕ¹Ì½íÉÕ¹}¥‘ôˆ°É•ÍÁ½¹Í•}µ½‘•°õ‘•Õ¥‘•‘IÕ¹I•ÍÁ½¹Í”¤)…Íå¹Œ‘•˜•‘•}Õ¥‘•‘}ÉÕ¸ (€€€ÉÕ¹}¥èUU%°Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t(¤€´ø‘•Õ¥‘•‘IÕ¹I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð•Ñ}•‘•}Õ¥‘•‘}ÉÕ¸¡Í•ÍÍ¥½¸°ÉÕ¹}¥¤(€€€•á•ÁÐ‘•Õ¥‘•‘I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð ˆ½•‘”µÕ¥‘•½…¹‘¥‘…Ñ•Ìˆ°É•ÍÁ½¹Í•}µ½‘•°õ‘•Õ¥‘•‘…¹‘¥‘…Ñ•A…•I•ÍÁ½¹Í”¤)…Íå¹Œ‘•˜•‘•}Õ¥‘•‘}…¹‘¥‘…Ñ•Ì (€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€ÉÕ¹}¥è¹¹½Ñ…Ñ•‘mUU%ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€…ÍÍ•Ðè¹¹½Ñ…Ñ•‘mÍÍ•Ðð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÁÉ½Ù¥‘•Èè¹¹½Ñ…Ñ•‘mAÉ½Ù¥‘•Èð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€µ…É­•Ñ}ÑåÁ”è¹¹½Ñ…Ñ•‘m5…É­•ÑQåÁ”ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€¥¹Ñ•ÉÙ…°è¹¹½Ñ…Ñ•‘m%¹Ñ•ÉÙ…°ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÍÑ…ÑÕÌè¹¹½Ñ…Ñ•‘m‘•Õ¥‘•‘…¹‘¥‘…Ñ•MÑ…ÑÕÌð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€…Ñ¥½¸è¹¹½Ñ…Ñ•‘m‘•Õ¥‘•‘Ñ¥½¹¥±Ñ•Èð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÁÉ•‘¥Ñ½É}¹…µ”è¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€½¹‘¥Ñ¥½¸è¹¹½Ñ…Ñ•‘m½¹‘¥Ñ¥½¸ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}}U%}A}1%5%P¥t€ô€ÔÀ°(€€€½™™Í•Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÀ¥t€ô€À°(¤€´ø‘•Õ¥‘•‘…¹‘¥‘…Ñ•A…•I•ÍÁ½¹Í”è(€€€É•ÑÕÉ¸…Ý…¥Ð±¥ÍÑ}•‘•}Õ¥‘•‘}…¹‘¥‘…Ñ•Ì (€€€€€€€Í•ÍÍ¥½¸°(€€€€€€€ÉÕ¹}¥õÉÕ¹}¥°(€€€€€€€…ÍÍ•Ðõ…ÍÍ•Ð°(€€€€€€€ÁÉ½Ù¥‘•ÈõÁÉ½Ù¥‘•È°(€€€€€€€µ…É­•Ñ}ÑåÁ”õµ…É­•Ñ}ÑåÁ”°(€€€€€€€¥¹Ñ•ÉÙ…°õ¥¹Ñ•ÉÙ…°°(€€€€€€€ÍÑ…ÑÕÌõÍÑ…ÑÕÌ°(€€€€€€€…Ñ¥½¸õ…Ñ¥½¸°(€€€€€€€ÁÉ•‘¥Ñ½É}¹…µ”õÁÉ•‘¥Ñ½É}¹…µ”°(€€€€€€€½¹‘¥Ñ¥½¸õ½¹‘¥Ñ¥½¸°(€€€€€€€±¥µ¥Ðõ±¥µ¥Ð°(€€€€€€€½™™Í•Ðõ½™™Í•Ð°(€€€€¤(()É½ÕÑ•È¹•Ð (€€€€ˆ½•‘”µÕ¥‘•½…¹‘¥‘…Ñ•Ì½í…¹‘¥‘…Ñ•}•Ù…±Õ…Ñ¥½¹}¥‘ôˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õ‘•Õ¥‘•‘…¹‘¥‘…Ñ••Ñ…¥±I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜•‘•}Õ¥‘•‘}…¹‘¥‘…Ñ” (€€€…¹‘¥‘…Ñ•}•Ù…±Õ…Ñ¥½¹}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(¤€´ø‘•Õ¥‘•‘…¹‘¥‘…Ñ••Ñ…¥±I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð•Ñ}•‘•}Õ¥‘•‘}…¹‘¥‘…Ñ”¡Í•ÍÍ¥½¸°…¹‘¥‘…Ñ•}•Ù…±Õ…Ñ¥½¹}¥¤(€€€•á•ÁÐ‘•Õ¥‘•‘I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð ˆ½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ìˆ°É•ÍÁ½¹Í•}µ½‘•°õAÉ•‘¥Ñ¥Ù•‘•IÕ¹A…•I•ÍÁ½¹Í”¤)…Íå¹Œ‘•˜ÁÉ•‘¥Ñ¥Ù•}•‘•}ÉÕ¹Ì (€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€…ÍÍ•Ðè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÁÉ½Ù¥‘•Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€µ…É­•Ñ}ÑåÁ”è¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€¥¹Ñ•ÉÙ…°è¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÍÑ…ÑÕÌè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}A}1%5%P¥t€ô€ÔÀ°(€€€­¥¹è1¥Ñ•É…±l‰…Ñ¥½¹…‰±”ˆ°€‰‘¥…¹½ÍÑ¥Œ‰t€ô€‰…Ñ¥½¹…‰±”ˆ°(¤€´øAÉ•‘¥Ñ¥Ù•‘•IÕ¹A…•I•ÍÁ½¹Í”è(€€€É•ÑÕÉ¸…Ý…¥Ð±¥ÍÑ}ÁÉ•‘¥Ñ¥Ù•}•‘•}ÉÕ¹Ì (€€€€€€€Í•ÍÍ¥½¸°(€€€€€€€…ÍÍ•Ðõ…ÍÍ•Ð°(€€€€€€€ÁÉ½Ù¥‘•ÈõÁÉ½Ù¥‘•È°(€€€€€€€µ…É­•Ñ}ÑåÁ”õµ…É­•Ñ}ÑåÁ”°(€€€€€€€¥¹Ñ•ÉÙ…°õ¥¹Ñ•ÉÙ…°°(€€€€€€€ÍÑ…ÑÕÌõÍÑ…ÑÕÌ°(€€€€€€€ÕÉÍ½ÈõÕÉÍ½È°(€€€€€€€±¥µ¥Ðõ±¥µ¥Ð°(€€€€¤(()É½ÕÑ•È¹•Ð (€€€€ˆ½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ôˆ°É•ÍÁ½¹Í•}µ½‘•°õAÉ•‘¥Ñ¥Ù•‘•IÕ¹•Ñ…¥±I•ÍÁ½¹Í”(¤)…Íå¹Œ‘•˜ÁÉ•‘¥Ñ¥Ù•}•‘•}ÉÕ¸ (€€€ÉÕ¹}¥èUU%°Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t(¤€´øAÉ•‘¥Ñ¥Ù•‘•IÕ¹•Ñ…¥±I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð•Ñ}ÁÉ•‘¥Ñ¥Ù•}•‘•}ÉÕ¸¡Í•ÍÍ¥½¸°ÉÕ¹}¥¤(€€€•á•ÁÐAÉ•‘¥Ñ¥Ù•‘•9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½•‘•Ìˆ°É•ÍÁ½¹Í•}µ½‘•°õAÉ•‘¥Ñ¥Ù•‘•A…•I•ÍÁ½¹Í”(¤)…Íå¹Œ‘•˜ÁÉ•‘¥Ñ¥Ù•}•‘•Ì (€€€ÉÕ¹}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€ÁÉ•‘¥Ñ½Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€¡½É¥é½¹}Í•½¹‘Ìè¹¹½Ñ…Ñ•‘m¥¹Ðð9½¹”°EÕ•Éä¡ÐôÀ¥t€ô9½¹”°(€€€½¹‘¥Ñ¥½¸è¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€±…ÍÍ¥™¥…Ñ¥½¸è¹¹½Ñ…Ñ•‘mÙ¥‘•¹•MÑ…ÑÕÌð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘mUU%ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}A}1%5%P¥t€ô€ÔÀ°(¤€´øAÉ•‘¥Ñ¥Ù•‘•A…•I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð±¥ÍÑ}ÁÉ•‘¥Ñ¥Ù•}•‘•Ì (€€€€€€€€€€€Í•ÍÍ¥½¸°(€€€€€€€€€€€ÉÕ¹}¥õÉÕ¹}¥°(€€€€€€€€€€€ÁÉ•‘¥Ñ½ÈõÁÉ•‘¥Ñ½È°(€€€€€€€€€€€¡½É¥é½¹}Í•½¹‘Ìõ¡½É¥é½¹}Í•½¹‘Ì°(€€€€€€€€€€€½¹‘¥Ñ¥½¸õ½¹‘¥Ñ¥½¸°(€€€€€€€€€€€±…ÍÍ¥™¥…Ñ¥½¸õ±…ÍÍ¥™¥…Ñ¥½¸°(€€€€€€€€€€€ÕÉÍ½ÈõÕÉÍ½È°(€€€€€€€€€€€±¥µ¥Ðõ±¥µ¥Ð°(€€€€€€€€¤(€€€•á•ÁÐAÉ•‘¥Ñ¥Ù•‘•9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½•‘•Ì½í•‘•}¥‘ô½Ý¥¹‘½ÝÌˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õAÉ•‘¥Ñ¥Ù•‘•]¥¹‘½ÝA…•I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜ÁÉ•‘¥Ñ¥Ù•}•‘•}Ý¥¹‘½ÝÌ (€€€ÉÕ¹}¥èUU%°(€€€•‘•}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘m¥¹Ðð9½¹”°EÕ•Éä¡”ôÀ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}A}1%5%P¥t€ô€ÔÀ°(¤€´øAÉ•‘¥Ñ¥Ù•‘•]¥¹‘½ÝA…•I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð±¥ÍÑ}ÁÉ•‘¥Ñ¥Ù•}•‘•}Ý¥¹‘½ÝÌ (€€€€€€€€€€€Í•ÍÍ¥½¸°ÉÕ¹}¥õÉÕ¹}¥°•‘•}¥õ•‘•}¥°ÕÉÍ½ÈõÕÉÍ½È°±¥µ¥Ðõ±¥µ¥Ð(€€€€€€€€¤(€€€•á•ÁÐAÉ•‘¥Ñ¥Ù•‘•9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½¡åÁ½Ñ¡•Í•Ìˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õMÑÉ…Ñ•å!åÁ½Ñ¡•Í¥ÍA…•I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜ÁÉ•‘¥Ñ¥Ù•}•‘•}¡åÁ½Ñ¡•Í•Ì (€€€ÉÕ¹}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}A}1%5%P¥t€ô€ÔÀ°(¤€´øMÑÉ…Ñ•å!åÁ½Ñ¡•Í¥ÍA…•I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð±¥ÍÑ}ÍÑÉ…Ñ•å}¡åÁ½Ñ¡•Í•Ì (€€€€€€€€€€€Í•ÍÍ¥½¸°ÉÕ¹}¥õÉÕ¹}¥°ÕÉÍ½ÈõÕÉÍ½È°±¥µ¥Ðõ±¥µ¥Ð(€€€€€€€€¤(€€€•á•ÁÐAÉ•‘¥Ñ¥Ù•‘•9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½‘¥…¹½ÍÑ¥Ìˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õAÉ•‘¥Ñ¥Ù•‘•¥…¹½ÍÑ¥I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜ÁÉ•‘¥Ñ¥Ù•}•‘•}‘¥…¹½ÍÑ¥Ì (€€€ÉÕ¹}¥èUU%°Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t(¤€´øAÉ•‘¥Ñ¥Ù•‘•¥…¹½ÍÑ¥I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð•Ñ}ÁÉ•‘¥Ñ¥Ù•}•‘•}‘¥…¹½ÍÑ¥Ì¡Í•ÍÍ¥½¸°ÉÕ¹}¥¤(€€€•á•ÁÐAÉ•‘¥Ñ¥Ù•‘•9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½¹•…Èµµ¥ÍÍ•Ìˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õAÉ•‘¥Ñ¥Ù•‘•9•…É5¥ÍÍA…•I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜ÁÉ•‘¥Ñ¥Ù•}•‘•}¹•…É}µ¥ÍÍ•Ì (€€€ÉÕ¹}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘m¥¹Ðð9½¹”°EÕ•Éä¡”ôÀ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}A}1%5%P¥t€ô€ÔÀ°(€€€­¥¹è1¥Ñ•É…±l‰…Ñ¥½¹…‰±”ˆ°€‰‘¥…¹½ÍÑ¥Œ‰t€ô€‰…Ñ¥½¹…‰±”ˆ°(¤€´øAÉ•‘¥Ñ¥Ù•‘•9•…É5¥ÍÍA…•I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð±¥ÍÑ}ÁÉ•‘¥Ñ¥Ù•}•‘•}¹•…É}µ¥ÍÍ•Ì (€€€€€€€€€€€Í•ÍÍ¥½¸°ÉÕ¹}¥õÉÕ¹}¥°ÕÉÍ½ÈõÕÉÍ½È°±¥µ¥Ðõ±¥µ¥Ð°­¥¹õ­¥¹(€€€€€€€€¤(€€€•á•ÁÐAÉ•‘¥Ñ¥Ù•‘•9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½ÁÉ•‘¥Ñ¥Ù”µ•‘”µÉÕ¹Ì½íÉÕ¹}¥‘ô½¹•…Èµµ¥ÍÍ•Ì½í¹•…É}µ¥ÍÍ}¥‘ôˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õAÉ•‘¥Ñ¥Ù•‘•9•…É5¥ÍÍ•Ñ…¥±I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜ÁÉ•‘¥Ñ¥Ù•}•‘•}¹•…É}µ¥ÍÌ (€€€ÉÕ¹}¥èUU%°(€€€¹•…É}µ¥ÍÍ}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(¤€´øAÉ•‘¥Ñ¥Ù•‘•9•…É5¥ÍÍ•Ñ…¥±I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥Ð•Ñ}ÁÉ•‘¥Ñ¥Ù•}•‘•}¹•…É}µ¥ÍÌ (€€€€€€€€€€€Í•ÍÍ¥½¸°ÉÕ¹}¥õÉÕ¹}¥°¹•…É}µ¥ÍÍ}¥õ¹•…É}µ¥ÍÍ}¥(€€€€€€€€¤(€€€•á•ÁÐAÉ•‘¥Ñ¥Ù•‘•9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð ˆ½É•™¥¹•µ•¹ÐµÉÕ¹Ìˆ°É•ÍÁ½¹Í•}µ½‘•°õ±¥ÍÑmI•™¥¹•µ•¹ÑIÕ¹MÕµµ…ÉåI•ÍÁ½¹Í•t¤)…Íå¹Œ‘•˜É•™¥¹•µ•¹Ñ}ÉÕ¹Ì (€€€ÅÕ•ÉäèIÕ¹EÕ•Éä°Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t(¤€´øÑÕÁ±•mI•™¥¹•µ•¹ÑIÕ¹MÕµµ…ÉåI•ÍÁ½¹Í”°€¸¸¹tè(€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}É•™¥¹•µ•¹Ñ}ÉÕ¹Ì¡Í•ÍÍ¥½¸°ÅÕ•Éä¤(()É½ÕÑ•È¹•Ð ˆ½É•™¥¹•µ•¹ÐµÉÕ¹Ì½íÉÕ¹}¥‘ôˆ°É•ÍÁ½¹Í•}µ½‘•°õI•™¥¹•µ•¹ÑIÕ¹•Ñ…¥±I•ÍÁ½¹Í”¤)…Íå¹Œ‘•˜É•™¥¹•µ•¹Ñ}ÉÕ¸ (€€€ÉÕ¹}¥èUU%°Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t(¤€´øI•™¥¹•µ•¹ÑIÕ¹•Ñ…¥±I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}É•™¥¹•µ•¹Ñ}ÉÕ¸¡Í•ÍÍ¥½¸°ÉÕ¹}¥¤(€€€•á•ÁÐ=ÁÑ¥µ¥é…Ñ¥½¹I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½É•™¥¹•µ•¹ÐµÉÕ¹Ì½íÉÕ¹}¥‘ô½•Ù…±Õ…Ñ¥½¹Ìˆ°É•ÍÁ½¹Í•}µ½‘•°õÙ…±Õ…Ñ¥½¹A…•I•ÍÁ½¹Í”(¤)…Íå¹Œ‘•˜É•™¥¹•µ•¹Ñ}ÉÕ¹}•Ù…±Õ…Ñ¥½¹Ì (€€€ÉÕ¹}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}=AQ%5%iQ%=9}A}M%i¥t€ô€ÄÀÀ°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(¤€´øÙ…±Õ…Ñ¥½¹A…•I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}ÉÕ¹}•Ù…±Õ…Ñ¥½¹Ì¡Í•ÍÍ¥½¸°ÉÕ¹}¥°±¥µ¥Ðõ±¥µ¥Ð°ÕÉÍ½ÈõÕÉÍ½È¤(€€€•á•ÁÐ=ÁÑ¥µ¥é…Ñ¥½¹I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(€€€•á•ÁÐY…±Õ•ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÈÈ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð ˆ½…¹‘¥‘…Ñ•Ìˆ°É•ÍÁ½¹Í•}µ½‘•°õ…¹‘¥‘…Ñ•A…•I•ÍÁ½¹Í”¤)…Íå¹Œ‘•˜…¹‘¥‘…Ñ•Ì (€€€ÅÕ•Éäè…¹‘¥‘…Ñ•ÍEÕ•Éä°Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t(¤€´ø…¹‘¥‘…Ñ•A…•I•ÍÁ½¹Í”è(€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}…¹‘¥‘…Ñ•Ì¡Í•ÍÍ¥½¸°ÅÕ•Éä¤(()É½ÕÑ•È¹•Ð ˆ½…¹‘¥‘…Ñ•Ì½íÍÑ…‰±•}…¹‘¥‘…Ñ•}¥‘ôˆ°É•ÍÁ½¹Í•}µ½‘•°õ…¹‘¥‘…Ñ••Ñ…¥±I•ÍÁ½¹Í”¤)…Íå¹Œ‘•˜…¹‘¥‘…Ñ” (€€€ÍÑ…‰±•}…¹‘¥‘…Ñ•}¥èÍÑÈ°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}=AQ%5%iQ%=9}A}M%i¥t€ô€ÄÀÀ°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(¤€´ø…¹‘¥‘…Ñ••Ñ…¥±I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}…¹‘¥‘…Ñ” (€€€€€€€€€€€Í•ÍÍ¥½¸°ÍÑ…‰±•}…¹‘¥‘…Ñ•}¥°±¥µ¥Ðõ±¥µ¥Ð°ÕÉÍ½ÈõÕÉÍ½È(€€€€€€€€¤(€€€•á•ÁÐ=ÁÑ¥µ¥é…Ñ¥½¹I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(€€€•á•ÁÐY…±Õ•ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÈÈ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½É•™¥¹•µ•¹ÐµÉÕ¹Ì½íÉÕ¹}¥‘ô½Ý…±¬µ™½ÉÝ…Éˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õ]…±­½ÉÝ…É‘IÕ¹I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜É•™¥¹•µ•¹Ñ}Ý…±­}™½ÉÝ…É (€€€ÉÕ¹}¥èUU%°Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t(¤€´ø]…±­½ÉÝ…É‘IÕ¹I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}É•™¥¹•µ•¹Ñ}Ý…±­}™½ÉÝ…É¡Í•ÍÍ¥½¸°ÉÕ¹}¥¤(€€€•á•ÁÐ]…±­½ÉÝ…É‘I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð ˆ½Ý…±¬µ™½ÉÝ…ÉµÉÕ¹Ì½íÉÕ¹}¥‘ôˆ°É•ÍÁ½¹Í•}µ½‘•°õ]…±­½ÉÝ…É‘IÕ¹•Ñ…¥±I•ÍÁ½¹Í”¤)…Íå¹Œ‘•˜Ý…±­}™½ÉÝ…É‘}ÉÕ¸ (€€€ÉÕ¹}¥èUU%°Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t(¤€´ø]…±­½ÉÝ…É‘IÕ¹•Ñ…¥±I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}Ý…±­}™½ÉÝ…É‘}ÉÕ¸¡Í•ÍÍ¥½¸°ÉÕ¹}¥¤(€€€•á•ÁÐ]…±­½ÉÝ…É‘I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½Ý…±¬µ™½ÉÝ…ÉµÉÕ¹Ì½íÉÕ¹}¥‘ô½…¹‘¥‘…Ñ•Ìˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õ]…±­½ÉÝ…É‘…¹‘¥‘…Ñ•A…•I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜Ý…±­}™½ÉÝ…É‘}…¹‘¥‘…Ñ•Ì (€€€ÉÕ¹}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€±…ÍÍ¥™¥…Ñ¥½¸è¹¹½Ñ…Ñ•‘m]…±­½ÉÝ…É‘±…ÍÍ¥™¥…Ñ¥½¸ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}=AQ%5%iQ%=9}A}M%i¥t€ô€ÔÀ°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(¤€´ø]…±­½ÉÝ…É‘…¹‘¥‘…Ñ•A…•I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}Ý…±­}™½ÉÝ…É‘}…¹‘¥‘…Ñ•Ì (€€€€€€€€€€€Í•ÍÍ¥½¸°(€€€€€€€€€€€ÉÕ¹}¥°(€€€€€€€€€€€±…ÍÍ¥™¥…Ñ¥½¸õ±…ÍÍ¥™¥…Ñ¥½¸°(€€€€€€€€€€€±¥µ¥Ðõ±¥µ¥Ð°(€€€€€€€€€€€ÕÉÍ½ÈõÕÉÍ½È°(€€€€€€€€¤(€€€•á•ÁÐ]…±­½ÉÝ…É‘I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(€€€•á•ÁÐY…±Õ•ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÈÈ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½Ý…±¬µ™½ÉÝ…ÉµÉÕ¹Ì½íÉÕ¹}¥‘ô½Í•±•Ñ¥½¹Ìˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õ]…±­½ÉÝ…É‘M•±•Ñ¥½¹A…•I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜Ý…±­}™½ÉÝ…É‘}Í•±•Ñ¥½¹Ì (€€€ÉÕ¹}¥èUU%°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€Í•±•Ñ•‘}™½É}Ñ•ÍÐè¹¹½Ñ…Ñ•‘m‰½½°ð9½¹”°EÕ•Éä ¥t€ô9½¹”°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}=AQ%5%iQ%=9}A}M%i¥t€ô€ÔÀ°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(¤€´ø]…±­½ÉÝ…É‘M•±•Ñ¥½¹A…•I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}Ý…±­}™½ÉÝ…É‘}Í•±•Ñ¥½¹Ì (€€€€€€€€€€€Í•ÍÍ¥½¸°(€€€€€€€€€€€ÉÕ¹}¥°(€€€€€€€€€€€Í•±•Ñ•‘}™½É}Ñ•ÍÐõÍ•±•Ñ•‘}™½É}Ñ•ÍÐ°(€€€€€€€€€€€±¥µ¥Ðõ±¥µ¥Ð°(€€€€€€€€€€€ÕÉÍ½ÈõÕÉÍ½È°(€€€€€€€€¤(€€€•á•ÁÐ]…±­½ÉÝ…É‘I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(€€€•á•ÁÐY…±Õ•ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÈÈ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(()É½ÕÑ•È¹•Ð (€€€€ˆ½Ý…±¬µ™½ÉÝ…ÉµÉÕ¹Ì½íÉÕ¹}¥‘ô½…¹‘¥‘…Ñ•Ì½í…¹‘¥‘…Ñ•}¥‘ô½Ý¥¹‘½ÝÌˆ°(€€€É•ÍÁ½¹Í•}µ½‘•°õ]…±­½ÉÝ…É‘]¥¹‘½ÝÙ…±Õ…Ñ¥½¹A…•I•ÍÁ½¹Í”°(¤)…Íå¹Œ‘•˜Ý…±­}™½ÉÝ…É‘}…¹‘¥‘…Ñ•}Ý¥¹‘½ÝÌ (€€€ÉÕ¹}¥èUU%°(€€€…¹‘¥‘…Ñ•}¥èÍÑÈ°(€€€Í•ÍÍ¥½¸è¹¹½Ñ…Ñ•‘mÍå¹M•ÍÍ¥½¸°•Á•¹‘Ì¡•Ñ}‘‰}Í•ÍÍ¥½¸¥t°(€€€±¥µ¥Ðè¹¹½Ñ…Ñ•‘m¥¹Ð°EÕ•Éä¡”ôÄ°±”õ5a}=AQ%5%iQ%=9}A}M%i¥t€ô€ÔÀ°(€€€ÕÉÍ½Èè¹¹½Ñ…Ñ•‘mÍÑÈð9½¹”°EÕ•Éä ¥t€ô9½¹”°(¤€´ø]…±­½ÉÝ…É‘]¥¹‘½ÝÙ…±Õ…Ñ¥½¹A…•I•ÍÁ½¹Í”è(€€€ÑÉäè(€€€€€€€É•ÑÕÉ¸…Ý…¥ÐÉ•…‘}Ý…±­}™½ÉÝ…É‘}…¹‘¥‘…Ñ•}Ý¥¹‘½ÝÌ (€€€€€€€€€€€Í•ÍÍ¥½¸°(€€€€€€€€€€€ÉÕ¹}¥°(€€€€€€€€€€€…¹‘¥‘…Ñ•}¥°(€€€€€€€€€€€±¥µ¥Ðõ±¥µ¥Ð°(€€€€€€€€€€€ÕÉÍ½ÈõÕÉÍ½È°(€€€€€€€€¤(€€€•á•ÁÐ]…±­½ÉÝ…É‘I•…‘9½Ñ½Õ¹‘ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÀÐ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(€€€•á•ÁÐY…±Õ•ÉÉ½È…Ì•ÉÉ½Èè(€€€€€€€É…¥Í”!QQAá•ÁÑ¥½¸¡ÍÑ…ÑÕÍ}½‘”ôÐÈÈ°‘•Ñ…¥°õÍÑÈ¡•ÉÉ½È¤¤™É½´•ÉÉ½È(