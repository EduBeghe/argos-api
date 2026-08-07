from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.core.database.session import get_db_session
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.campaign.contracts import (
    CampaignContract,
    CampaignContractsResponse,
    CampaignRequest,
)
from app.ml.campaign.models import (
    MLEconomicQualificationCampaignRecord,
    MLEconomicQualificationRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
)
from app.ml.campaign.schemas import (
    CampaignEnqueueResponse,
    CampaignPage,
    CampaignResponse,
    FamilyPage,
    FamilyResponse,
    QualificationPage,
    QualificationResponse,
)
from app.ml.campaign.service import (
    CampaignConflictError,
    CampaignNotFoundError,
    CampaignSourceError,
    cancel_campaign,
    enqueue_campaign,
    get_campaign,
    retry_campaign,
)
from app.ml.refinement.contract_service import refinement_contract
from app.ml.refinement.contracts import (
    MLRefinementRequest,
    PredictionObjective,
    RefinementClassification,
    RefinementStatus,
)
from app.ml.refinement.persistence import (
    SourceScreeningIncompleteError,
    SourceScreeningMismatchError,
    SourceScreeningResultNotFoundError,
    SourceScreeningRunNotFoundError,
)
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


@router.get("/ml-search-campaign-contract", response_model=CampaignContract)
async def ml_search_campaign_contract() -> CampaignContract:
    return CampaignContract()


@router.get("/ml-search-campaign-contract/v2", response_model=CampaignContractsResponse)
async def ml_search_campaign_contract_v2() -> CampaignContractsResponse:
    return CampaignContractsResponse()


@router.post(
    "/ml-search-campaigns",
    response_model=CampaignEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_ml_search_campaign(
    request: CampaignRequest,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CampaignEnqueueResponse:
    try:
        result = await enqueue_campaign(session, request)
    except CampaignSourceError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    location = f"/optimization/ml-search-campaigns/{result.campaign.id}"
    response.headers["Location"] = location
    if not result.created and result.campaign.status == "completed":
        response.status_code = status.HTTP_200_OK
    return CampaignEnqueueResponse(
        created=result.created,
        campaign=CampaignResponse.model_validate(result.campaign),
        location=location,
    )


@router.get("/ml-search-campaigns", response_model=CampaignPage)
async def list_ml_search_campaigns(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    campaign_status: str | None = Query(default=None, alias="status"),
    outcome: str | None = None,
    preset: str | None = None,
    source_screening_run_id: UUID | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> CampaignPage:
    statement = select(MLSearchCampaignRecord)
    count_statement = select(func.count()).select_from(MLSearchCampaignRecord)
    predicates = []
    if campaign_status is not None:
        predicates.append(MLSearchCampaignRecord.status == campaign_status)
    if outcome is not None:
        predicates.append(MLSearchCampaignRecord.outcome == outcome)
    if preset is not None:
        predicates.append(MLSearchCampaignRecord.preset == preset)
    if source_screening_run_id is not None:
        from app.ml.campaign.models import MLSearchCampaignSourceRecord

        predicates.append(
            MLSearchCampaignRecord.id.in_(
                select(MLSearchCampaignSourceRecord.campaign_id).where(
                    MLSearchCampaignSourceRecord.source_screening_run_id
                    == source_screening_run_id
                )
            )
        )
    statement = statement.where(*predicates)
    count_statement = count_statement.where(*predicates)
    total = int(await session.scalar(count_statement) or 0)
    rows = tuple(
        (
            await session.scalars(
                statement.order_by(
                    MLSearchCampaignRecord.created_at.desc(), MLSearchCampaignRecord.id
                )
                .offset(offset)
                .limit(limit)
            )
        ).all()
    )
    return CampaignPage(
        items=tuple(CampaignResponse.model_validate(item) for item in rows),
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + len(rows) < total,
    )


@router.get("/ml-search-campaigns/{campaign_id}", response_model=CampaignResponse)
async def get_ml_search_campaign(
    campaign_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> CampaignResponse:
    try:
        campaign = await get_campaign(session, campaign_id)
    except CampaignNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    family_groups = tuple(
        (
            await session.execute(
                select(
                    MLSearchCampaignFamilyRecord.search_track,
                    MLSearchCampaignFamilyRecord.state,
                    MLSearchCampaignFamilyRecord.current_stage,
                    func.count(),
                )
                .where(MLSearchCampaignFamilyRecord.campaign_id == campaign_id)
                .group_by(
                    MLSearchCampaignFamilyRecord.search_track,
                    MLSearchCampaignFamilyRecord.state,
                    MLSearchCampaignFamilyRecord.current_stage,
                )
            )
        ).all()
    )
    qualification_groups = tuple(
        (
            await session.execute(
                select(
                    MLEconomicQualificationRecord.search_track,
                    func.count(),
                )
                .where(
                    MLEconomicQualificationRecord.id.in_(
                        select(
                            MLEconomicQualificationCampaignRecord.qualification_id
                        ).where(
                            MLEconomicQualificationCampaignRecord.campaign_id
                            == campaign_id
                        )
                    ),
                    MLEconomicQualificationRecord.classification == "qualified",
                )
                .group_by(MLEconomicQualificationRecord.search_track)
            )
        ).all()
    )
    counts: dict[str, dict[str, int]] = {}
    for track, state, current_stage, count in family_groups:
        values = counts.setdefault(
            str(track),
            {"planned": 0, "covered": 0, "survived": 0, "discarded": 0, "qualified": 0},
        )
        values["planned"] += int(count)
        if int(current_stage) >= 1:
            values["covered"] += int(count)
        if state in {"survived", "strong_development_candidate"}:
            values["survived"] += int(count)
        if state == "discarded":
            values["discarded"] += int(count)
    for track, count in qualification_groups:
        counts.setdefault(
            str(track),
            {"planned": 0, "covered": 0, "survived": 0, "discarded": 0, "qualified": 0},
        )["qualified"] = int(count)
    return CampaignResponse.model_validate(campaign).model_copy(
        update={"track_counts": counts}
    )


@router.get("/ml-search-campaigns/{campaign_id}/families", response_model=FamilyPage)
async def list_ml_search_campaign_families(
    campaign_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    engine: ScreeningEngine | None = None,
    objective: PredictionObjective | None = None,
    feature_strategy: str | None = None,
    minimum_duration_seconds: int | None = Query(default=None, ge=1),
    maximum_duration_seconds: int | None = Query(default=None, ge=1),
    stage: int | None = Query(default=None, ge=0, le=4),
    family_state: str | None = Query(default=None, alias="state"),
    search_track: str | None = None,
    seed_family_id: UUID | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> FamilyPage:
    await get_ml_search_campaign(campaign_id, session)
    predicates = [MLSearchCampaignFamilyRecord.campaign_id == campaign_id]
    for column, value in (
        (MLSearchCampaignFamilyRecord.engine, engine),
        (MLSearchCampaignFamilyRecord.objective, objective),
        (MLSearchCampaignFamilyRecord.feature_variant, feature_strategy),
        (MLSearchCampaignFamilyRecord.current_stage, stage),
        (MLSearchCampaignFamilyRecord.state, family_state),
        (MLSearchCampaignFamilyRecord.search_track, search_track),
        (MLSearchCampaignFamilyRecord.seed_family_id, seed_family_id),
    ):
        if value is not None:
            predicates.append(
                column == (value.value if hasattr(value, "value") else value)
            )
    if minimum_duration_seconds is not None:
        predicates.append(
            MLSearchCampaignFamilyRecord.future_duration_seconds
            >= minimum_duration_seconds
        )
    if maximum_duration_seconds is not None:
        predicates.append(
            MLSearchCampaignFamilyRecord.future_duration_seconds
            <= maximum_duration_seconds
        )
    total = int(
        await session.scalar(
            select(func.count())
            .select_from(MLSearchCampaignFamilyRecord)
            .where(*predicates)
        )
        or 0
    )
    rows = tuple(
        (
            await session.scalars(
                select(MLSearchCampaignFamilyRecord)
                .where(*predicates)
                .order_by(
                    MLSearchCampaignFamilyRecord.deterministic_ordinal,
                    MLSearchCampaignFamilyRecord.id,
                )
                .offset(offset)
                .limit(limit)
            )
        ).all()
    )
    return FamilyPage(
        items=tuple(FamilyResponse.model_validate(item) for item in rows),
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + len(rows) < total,
    )


@router.get(
    "/ml-search-campaigns/{campaign_id}/families/{family_id}",
    response_model=FamilyResponse,
)
async def get_ml_search_campaign_family(
    campaign_id: UUID,
    family_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> FamilyResponse:
    record = await session.scalar(
        select(MLSearchCampaignFamilyRecord).where(
            MLSearchCampaignFamilyRecord.campaign_id == campaign_id,
            MLSearchCampaignFamilyRecord.id == family_id,
        )
    )
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"ML search campaign family {family_id} not found"
        )
    return FamilyResponse.model_validate(record)


@router.get(
    "/ml-search-campaigns/{campaign_id}/qualifications",
    response_model=QualificationPage,
)
async def list_ml_search_campaign_qualifications(
    campaign_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    classification: str | None = None,
    engine: ScreeningEngine | None = None,
    objective: PredictionObjective | None = None,
    search_track: str | None = None,
    seed_family_id: UUID | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> QualificationPage:
    await get_ml_search_campaign(campaign_id, session)
    predicates: list[ColumnElement[bool]] = [
        MLEconomicQualificationRecord.id.in_(
            select(MLEconomicQualificationCampaignRecord.qualification_id).where(
                MLEconomicQualificationCampaignRecord.campaign_id == campaign_id
            )
        )
    ]
    for column, value in (
        (MLEconomicQualificationRecord.classification, classification),
        (MLEconomicQualificationRecord.engine, engine),
        (MLEconomicQualificationRecord.objective, objective),
        (MLEconomicQualificationRecord.search_track, search_track),
        (MLEconomicQualificationRecord.seed_family_id, seed_family_id),
    ):
        if value is not None:
            predicates.append(
                column == (value.value if hasattr(value, "value") else value)
            )
    total = int(
        await session.scalar(
            select(func.count())
            .select_from(MLEconomicQualificationRecord)
            .where(*predicates)
        )
        or 0
    )
    rows = tuple(
        (
            await session.scalars(
                select(MLEconomicQualificationRecord)
                .where(*predicates)
                .order_by(
                    MLEconomicQualificationRecord.created_at,
                    MLEconomicQualificationRecord.id,
                )
                .offset(offset)
                .limit(limit)
            )
        ).all()
    )
    family_by_qualification = {
        qualification_id: family_id
        for qualification_id, family_id in (
            await session.execute(
                select(
                    MLEconomicQualificationCampaignRecord.qualification_id,
                    MLEconomicQualificationCampaignRecord.family_id,
                ).where(
                    MLEconomicQualificationCampaignRecord.campaign_id == campaign_id,
                    MLEconomicQualificationCampaignRecord.qualification_id.in_(
                        item.id for item in rows
                    ),
                )
            )
        ).all()
    }
    return QualificationPage(
        items=tuple(
            QualificationResponse.model_validate(item).model_copy(
                update={
                    "campaign_id": campaign_id,
                    "family_id": family_by_qualification[item.id],
                }
            )
            for item in rows
        ),
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + len(rows) < total,
    )


@router.get(
    "/ml-search-campaigns/{campaign_id}/qualifications/{qualification_id}",
    response_model=QualificationResponse,
)
async def get_ml_search_campaign_qualification(
    campaign_id: UUID,
    qualification_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> QualificationResponse:
    row = (
        await session.execute(
            select(
                MLEconomicQualificationRecord,
                MLEconomicQualificationCampaignRecord.family_id,
            )
            .join(
                MLEconomicQualificationCampaignRecord,
                MLEconomicQualificationCampaignRecord.qualification_id
                == MLEconomicQualificationRecord.id,
            )
            .where(
                MLEconomicQualificationRecord.id == qualification_id,
                MLEconomicQualificationCampaignRecord.campaign_id == campaign_id,
            )
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="ML qualification not found")
    record, family_id = row
    return QualificationResponse.model_validate(record).model_copy(
        update={"campaign_id": campaign_id, "family_id": family_id}
    )


@router.post(
    "/ml-search-campaigns/{campaign_id}/cancel", response_model=CampaignResponse
)
async def cancel_ml_search_campaign(
    campaign_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> CampaignResponse:
    try:
        return CampaignResponse.model_validate(
            await cancel_campaign(session, campaign_id)
        )
    except CampaignNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except CampaignConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post(
    "/ml-search-campaigns/{campaign_id}/retry", response_model=CampaignResponse
)
async def retry_ml_search_campaign(
    campaign_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> CampaignResponse:
    try:
        return CampaignResponse.model_validate(
            await retry_campaign(session, campaign_id)
        )
    except CampaignNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except CampaignConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


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
    except (
        SourceScreeningRunNotFoundError,
        SourceScreeningResultNotFoundError,
    ) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (SourceScreeningMismatchError, SourceScreeningIncompleteError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
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
    run_id: UUID,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RefinementEnqueueResponse:
    try:
        result = await retry_refinement(session, run_id)
    except RefinementQueueNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RefinementQueueConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    location = f"/optimization/ml-refinement-runs/{run_id}"
    response.headers["Location"] = location
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
    selection_track: Annotated[CompositeSelectionTrack | None, Query()] = None,
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
            selection_track=selection_track,
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
