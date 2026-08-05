from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session

from .audit import assemble_family_audit
from .models import MLFinalValidationRecord, MLTemporalManifestRecord
from .queue import (
    ProtectedWorkError,
    cancel_work,
    enqueue_final_validation,
    enqueue_manifest_reconstruction,
    retry_final_validation,
    retry_manifest,
)
from .schemas import (
    CampaignFamilyAuditResponse,
    FinalValidationPage,
    FinalValidationResponse,
    TemporalManifestResponse,
)

router = APIRouter(prefix="/optimization", tags=["optimization"])


def _http_error(error: ProtectedWorkError) -> HTTPException:
    return HTTPException(
        status_code=404 if error.code.endswith("not_found") else 409,
        detail={"code": error.code, "message": str(error)},
    )


@router.get(
    "/ml-search-campaigns/{campaign_id}/families/{family_id}/audit",
    response_model=CampaignFamilyAuditResponse,
)
async def family_audit(
    campaign_id: UUID,
    family_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CampaignFamilyAuditResponse:
    result = await assemble_family_audit(
        session, campaign_id=campaign_id, family_id=family_id
    )
    if result is None:
        raise HTTPException(status_code=404, detail="campaign family not found")
    return result


@router.post(
    "/ml-search-campaigns/{campaign_id}/families/{family_id}/development-manifest-reconstructions",
    response_model=TemporalManifestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reconstruct_development_manifest(
    campaign_id: UUID,
    family_id: UUID,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MLTemporalManifestRecord:
    try:
        record = await enqueue_manifest_reconstruction(
            session, campaign_id=campaign_id, family_id=family_id
        )
    except ProtectedWorkError as error:
        raise _http_error(error) from error
    response.headers["Location"] = (
        f"/optimization/ml-development-manifest-reconstructions/{record.id}"
    )
    return record


@router.get(
    "/ml-development-manifest-reconstructions/{manifest_id}",
    response_model=TemporalManifestResponse,
)
async def development_manifest(
    manifest_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MLTemporalManifestRecord:
    record = await session.get(MLTemporalManifestRecord, manifest_id)
    if record is None:
        raise HTTPException(status_code=404, detail="development manifest not found")
    return record


@router.post(
    "/ml-development-manifest-reconstructions/{manifest_id}/cancel",
    response_model=TemporalManifestResponse,
)
async def cancel_development_manifest(
    manifest_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MLTemporalManifestRecord:
    try:
        return await cancel_work(session, MLTemporalManifestRecord, manifest_id)
    except ProtectedWorkError as error:
        raise _http_error(error) from error


@router.post(
    "/ml-development-manifest-reconstructions/{manifest_id}/retry",
    response_model=TemporalManifestResponse,
)
async def retry_development_manifest(
    manifest_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MLTemporalManifestRecord:
    try:
        return await retry_manifest(session, manifest_id)
    except ProtectedWorkError as error:
        raise _http_error(error) from error


@router.post(
    "/ml-search-campaigns/{campaign_id}/families/{family_id}/final-validations",
    response_model=FinalValidationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_final_validation(
    campaign_id: UUID,
    family_id: UUID,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MLFinalValidationRecord:
    try:
        record = await enqueue_final_validation(
            session, campaign_id=campaign_id, family_id=family_id
        )
    except ProtectedWorkError as error:
        raise _http_error(error) from error
    response.headers["Location"] = f"/optimization/ml-final-validations/{record.id}"
    return record


@router.get("/ml-final-validations", response_model=FinalValidationPage)
async def final_validations(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    campaign_id: UUID | None = None,
    family_id: UUID | None = None,
    validation_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: UUID | None = None,
) -> FinalValidationPage:
    predicates = []
    for column, value in (
        (MLFinalValidationRecord.campaign_id, campaign_id),
        (MLFinalValidationRecord.family_id, family_id),
        (MLFinalValidationRecord.status, validation_status),
    ):
        if value is not None:
            predicates.append(column == value)
    if cursor is not None:
        predicates.append(MLFinalValidationRecord.id > cursor)
    rows = tuple(
        (
            await session.scalars(
                select(MLFinalValidationRecord)
                .where(*predicates)
                .order_by(MLFinalValidationRecord.id)
                .limit(limit + 1)
            )
        ).all()
    )
    has_more = len(rows) > limit
    page = rows[:limit]
    return FinalValidationPage(
        items=tuple(FinalValidationResponse.model_validate(item) for item in page),
        next_cursor=str(page[-1].id) if has_more and page else None,
        has_more=has_more,
    )


@router.get(
    "/ml-final-validations/{validation_id}", response_model=FinalValidationResponse
)
async def final_validation(
    validation_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MLFinalValidationRecord:
    record = await session.get(MLFinalValidationRecord, validation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="final validation not found")
    return record


@router.post(
    "/ml-final-validations/{validation_id}/retry",
    response_model=FinalValidationResponse,
)
async def retry_validation(
    validation_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MLFinalValidationRecord:
    try:
        return await retry_final_validation(session, validation_id)
    except ProtectedWorkError as error:
        raise _http_error(error) from error


@router.post(
    "/ml-final-validations/{validation_id}/cancel",
    response_model=FinalValidationResponse,
)
async def cancel_validation(
    validation_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MLFinalValidationRecord:
    try:
        return await cancel_work(session, MLFinalValidationRecord, validation_id)
    except ProtectedWorkError as error:
        raise _http_error(error) from error
