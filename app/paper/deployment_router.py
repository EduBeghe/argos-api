from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.ml.protected.models import MLModelArtifactRecord
from app.ml.protected.schemas import (
    PaperDeploymentCreateRequest,
    PaperDeploymentPage,
    PaperDeploymentResponse,
)
from app.paper.deployment_models import PaperDeploymentRecord
from app.paper.deployments import (
    PaperDeploymentError,
    create_deployment,
    deployment_read_details,
    transition_deployment,
)

router = APIRouter(prefix="/paper/deployments", tags=["paper"])


def _response(
    record: PaperDeploymentRecord,
    details: tuple[Decimal | None, int, MLModelArtifactRecord | None],
) -> PaperDeploymentResponse:
    cash, position_count, artifact = details
    payload = {
        column.name: getattr(record, column.name) for column in record.__table__.columns
    }
    payload["current_cash"] = cash
    payload["open_position_count"] = position_count
    if artifact is not None:
        payload["artifact"] = {
            "id": artifact.id,
            "format_version": artifact.format_version,
            "compression": artifact.compression,
            "byte_length": artifact.byte_length,
            "sha256": artifact.sha256,
            "engine": artifact.engine,
        }
    return PaperDeploymentResponse.model_validate(payload)


def _error(error: PaperDeploymentError) -> HTTPException:
    return HTTPException(
        status_code=404 if error.code == "deployment_not_found" else 409,
        detail={"code": error.code, "message": str(error)},
    )


@router.get("", response_model=PaperDeploymentPage)
async def deployments(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    lifecycle: str | None = Query(default=None, alias="status"),
    strategy_kind: str | None = None,
    include_archived: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: UUID | None = None,
) -> PaperDeploymentPage:
    predicates = []
    if lifecycle is not None:
        predicates.append(PaperDeploymentRecord.lifecycle == lifecycle)
    elif not include_archived:
        predicates.append(PaperDeploymentRecord.lifecycle != "archived")
    if strategy_kind is not None:
        predicates.append(PaperDeploymentRecord.strategy_kind == strategy_kind)
    if cursor is not None:
        predicates.append(PaperDeploymentRecord.id > cursor)
    rows = tuple(
        (
            await session.scalars(
                select(PaperDeploymentRecord)
                .where(*predicates)
                .order_by(PaperDeploymentRecord.id)
                .limit(limit + 1)
            )
        ).all()
    )
    page = rows[:limit]
    details = await deployment_read_details(session, page)
    return PaperDeploymentPage(
        items=tuple(_response(item, details[item.id]) for item in page),
        next_cursor=str(page[-1].id) if len(rows) > limit and page else None,
        has_more=len(rows) > limit,
    )


@router.get("/{deployment_id}", response_model=PaperDeploymentResponse)
async def deployment(
    deployment_id: UUID, session: Annotated[AsyncSession, Depends(get_db_session)]
) -> PaperDeploymentResponse:
    record = await session.get(PaperDeploymentRecord, deployment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="paper deployment not found")
    details = await deployment_read_details(session, (record,))
    return _response(record, details[record.id])


@router.post("", response_model=PaperDeploymentResponse)
async def add_deployment(
    request: PaperDeploymentCreateRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> PaperDeploymentResponse:
    try:
        record = await create_deployment(
            session,
            final_validation_id=request.final_validation_id,
            display_name=request.display_name,
            make_primary=request.make_primary,
        )
        details = await deployment_read_details(session, (record,))
        return _response(record, details[record.id])
    except PaperDeploymentError as error:
        raise _error(error) from error


@router.post("/{deployment_id}/{action}", response_model=PaperDeploymentResponse)
async def deployment_action(
    deployment_id: UUID,
    action: Literal["pause", "resume", "archive", "make-primary"],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> PaperDeploymentResponse:
    try:
        record = await transition_deployment(
            session, deployment_id=deployment_id, action=action
        )
        details = await deployment_read_details(session, (record,))
        return _response(record, details[record.id])
    except PaperDeploymentError as error:
        raise _error(error) from error
