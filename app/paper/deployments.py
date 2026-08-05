from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.protected.artifacts import (
    TRUSTED_PROVENANCE,
    ArtifactIntegrityError,
    load_trusted_artifact,
)
from app.ml.protected.contracts import INITIAL_CASH
from app.ml.protected.models import (
    MLFinalValidationRecord,
    MLModelArtifactRecord,
    MLStrategySpecificationRecord,
)
from app.ml.screening.engines import library_versions
from app.paper.config import VERSION_1
from app.paper.deployment_models import PaperDeploymentRecord
from app.paper.persistence import create_run


class PaperDeploymentError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


async def deployment_read_details(
    session: AsyncSession, deployments: tuple[PaperDeploymentRecord, ...]
) -> dict[UUID, tuple[Decimal | None, int, MLModelArtifactRecord | None]]:
    run_ids = tuple(item.current_run_id for item in deployments if item.current_run_id)
    cash_by_run: dict[UUID, Decimal] = {}
    positions_by_run: dict[UUID, int] = {}
    if run_ids:
        from app.paper.persistence import PaperPositionRecord, PaperTradingRunRecord

        cash_by_run = dict(
            (
                await session.execute(
                    select(
                        PaperTradingRunRecord.id, PaperTradingRunRecord.current_cash
                    ).where(PaperTradingRunRecord.id.in_(run_ids))
                )
            ).tuples()
        )
        positions_by_run = {
            run_id: int(count)
            for run_id, count in (
                await session.execute(
                    select(PaperPositionRecord.run_id, func.count())
                    .where(
                        PaperPositionRecord.run_id.in_(run_ids),
                        PaperPositionRecord.status == "open",
                    )
                    .group_by(PaperPositionRecord.run_id)
                )
            ).tuples()
        }
    specification_ids = tuple(
        item.strategy_specification_id
        for item in deployments
        if item.strategy_specification_id
    )
    artifact_by_specification: dict[UUID, MLModelArtifactRecord] = {}
    if specification_ids:
        artifact_by_specification = {
            artifact.strategy_specification_id: artifact
            for artifact in (
                await session.scalars(
                    select(MLModelArtifactRecord).where(
                        MLModelArtifactRecord.strategy_specification_id.in_(
                            specification_ids
                        )
                    )
                )
            ).all()
        }
    return {
        deployment.id: (
            cash_by_run.get(deployment.current_run_id)
            if deployment.current_run_id
            else None,
            positions_by_run.get(deployment.current_run_id, 0)
            if deployment.current_run_id
            else 0,
            artifact_by_specification.get(deployment.strategy_specification_id)
            if deployment.strategy_specification_id
            else None,
        )
        for deployment in deployments
    }


def normalized_display_name(value: str) -> str:
    normalized = value.strip()
    if not 1 <= len(normalized) <= 80 or any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        raise PaperDeploymentError(
            "invalid_deployment_name",
            "display name must contain 1 to 80 printable characters",
        )
    return normalized


async def create_deployment(
    session: AsyncSession,
    *,
    final_validation_id: UUID,
    display_name: str,
    make_primary: bool,
    now: datetime | None = None,
) -> PaperDeploymentRecord:
    timestamp = now or datetime.now(UTC)
    name = normalized_display_name(display_name)
    await session.execute(select(PaperDeploymentRecord.id).with_for_update())
    validation = await session.get(MLFinalValidationRecord, final_validation_id)
    if (
        validation is None
        or validation.classification != "passed"
        or validation.status != "completed"
    ):
        raise PaperDeploymentError(
            "final_validation_not_passed", "a completed passed validation is required"
        )
    specification = await session.scalar(
        select(MLStrategySpecificationRecord).where(
            MLStrategySpecificationRecord.final_validation_id == validation.id
        )
    )
    if specification is None:
        raise PaperDeploymentError(
            "strategy_specification_missing",
            "validated strategy specification is missing",
        )
    artifact = await session.scalar(
        select(MLModelArtifactRecord).where(
            MLModelArtifactRecord.strategy_specification_id == specification.id
        )
    )
    if artifact is None or artifact.trusted_provenance != TRUSTED_PROVENANCE:
        raise PaperDeploymentError(
            "artifact_incompatible", "trusted compatible model artifact is required"
        )
    try:
        loaded = load_trusted_artifact(
            artifact.artifact,
            expected_sha256=artifact.sha256,
            provenance=artifact.trusted_provenance,
            format_version=artifact.format_version,
            stored_library_versions=artifact.library_versions,
            current_library_versions=library_versions(),
        )
    except ArtifactIntegrityError as error:
        raise PaperDeploymentError("artifact_incompatible", str(error)) from error
    if (
        not isinstance(loaded, dict)
        or loaded.get("features") != tuple(artifact.feature_schema)
        or loaded.get("model") is None
    ):
        raise PaperDeploymentError(
            "artifact_incompatible", "model artifact feature order is incompatible"
        )
    duplicate = await session.scalar(
        select(PaperDeploymentRecord).where(
            PaperDeploymentRecord.strategy_specification_id == specification.id,
            PaperDeploymentRecord.lifecycle != "archived",
        )
    )
    if duplicate is not None:
        return duplicate
    collision = await session.scalar(
        select(PaperDeploymentRecord.id).where(
            func.lower(PaperDeploymentRecord.display_name) == name.lower(),
            PaperDeploymentRecord.lifecycle != "archived",
        )
    )
    if collision is not None:
        raise PaperDeploymentError(
            "deployment_name_conflict", "display name is already in use"
        )
    deployment = PaperDeploymentRecord(
        id=uuid4(),
        display_name=name,
        lifecycle="active",
        strategy_kind="ml_fixed_horizon_long_only",
        strategy_specification_id=specification.id,
        current_run_id=None,
        is_primary=False,
        created_at=timestamp,
    )
    session.add(deployment)
    await session.flush()
    run = await create_run(
        session,
        initial_cash=INITIAL_CASH,
        configuration=VERSION_1,
        started_at=timestamp,
        deployment_id=deployment.id,
    )
    deployment.current_run_id = run.id
    existing_count = int(
        await session.scalar(
            select(func.count())
            .select_from(PaperDeploymentRecord)
            .where(PaperDeploymentRecord.lifecycle != "archived")
        )
        or 0
    )
    if make_primary or existing_count == 1:
        await session.execute(
            update(PaperDeploymentRecord)
            .where(PaperDeploymentRecord.is_primary.is_(True))
            .values(is_primary=False)
        )
        deployment.is_primary = True
    await session.commit()
    return deployment


async def transition_deployment(
    session: AsyncSession,
    *,
    deployment_id: UUID,
    action: str,
    now: datetime | None = None,
) -> PaperDeploymentRecord:
    timestamp = now or datetime.now(UTC)
    record = await session.scalar(
        select(PaperDeploymentRecord)
        .where(PaperDeploymentRecord.id == deployment_id)
        .with_for_update()
    )
    if record is None:
        raise PaperDeploymentError(
            "deployment_not_found", "paper deployment was not found"
        )
    if action == "pause":
        if record.lifecycle == "active":
            record.lifecycle = "paused"
            record.paused_at = timestamp
    elif action == "resume":
        if record.lifecycle == "archived":
            raise PaperDeploymentError(
                "invalid_deployment_transition", "archived deployment cannot resume"
            )
        record.lifecycle = "active"
        record.paused_at = None
    elif action == "archive":
        if record.is_primary:
            raise PaperDeploymentError(
                "primary_deployment_requires_replacement",
                "replace the primary deployment before archiving",
            )
        if record.lifecycle != "paused":
            raise PaperDeploymentError(
                "deployment_must_be_paused",
                "deployment must be paused before archiving",
            )
        if record.current_run_id is not None:
            from app.paper.persistence import PaperTradingRunRecord

            current_run = await session.scalar(
                select(PaperTradingRunRecord)
                .where(PaperTradingRunRecord.id == record.current_run_id)
                .with_for_update()
            )
            if current_run is None:
                raise PaperDeploymentError(
                    "deployment_run_missing", "deployment current run is missing"
                )
            if current_run.status == "active":
                current_run.status = "stopped"
                current_run.stopped_at = timestamp
                current_run.stop_reason = "deployment archived"
        record.lifecycle = "archived"
        record.archived_at = timestamp
    elif action == "make-primary":
        if record.lifecycle == "archived":
            raise PaperDeploymentError(
                "archived_deployment", "archived deployment cannot be primary"
            )
        await session.execute(
            update(PaperDeploymentRecord)
            .where(PaperDeploymentRecord.is_primary.is_(True))
            .values(is_primary=False)
        )
        record.is_primary = True
    else:
        raise ValueError("unknown deployment transition")
    await session.commit()
    return record
