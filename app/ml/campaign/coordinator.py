from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.contract import MISSING_DATA_POLICY_VERSION, VARIABLE_REGISTRY_VERSION
from app.ml.refinement.contracts import (
    ML_REFINEMENT_CLASSIFICATION_VERSION,
    ML_REFINEMENT_CONTRACT_VERSION,
    AllFeatures,
    EconomicPolicySearchConfiguration,
    FeatureSelectionSearch,
    MLRefinementRequest,
    TopKFeatures,
)
from app.ml.refinement.models import MLRefinementEvaluationRecord, MLRefinementRunRecord
from app.ml.refinement.persistence import (
    completed_run_is_complete,
    experiment_key,
    resolve_source,
)
from app.ml.refinement.planning import build_refinement_plan
from app.ml.screening.config import ScreeningEngine
from app.ml.screening.engines import library_versions
from app.ml.screening.models import MLModelScreeningResultRecord

from .contracts import (
    CAMPAIGN_CONFIDENCE_BUCKETS,
    CAMPAIGN_FEE_RATE,
    CAMPAIGN_SEED,
    CAMPAIGN_SLIPPAGE_RATE,
    ECONOMIC_POLICY_SEARCH_V2_PRESET,
    ECONOMIC_V2_CONFIDENCE_FRACTIONS,
    ECONOMIC_V2_HOLDING_MULTIPLIERS,
    ECONOMIC_V2_QUALIFICATION_END,
    ECONOMIC_V2_QUALIFICATION_VERSION,
    ECONOMIC_V2_REGIME_POLICIES,
    ECONOMIC_V2_SEED,
    ECONOMIC_V2_STRATEGY_POLICY_VERSION,
    STAGE_TRIALS,
    FeatureVariant,
    campaign_search_spaces,
    seeded_local_search_space,
)
from .economic_v2_identity import (
    qualification_identity,
    qualification_identity_payload,
)
from .economic_v2_promotion import (
    EconomicV2Evidence,
    capped_economic_v2_promotions,
    economic_v2_rank,
    is_economic_v2_development_candidate,
)
from .models import (
    MLEconomicQualificationCampaignRecord,
    MLEconomicQualificationRecord,
    MLSearchCampaignChildRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
)
from .promotion import EconomicEvidence, capped_promotions, is_strong_candidate

logger = logging.getLogger(__name__)


class CampaignCoordinationError(RuntimeError):
    """Expected persisted-state error that fails only the affected campaign."""

    code = "campaign_coordination_error"


class IncompatibleCampaignEvidenceError(CampaignCoordinationError):
    code = "incompatible_economic_policy_evidence"


def _number(value: object) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _integer(value: object) -> int:
    return int(str(value))


@dataclass(frozen=True, slots=True)
class SelectedEconomicEvidence:
    evaluation: MLRefinementEvaluationRecord
    evidence: EconomicEvidence


@dataclass(frozen=True, slots=True)
class SelectedEconomicV2Evidence:
    evaluation: MLRefinementEvaluationRecord
    evidence: EconomicV2Evidence
    policy: dict[str, object]


def _missing_last_descending(value: Decimal | None) -> Decimal:
    return -(value if value is not None else Decimal("-Infinity"))


def select_economic_evidence(
    family: MLSearchCampaignFamilyRecord, rows: tuple[MLRefinementEvaluationRecord, ...]
) -> SelectedEconomicEvidence:
    if not rows:
        raise ValueError("campaign child has no evaluation evidence")

    def key(row: MLRefinementEvaluationRecord) -> tuple[object, ...]:
        selected = _mapping(row.aggregate_metrics.get("selected_confidence_bucket"))
        return (
            -_integer(selected.get("folds_with_positive_net_return", 0)),
            _missing_last_descending(_number(selected.get("median_net_return"))),
            _missing_last_descending(_number(selected.get("worst_net_return"))),
            -_integer(row.aggregate_metrics.get("folds_beating_benchmark", 0)),
            -_integer(selected.get("observation_count", 0)),
            row.fingerprint,
        )

    best = min(rows, key=key)
    selected = _mapping(best.aggregate_metrics.get("selected_confidence_bucket"))
    return SelectedEconomicEvidence(
        best,
        EconomicEvidence(
            family.fingerprint,
            str(family.source_screening_run_id),
            family.engine,
            _integer(selected.get("observation_count", 0)),
            _integer(selected.get("folds_with_favorable_direction", 0)),
            _integer(selected.get("folds_with_positive_net_return", 0)),
            _integer(best.aggregate_metrics.get("folds_beating_benchmark", 0)),
            _number(selected.get("median_net_return")),
            _number(selected.get("worst_net_return")),
            sum(
                row.classification == "promising_for_strategy_simulation"
                for row in rows
            ),
            _integer(best.aggregate_metrics.get("complete_fold_count", 0)) == 7,
        ),
    )


def select_economic_v2_evidence(
    family: MLSearchCampaignFamilyRecord,
    rows: tuple[MLRefinementEvaluationRecord, ...],
) -> SelectedEconomicV2Evidence:
    candidates: list[SelectedEconomicV2Evidence] = []
    for row in rows:
        policy = _mapping(row.aggregate_metrics.get("selected_economic_policy_v2"))
        if not policy:
            continue
        try:
            evidence = EconomicV2Evidence(
                family.fingerprint,
                str(policy["policy_fingerprint"]),
                str(family.source_screening_run_id),
                family.engine,
                family.objective,
                _integer(policy["eligible_observations"]),
                _integer(policy["round_trips"]),
                _integer(policy["positive_net_folds"]),
                _integer(policy["folds_beating_buy_and_hold"]),
                Decimal(str(policy["median_net_return"])),
                Decimal(str(policy["worst_net_return"])),
                Decimal(str(policy["maximum_drawdown"])),
                _integer(policy["inactive_folds"]),
                Decimal(str(policy["compounded_return_1_5x"])),
                Decimal(str(policy["compounded_return_2x"])),
            )
        except (KeyError, ValueError) as error:
            raise IncompatibleCampaignEvidenceError(
                f"campaign v2 evaluation {row.id} has malformed "
                "strategy-policy evidence"
            ) from error
        candidates.append(SelectedEconomicV2Evidence(row, evidence, policy))
    if not candidates:
        raise IncompatibleCampaignEvidenceError(
            "campaign v2 child has no compatible strategy-policy evidence"
        )
    return min(candidates, key=lambda item: economic_v2_rank(item.evidence))


async def _active_shared_dependents(
    session: AsyncSession, campaign: MLSearchCampaignRecord, run_id: UUID
) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(MLSearchCampaignChildRecord)
            .join(
                MLSearchCampaignRecord,
                MLSearchCampaignRecord.id == MLSearchCampaignChildRecord.campaign_id,
            )
            .where(
                MLSearchCampaignChildRecord.refinement_run_id == run_id,
                MLSearchCampaignChildRecord.campaign_id != campaign.id,
                MLSearchCampaignRecord.status.in_(("queued", "running")),
                MLSearchCampaignRecord.cancellation_requested_at.is_(None),
            )
        )
        or 0
    )


async def _fail_campaign(
    session: AsyncSession,
    campaign: MLSearchCampaignRecord,
    *,
    failure_type: str,
    message: str,
) -> None:
    logger.error(
        "ML search campaign coordination failed",
        extra={
            "structured_fields": {
                "campaign_id": str(campaign.id),
                "failure_type": failure_type,
            }
        },
    )
    campaign.status = "failed"
    campaign.failed_at = datetime.now(UTC)
    campaign.failure_type = failure_type
    campaign.failure_message = message[:512]
    await session.commit()


async def _reconcile(session: AsyncSession, campaign: MLSearchCampaignRecord) -> bool:
    child = await session.scalar(
        select(MLSearchCampaignChildRecord)
        .where(
            MLSearchCampaignChildRecord.campaign_id == campaign.id,
        )
        .order_by(MLSearchCampaignChildRecord.created_at.desc())
        .limit(1)
    )
    if child is None:
        return False
    run = await session.get(MLRefinementRunRecord, child.refinement_run_id)
    family = await session.get(MLSearchCampaignFamilyRecord, child.family_id)
    if run is None or family is None:
        await _fail_campaign(
            session,
            campaign,
            failure_type="MissingCampaignChildReference",
            message=f"campaign child {child.id} references missing persisted state",
        )
        return True
    if run.status in {"queued", "running"}:
        if campaign.cancellation_requested_at is not None:
            if await _active_shared_dependents(session, campaign, run.id):
                campaign.status = "cancelled"
                campaign.completed_at = datetime.now(UTC)
                await session.commit()
                return True
            if run.status == "queued":
                run.status = "cancelled"
                run.completed_at = datetime.now(UTC)
                run.termination_reason = "cancelled_by_campaign"
                campaign.status = "cancelled"
                campaign.completed_at = run.completed_at
            elif run.cancellation_requested_at is None:
                run.cancellation_requested_at = datetime.now(UTC)
            await session.commit()
        return True
    if family.current_stage >= child.stage:
        return False
    if run.status in {"failed", "cancelled"}:
        if campaign.cancellation_requested_at is not None:
            campaign.status = "cancelled"
            campaign.completed_at = datetime.now(UTC)
        else:
            await _fail_campaign(
                session,
                campaign,
                failure_type="ChildRefinementFailed",
                message=f"child refinement run {run.id} ended as {run.status}",
            )
            return True
        await session.commit()
        return True
    evaluations = tuple(
        (
            await session.scalars(
                select(MLRefinementEvaluationRecord).where(
                    MLRefinementEvaluationRecord.run_id == run.id
                )
            )
        ).all()
    )
    if len(evaluations) != run.evaluations_selected:
        await _fail_campaign(
            session,
            campaign,
            failure_type="IncompleteChildEvidence",
            message=f"child refinement run {run.id} has incomplete evidence",
        )
        return True
    try:
        selected_evidence = (
            select_economic_v2_evidence(family, evaluations)
            if campaign.preset == ECONOMIC_POLICY_SEARCH_V2_PRESET
            else select_economic_evidence(family, evaluations)
        )
    except CampaignCoordinationError as error:
        await _fail_campaign(
            session,
            campaign,
            failure_type=error.code,
            message=str(error),
        )
        return True
    evidence = selected_evidence.evidence
    family.current_stage = child.stage
    family.evaluations_completed = len(evaluations)
    family.evaluations_reused = (
        len(evaluations) if child.reuses_existing_run else run.evaluations_reused
    )
    family.best_evaluation_id = selected_evidence.evaluation.id
    if isinstance(evidence, EconomicV2Evidence):
        family.best_metrics = {
            "policy_fingerprint": evidence.policy_fingerprint,
            "observations": evidence.observations,
            "round_trips": evidence.round_trips,
            "positive_net_folds": evidence.positive_net_folds,
            "folds_beating_benchmark": evidence.folds_beating_benchmark,
            "median_net_return": str(evidence.median_net_return),
            "worst_net_return": str(evidence.worst_net_return),
            "maximum_drawdown": str(evidence.maximum_drawdown),
            "inactive_folds": evidence.inactive_folds,
            "compounded_return_1_5x": str(evidence.compounded_return_1_5x),
            "compounded_return_2x": str(evidence.compounded_return_2x),
            "complete_and_finite": evidence.complete_and_finite,
        }
        assert isinstance(selected_evidence, SelectedEconomicV2Evidence)
        family.selected_policy = selected_evidence.policy
        family.selected_policy_fingerprint = evidence.policy_fingerprint
    else:
        family.best_metrics = {
            "observations": evidence.observations,
            "favorable_direction_folds": evidence.favorable_direction_folds,
            "positive_net_folds": evidence.positive_net_folds,
            "folds_beating_benchmark": evidence.folds_beating_benchmark,
            "median_net_return": (
                str(evidence.median_net_return)
                if evidence.median_net_return is not None
                else None
            ),
            "worst_net_return": (
                str(evidence.worst_net_return)
                if evidence.worst_net_return is not None
                else None
            ),
            "promising_configurations": evidence.promising_configurations,
            "complete_and_finite": evidence.complete_and_finite,
        }
    family.state = "survived"
    campaign.evaluations_computed += (
        0 if child.reuses_existing_run else run.evaluations_computed
    )
    campaign.evaluations_reused += (
        len(evaluations) if child.reuses_existing_run else run.evaluations_reused
    )
    if child.stage == 1:
        campaign.families_covered += 1
        campaign.families_uncovered -= 1
    await session.commit()
    return False


async def _reconcile_v2_qualifications(
    session: AsyncSession, campaign: MLSearchCampaignRecord
) -> bool:
    if (
        campaign.preset != ECONOMIC_POLICY_SEARCH_V2_PRESET
        or campaign.current_stage != 5
    ):
        return False
    rows = tuple(
        (
            await session.scalars(
                select(MLEconomicQualificationRecord).where(
                    MLEconomicQualificationRecord.id.in_(
                        select(
                            MLEconomicQualificationCampaignRecord.qualification_id
                        ).where(
                            MLEconomicQualificationCampaignRecord.campaign_id
                            == campaign.id
                        )
                    )
                )
            )
        ).all()
    )
    if campaign.cancellation_requested_at is not None:
        for row in rows:
            if row.status == "queued":
                row.status = "cancelled"
                row.completed_at = datetime.now(UTC)
            elif row.status == "running" and row.cancellation_requested_at is None:
                row.cancellation_requested_at = datetime.now(UTC)
        if any(row.status == "running" for row in rows):
            await session.commit()
            return True
        campaign.status = "cancelled"
        campaign.completed_at = datetime.now(UTC)
        await session.commit()
        return True
    if any(row.status in {"queued", "running"} for row in rows):
        return True
    campaign.status = "completed"
    campaign.completed_at = datetime.now(UTC)
    if not rows or any(row.status != "completed" for row in rows):
        campaign.outcome = "coverage_incomplete"
    elif any(row.classification == "qualified" for row in rows):
        campaign.outcome = "qualified_candidates_found"
    else:
        campaign.outcome = "no_qualified_candidate"
    await session.commit()
    return True


def _restore_evidence(family: MLSearchCampaignFamilyRecord) -> EconomicEvidence:
    metrics = family.best_metrics or {}
    return EconomicEvidence(
        family.fingerprint,
        str(family.source_screening_run_id),
        family.engine,
        _integer(metrics.get("observations", 0)),
        _integer(metrics.get("favorable_direction_folds", 0)),
        _integer(metrics.get("positive_net_folds", 0)),
        _integer(metrics.get("folds_beating_benchmark", 0)),
        _number(metrics.get("median_net_return")),
        _number(metrics.get("worst_net_return")),
        _integer(metrics.get("promising_configurations", 0)),
        bool(metrics.get("complete_and_finite", False)),
    )


def _restore_v2_evidence(family: MLSearchCampaignFamilyRecord) -> EconomicV2Evidence:
    metrics = family.best_metrics or {}
    return EconomicV2Evidence(
        family.fingerprint,
        str(metrics["policy_fingerprint"]),
        str(family.source_screening_run_id),
        family.engine,
        family.objective,
        _integer(metrics["observations"]),
        _integer(metrics["round_trips"]),
        _integer(metrics["positive_net_folds"]),
        _integer(metrics["folds_beating_benchmark"]),
        Decimal(str(metrics["median_net_return"])),
        Decimal(str(metrics["worst_net_return"])),
        Decimal(str(metrics["maximum_drawdown"])),
        _integer(metrics["inactive_folds"]),
        Decimal(str(metrics["compounded_return_1_5x"])),
        Decimal(str(metrics["compounded_return_2x"])),
        bool(metrics.get("complete_and_finite", False)),
    )


async def _advance_stage(
    session: AsyncSession, campaign: MLSearchCampaignRecord
) -> bool:
    stage = campaign.current_stage
    families = tuple(
        (
            await session.scalars(
                select(MLSearchCampaignFamilyRecord).where(
                    MLSearchCampaignFamilyRecord.campaign_id == campaign.id,
                    MLSearchCampaignFamilyRecord.current_stage == stage,
                    MLSearchCampaignFamilyRecord.state == "survived",
                )
            )
        ).all()
    )
    if not families:
        return False
    outstanding = int(
        await session.scalar(
            select(func.count())
            .select_from(MLSearchCampaignFamilyRecord)
            .where(
                MLSearchCampaignFamilyRecord.campaign_id == campaign.id,
                MLSearchCampaignFamilyRecord.current_stage < stage,
                MLSearchCampaignFamilyRecord.state.in_(
                    ("pending", "scheduled", "survived")
                ),
            )
        )
        or 0
    )
    if outstanding:
        return False
    if campaign.preset == ECONOMIC_POLICY_SEARCH_V2_PRESET:
        v2_evidence = tuple(_restore_v2_evidence(item) for item in families)
        if stage < 4:
            promoted = {
                item.family_fingerprint
                for item in capped_economic_v2_promotions(stage, v2_evidence)
            }
            for family in families:
                if family.fingerprint not in promoted:
                    family.state = "discarded"
                    family.decision_reason = f"failed_stage_{stage}_economic_v2_gate"
            campaign.current_stage += 1
            await session.commit()
            return True
        strong = {
            item.family_fingerprint
            for item in v2_evidence
            if is_economic_v2_development_candidate(item)
        }
        for family in families:
            family.state = (
                "strong_development_candidate"
                if family.fingerprint in strong
                else "discarded"
            )
            family.decision_reason = (
                "economic_v2_development_gate_passed"
                if family.fingerprint in strong
                else "failed_economic_v2_development_gate"
            )
        if not strong:
            campaign.status = "completed"
            campaign.outcome = "no_strong_candidate"
            campaign.completed_at = datetime.now(UTC)
        else:
            for family in families:
                if family.fingerprint not in strong:
                    continue
                if family.best_evaluation_id is None or family.selected_policy is None:
                    raise CampaignCoordinationError(
                        "development candidate lacks frozen qualification identity"
                    )
                evaluation = await session.get(
                    MLRefinementEvaluationRecord, family.best_evaluation_id
                )
                if evaluation is None:
                    raise CampaignCoordinationError(
                        "development candidate evaluation disappeared"
                    )
                run = await session.get(MLRefinementRunRecord, evaluation.run_id)
                if (
                    run is None
                    or run.status != "completed"
                    or run.source_screening_run_id != family.source_screening_run_id
                    or run.dataset_fingerprint != family.dataset_fingerprint
                    or evaluation.engine != family.engine
                    or evaluation.objective != family.objective
                    or evaluation.future_duration_seconds
                    != family.future_duration_seconds
                ):
                    raise CampaignCoordinationError(
                        "development candidate refinement identity is incompatible"
                    )
                frozen_identity = qualification_identity_payload(
                    family=family,
                    evaluation=evaluation,
                    run=run,
                    policy=family.selected_policy,
                )
                candidate_fingerprint = qualification_identity(frozen_identity)
                existing = await session.scalar(
                    select(MLEconomicQualificationRecord).where(
                        MLEconomicQualificationRecord.candidate_fingerprint
                        == candidate_fingerprint,
                        MLEconomicQualificationRecord.qualification_version
                        == ECONOMIC_V2_QUALIFICATION_VERSION,
                    )
                )
                if existing is not None:
                    if existing.frozen_inputs != frozen_identity:
                        raise CampaignCoordinationError(
                            "qualification scientific identity mismatch"
                        )
                    if existing.status in {"failed", "cancelled"}:
                        existing.status = "queued"
                        existing.classification = None
                        existing.scenario_metrics = None
                        existing.quarter_returns = None
                        existing.trades = None
                        existing.failed_rules = None
                        existing.completed_at = None
                        existing.failed_at = None
                        existing.failure_type = None
                        existing.failure_message = None
                        existing.worker_id = None
                        existing.claimed_at = None
                        existing.heartbeat_at = None
                        existing.cancellation_requested_at = None
                    linked = await session.get(
                        MLEconomicQualificationCampaignRecord,
                        (campaign.id, existing.id),
                    )
                    if linked is None:
                        session.add(
                            MLEconomicQualificationCampaignRecord(
                                campaign_id=campaign.id,
                                qualification_id=existing.id,
                                family_id=family.id,
                                created_at=datetime.now(UTC),
                            )
                        )
                    continue
                manifest = {
                    "qualification_start": "2024-01-01T00:00:00+00:00",
                    "qualification_end_exclusive": ECONOMIC_V2_QUALIFICATION_END,
                    "status": "not_read",
                }
                qualification_id = uuid4()
                session.add(
                    MLEconomicQualificationRecord(
                        id=qualification_id,
                        campaign_id=campaign.id,
                        family_id=family.id,
                        candidate_fingerprint=candidate_fingerprint,
                        policy_fingerprint=str(
                            family.selected_policy["policy_fingerprint"]
                        ),
                        qualification_version=ECONOMIC_V2_QUALIFICATION_VERSION,
                        status="queued",
                        classification=None,
                        engine=family.engine,
                        objective=family.objective,
                        search_track=family.search_track,
                        seed_family_id=family.seed_family_id,
                        frozen_inputs=frozen_identity,
                        temporal_manifest=manifest,
                        temporal_manifest_fingerprint=qualification_identity(manifest),
                        policy_identity=family.selected_policy,
                        scenario_metrics=None,
                        quarter_returns=None,
                        trades=None,
                        failed_rules=None,
                        evidence_end_exclusive=datetime(2025, 1, 1, tzinfo=UTC),
                        worker_id=None,
                        claimed_at=None,
                        heartbeat_at=None,
                        cancellation_requested_at=None,
                        completed_at=None,
                        failed_at=None,
                        failure_type=None,
                        failure_message=None,
                        created_at=datetime.now(UTC),
                    )
                )
                await session.flush()
                session.add(
                    MLEconomicQualificationCampaignRecord(
                        campaign_id=campaign.id,
                        qualification_id=qualification_id,
                        family_id=family.id,
                        created_at=datetime.now(UTC),
                    )
                )
            campaign.current_stage = 5
        await session.commit()
        return True
    evidence = tuple(_restore_evidence(item) for item in families)
    if stage < 4:
        promoted = {
            item.family_fingerprint for item in capped_promotions(stage, evidence)
        }
        for family in families:
            if family.fingerprint not in promoted:
                family.state = "discarded"
                family.decision_reason = f"failed_stage_{stage}_economic_gate"
        campaign.current_stage += 1
        await session.commit()
        return True
    strong = {item.family_fingerprint for item in evidence if is_strong_candidate(item)}
    for family in families:
        family.state = (
            "strong_development_candidate"
            if family.fingerprint in strong
            else "discarded"
        )
        family.decision_reason = (
            "strong_development_gate_passed"
            if family.fingerprint in strong
            else "failed_strong_development_gate"
        )
    campaign.status = "completed"
    campaign.outcome = "strong_candidates_found" if strong else "no_strong_candidate"
    campaign.completed_at = datetime.now(UTC)
    await session.commit()
    return True


async def _request_for_family(
    session: AsyncSession,
    family: MLSearchCampaignFamilyRecord,
    stage: int,
    runtime: int,
) -> MLRefinementRequest:
    source_result = await session.scalar(
        select(MLModelScreeningResultRecord).where(
            MLModelScreeningResultRecord.run_id == family.source_screening_run_id,
            MLModelScreeningResultRecord.engine == family.engine,
            MLModelScreeningResultRecord.future_duration_seconds
            == family.future_duration_seconds,
        )
    )
    if source_result is None:
        raise CampaignCoordinationError("campaign source result disappeared")
    variant = FeatureVariant(family.feature_variant)
    feature = (
        AllFeatures(strategy="all")
        if variant is FeatureVariant.ALL
        else TopKFeatures(strategy="top_k", values=(variant.top_k,))
    )
    if family.search_track == "seeded_local":
        if family.seed_family_id is None:
            raise CampaignCoordinationError("seeded-local family lacks seed identity")
        seed = await session.get(MLSearchCampaignFamilyRecord, family.seed_family_id)
        seed_evaluation = (
            await session.get(MLRefinementEvaluationRecord, seed.best_evaluation_id)
            if seed is not None and seed.best_evaluation_id is not None
            else None
        )
        if (
            seed is None
            or seed_evaluation is None
            or seed.engine != family.engine
            or seed.objective != family.objective
            or seed.feature_variant != family.feature_variant
            or seed.source_screening_run_id != family.source_screening_run_id
        ):
            raise CampaignCoordinationError(
                "seeded-local family has incompatible frozen seed evidence"
            )
        try:
            search_space = seeded_local_search_space(
                engine=ScreeningEngine(family.engine),
                hyperparameters=seed_evaluation.hyperparameters,
            )
        except ValueError as error:
            raise CampaignCoordinationError(str(error)) from error
    else:
        search_space = next(
            item
            for item in campaign_search_spaces()
            if item.engine.value == family.engine
        )
    return MLRefinementRequest(
        source_screening_run_id=family.source_screening_run_id,
        source_screening_result_id=source_result.id,
        engines=(search_space.engine,),
        future_durations_seconds=(family.future_duration_seconds,),
        search_space=(search_space,),
        feature_selection=FeatureSelectionSearch(strategies=(feature,)),
        prediction_objectives=(family.objective,),
        confidence_buckets=CAMPAIGN_CONFIDENCE_BUCKETS,
        fee_rate=CAMPAIGN_FEE_RATE,
        slippage_rate=CAMPAIGN_SLIPPAGE_RATE,
        maximum_trials=STAGE_TRIALS[stage - 1],
        maximum_runtime_seconds=runtime,
        deterministic_seed=(
            ECONOMIC_V2_SEED
            if family.preset == ECONOMIC_POLICY_SEARCH_V2_PRESET
            else CAMPAIGN_SEED
        ),
        economic_policy_search=(
            EconomicPolicySearchConfiguration(
                confidence_fractions=ECONOMIC_V2_CONFIDENCE_FRACTIONS,
                holding_multipliers=ECONOMIC_V2_HOLDING_MULTIPLIERS,
                regime_policies=ECONOMIC_V2_REGIME_POLICIES,
                version=ECONOMIC_V2_STRATEGY_POLICY_VERSION,
            )
            if family.preset == ECONOMIC_POLICY_SEARCH_V2_PRESET
            else None
        ),
        notes=f"ML search campaign family {family.fingerprint} stage {stage}",
    )


async def _stage_child_run(
    session: AsyncSession, request: MLRefinementRequest, timestamp: datetime
) -> tuple[MLRefinementRunRecord, bool]:
    """Stage a child without releasing the campaign row lock."""
    source, _ = await resolve_source(session, request)
    plan = build_refinement_plan(request)
    versions = library_versions()
    key = experiment_key(
        request=request,
        dataset_fingerprint=source.dataset_fingerprint,
        versions=versions,
    )
    existing = await session.scalar(
        select(MLRefinementRunRecord)
        .where(MLRefinementRunRecord.experiment_key == key)
        .with_for_update()
    )
    if existing is not None:
        if existing.status in {"queued", "running"}:
            return existing, True
        if existing.status == "completed" and await completed_run_is_complete(
            session, existing, request
        ):
            return existing, True
        if existing.status == "completed":
            raise CampaignCoordinationError(
                "identical completed child refinement is incomplete"
            )
        if existing.status not in {"failed", "cancelled"}:
            raise CampaignCoordinationError(
                "identical child refinement has an invalid lifecycle"
            )
        return existing, True
    record = MLRefinementRunRecord(
        id=uuid4(),
        experiment_key=key,
        source_screening_run_id=request.source_screening_run_id,
        source_screening_result_id=request.source_screening_result_id,
        status="queued",
        contract_version=ML_REFINEMENT_CONTRACT_VERSION,
        classification_version=ML_REFINEMENT_CLASSIFICATION_VERSION,
        dataset_fingerprint=source.dataset_fingerprint,
        variable_registry_version=VARIABLE_REGISTRY_VERSION,
        missing_data_policy_version=MISSING_DATA_POLICY_VERSION,
        library_versions=versions,
        normalized_configuration=request.normalized_payload(),
        notes=request.notes,
        maximum_trials=request.maximum_trials,
        maximum_runtime_seconds=request.maximum_runtime_seconds,
        combinations_planned=plan.total_combinations,
        evaluations_computed=0,
        evaluations_reused=0,
        evaluations_omitted=plan.omitted_by_trial_budget,
        evaluations_selected=len(plan.evaluations),
        budget_exhausted=plan.omitted_by_trial_budget > 0,
        termination_reason=None,
        started_at=None,
        completed_at=None,
        failed_at=None,
        failure_type=None,
        failure_message=None,
        worker_id=None,
        claimed_at=None,
        heartbeat_at=None,
        cancellation_requested_at=None,
        elapsed_seconds=None,
        created_at=timestamp,
    )
    session.add(record)
    return record, False


async def coordinate_next_campaign(
    session: AsyncSession, *, now: datetime | None = None
) -> bool:
    timestamp = now or datetime.now(UTC)
    campaign = await session.scalar(
        select(MLSearchCampaignRecord)
        .where(MLSearchCampaignRecord.status.in_(("queued", "running")))
        .order_by(MLSearchCampaignRecord.created_at, MLSearchCampaignRecord.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if campaign is None:
        await session.rollback()
        return False
    resumed = campaign.status == "queued" and campaign.first_claimed_at is not None
    if campaign.status == "queued":
        campaign.status = "running"
        if campaign.first_claimed_at is None:
            campaign.first_claimed_at = timestamp
    if (
        resumed
        and (timestamp - cast(datetime, campaign.first_claimed_at)).total_seconds()
        >= campaign.maximum_runtime_seconds
    ):
        campaign.status = "completed"
        campaign.outcome = (
            "coverage_incomplete"
            if campaign.families_uncovered
            else "search_budget_exhausted"
        )
        campaign.completed_at = timestamp
        await session.commit()
        return True
    active = await _reconcile(session, campaign)
    if active or campaign.status not in {"queued", "running"}:
        return True
    if await _reconcile_v2_qualifications(session, campaign):
        return True
    if campaign.cancellation_requested_at is not None:
        campaign.status = "cancelled"
        campaign.completed_at = timestamp
        await session.commit()
        return True
    elapsed = int(
        (timestamp - (campaign.first_claimed_at or timestamp)).total_seconds()
    )
    remaining_runtime = campaign.maximum_runtime_seconds - elapsed
    if remaining_runtime <= 0:
        campaign.status = "completed"
        campaign.outcome = (
            "coverage_incomplete"
            if campaign.families_uncovered
            else "search_budget_exhausted"
        )
        campaign.completed_at = timestamp
        await session.commit()
        return True
    await _advance_stage(session, campaign)
    if campaign.status == "completed":
        return True
    if (
        campaign.preset == ECONOMIC_POLICY_SEARCH_V2_PRESET
        and campaign.current_stage == 5
    ):
        # Qualification is its own lifecycle phase. Never fall through into the
        # ordinary family scheduler after stage 4 queued immutable evidence.
        return True
    stage = campaign.current_stage
    required_previous = stage - 1
    family = await session.scalar(
        select(MLSearchCampaignFamilyRecord)
        .where(
            MLSearchCampaignFamilyRecord.campaign_id == campaign.id,
            MLSearchCampaignFamilyRecord.current_stage == required_previous,
            MLSearchCampaignFamilyRecord.state.in_(("pending", "survived")),
        )
        .order_by(MLSearchCampaignFamilyRecord.deterministic_ordinal)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if family is None:
        remaining = int(
            await session.scalar(
                select(func.count())
                .select_from(MLSearchCampaignFamilyRecord)
                .where(
                    MLSearchCampaignFamilyRecord.campaign_id == campaign.id,
                    MLSearchCampaignFamilyRecord.state.in_(
                        ("pending", "scheduled", "survived")
                    ),
                )
            )
            or 0
        )
        if remaining == 0:
            campaign.status = "completed"
            campaign.outcome = "no_strong_candidate"
            campaign.completed_at = timestamp
            await session.commit()
        return True
    cumulative = STAGE_TRIALS[stage - 1]
    additional = cumulative - family.evaluations_completed
    try:
        request = await _request_for_family(
            session, family, stage, min(86_400, remaining_runtime)
        )
        child_run, reuses_existing = await _stage_child_run(session, request, timestamp)
    except CampaignCoordinationError as error:
        await _fail_campaign(
            session,
            campaign,
            failure_type=type(error).__name__,
            message=str(error),
        )
        return True
    if (
        not reuses_existing
        and campaign.evaluations_computed + additional > campaign.maximum_evaluations
    ):
        session.expunge(child_run)
        campaign.status = "completed"
        campaign.outcome = (
            "coverage_incomplete"
            if campaign.families_uncovered
            else "search_budget_exhausted"
        )
        campaign.completed_at = timestamp
        await session.commit()
        return True
    await session.flush()
    logger.info(
        "ML search campaign child selected",
        extra={
            "structured_fields": {
                "campaign_id": str(campaign.id),
                "family_id": str(family.id),
                "refinement_run_id": str(child_run.id),
                "stage": stage,
                "reuses_existing_run": reuses_existing,
            }
        },
    )
    session.add(
        MLSearchCampaignChildRecord(
            id=uuid4(),
            campaign_id=campaign.id,
            family_id=family.id,
            refinement_run_id=child_run.id,
            stage=stage,
            reserved_evaluations=additional,
            reuses_existing_run=reuses_existing,
            created_at=timestamp,
        )
    )
    family.state = "scheduled"
    family.evaluations_planned = cumulative
    await session.commit()
    return True
