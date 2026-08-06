"""Deterministic, read-only development economic re-audit for ML campaigns."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Final
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument, InstrumentCapability, resolve_instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.contract import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FUTURE_DURATIONS,
    SPLIT_GAP,
    TemporalFoldDefinition,
    TrainingHistoryMode,
)
from app.ml.dataset import (
    MLExperimentDataset,
    TemporalFoldManifest,
    build_ml_experiment_dataset,
)
from app.ml.gaps import datetime_to_ns, ns_to_datetime
from app.ml.protected.contracts import (
    ConfidenceBucketContractError,
    canonicalize_confidence_bucket,
)
from app.ml.protected.policy import (
    FinalValidationPolicyEvidence,
    failed_final_validation_rules,
)
from app.ml.protected.simulation import (
    ProtectedBenchmarkResult,
    ProtectedPrediction,
    ProtectedSimulationResult,
    simulate_buy_and_hold,
    simulate_fixed_horizon_long_only,
)
from app.ml.protected.validation import (
    _candidate,
    _signed_positive_scores,
    confidence_score_cutoff,
    is_decision_boundary_ns,
)
from app.ml.reader import MLCandleColumns, read_ml_experiment_candles
from app.ml.refinement.calculation import _selected_columns, fit_refinement_model
from app.ml.refinement.contracts import PredictionObjective
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)
from app.ml.registry import VARIABLE_REGISTRY
from app.ml.screening.config import ScreeningEngine
from app.ml.screening.models import MLModelScreeningRunRecord
from app.paper.config import SimulationConfiguration, quantize_financial

from .contracts import (
    CAMPAIGN_CLASSIFICATION_VERSION,
    CAMPAIGN_CONFIDENCE_BUCKETS,
    CAMPAIGN_CONTRACT_VERSION,
    CAMPAIGN_FEE_RATE,
    CAMPAIGN_POLICY_VERSION,
    CAMPAIGN_PRESET,
    CAMPAIGN_SEED,
    CAMPAIGN_SLIPPAGE_RATE,
    FAMILIES_PER_INSTRUMENT,
    PROMOTION_POLICY,
    STAGE_TRIALS,
    FeatureVariant,
)
from .models import (
    MLSearchCampaignChildRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
    MLSearchCampaignSourceRecord,
)

REPORT_CONTRACT_VERSION: Final = 1
DEFAULT_MAXIMUM_CANDIDATES: Final = 15
MAXIMUM_CANDIDATES: Final = 25
POLICY_RULE_SOURCE: Final = "final_validation_policy_v1"
SELECTION_POLICY_VERSION: Final = 1
ACCUMULATING_FOLDS: Final = tuple(
    TemporalFoldDefinition(
        name=f"{validation_start:%Y%m}_accumulating",
        history_mode=TrainingHistoryMode.ACCUMULATING,
        training_start=DEVELOPMENT_START,
        training_end=validation_start - SPLIT_GAP,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    for validation_start, validation_end in (
        (datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 7, 1, tzinfo=UTC)),
        (datetime(2022, 7, 1, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC)),
        (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 7, 1, tzinfo=UTC)),
        (datetime(2023, 7, 1, tzinfo=UTC), DEVELOPMENT_END),
    )
)
ACCUMULATING_FOLD_NAMES: Final = tuple(fold.name for fold in ACCUMULATING_FOLDS)
assert len(ACCUMULATING_FOLD_NAMES) == 4


class DevelopmentReauditError(RuntimeError):
    """The campaign cannot produce a scientifically valid diagnostic report."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    family: MLSearchCampaignFamilyRecord
    evaluation: MLRefinementEvaluationRecord
    confidence_fraction: Decimal
    observations: int
    positive_net_folds: int
    favorable_direction_folds: int
    folds_beating_benchmark: int
    median_net_return: Decimal
    worst_net_return: Decimal


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a finite Decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a finite Decimal") from error
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def normalized_selection_policy() -> dict[str, object]:
    return {
        "version": SELECTION_POLICY_VERSION,
        "family_state": "discarded",
        "current_stage": 4,
        "objective": PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION.value,
        "complete_and_finite": True,
        "minimum_observations": 500,
        "minimum_positive_net_folds": 5,
        "minimum_favorable_direction_folds": 5,
        "minimum_folds_beating_benchmark": 4,
        "minimum_exclusive_median_net_return": "0",
        "minimum_inclusive_worst_net_return": "-0.02",
        "ranking": [
            "positive_net_folds_desc",
            "favorable_direction_folds_desc",
            "folds_beating_benchmark_desc",
            "worst_net_return_desc",
            "median_net_return_desc",
            "observations_desc",
            "deterministic_ordinal_asc",
            "family_id_asc",
        ],
    }


def candidate_evidence(
    family: MLSearchCampaignFamilyRecord,
    evaluation: MLRefinementEvaluationRecord,
) -> CandidateEvidence:
    metrics = family.best_metrics
    if not isinstance(metrics, dict):
        raise ValueError("best_metrics is missing")
    if metrics.get("complete_and_finite") is not True:
        raise ValueError("best_metrics is incomplete or non-finite")
    if evaluation.objective != PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION:
        raise ValueError("best evaluation objective is unsupported")
    selected = evaluation.aggregate_metrics.get("selected_confidence_bucket")
    if not isinstance(selected, dict):
        raise ValueError("selected confidence bucket is missing")
    try:
        fraction = canonicalize_confidence_bucket(
            selected.get("fraction"), evaluation.confidence_buckets
        )
    except ConfidenceBucketContractError as error:
        raise ValueError("selected confidence bucket is invalid") from error
    return CandidateEvidence(
        family,
        evaluation,
        fraction,
        _integer(metrics.get("observations"), name="observations"),
        _integer(metrics.get("positive_net_folds"), name="positive_net_folds"),
        _integer(
            metrics.get("favorable_direction_folds"),
            name="favorable_direction_folds",
        ),
        _integer(
            metrics.get("folds_beating_benchmark"),
            name="folds_beating_benchmark",
        ),
        _decimal(metrics.get("median_net_return"), name="median_net_return"),
        _decimal(metrics.get("worst_net_return"), name="worst_net_return"),
    )


def is_eligible(item: CandidateEvidence) -> bool:
    family = item.family
    return (
        family.state == "discarded"
        and family.current_stage == 4
        and family.best_evaluation_id == item.evaluation.id
        and item.observations >= 500
        and item.positive_net_folds >= 5
        and item.favorable_direction_folds >= 5
        and item.folds_beating_benchmark >= 4
        and item.median_net_return > 0
        and item.worst_net_return >= Decimal("-0.02")
    )


def candidate_rank(item: CandidateEvidence) -> tuple[object, ...]:
    return (
        -item.positive_net_folds,
        -item.favorable_direction_folds,
        -item.folds_beating_benchmark,
        -item.worst_net_return,
        -item.median_net_return,
        -item.observations,
        item.family.deterministic_ordinal,
        str(item.family.id),
    )


def _horizon_band(seconds: int) -> str:
    if seconds < 12 * 3600:
        return "under_12h"
    if seconds < 24 * 3600:
        return "12h_to_under_24h"
    if seconds <= 48 * 3600:
        return "24h_through_48h"
    return "over_48h"


def funnel_diagnosis(
    families: Sequence[MLSearchCampaignFamilyRecord], *, expected_total: int
) -> dict[str, object]:
    allowed_states = {
        "pending",
        "scheduled",
        "survived",
        "discarded",
        "strong_development_candidate",
    }
    if len(families) != expected_total:
        raise DevelopmentReauditError(
            "campaign_family_count_mismatch",
            "persisted family count does not match campaign total",
        )
    identities: set[tuple[UUID, str]] = set()
    dimensions: dict[str, Counter[str]] = {
        "objective": Counter(),
        "engine": Counter(),
        "feature_variant": Counter(),
        "horizon_band": Counter(),
    }
    terminal: Counter[str] = Counter()
    for family in families:
        identity = (family.id, family.fingerprint)
        if identity in identities:
            raise DevelopmentReauditError(
                "duplicate_family_identity", "duplicate family identity detected"
            )
        identities.add(identity)
        if family.state not in allowed_states or not 0 <= family.current_stage <= 4:
            raise DevelopmentReauditError(
                "invalid_family_state", "family state or stage is unsupported"
            )
        terminal_key = f"stage_{family.current_stage}:{family.state}"
        terminal[terminal_key] += 1
        dimensions["objective"][f"{family.objective}:{terminal_key}"] += 1
        dimensions["engine"][f"{family.engine}:{terminal_key}"] += 1
        dimensions["feature_variant"][f"{family.feature_variant}:{terminal_key}"] += 1
        dimensions["horizon_band"][
            f"{_horizon_band(family.future_duration_seconds)}:{terminal_key}"
        ] += 1
    if sum(terminal.values()) != expected_total:
        raise DevelopmentReauditError(
            "campaign_funnel_mismatch", "campaign funnel does not reconcile"
        )
    return {
        "families_total": expected_total,
        "terminal_stage_state": dict(sorted(terminal.items())),
        "by_objective": dict(sorted(dimensions["objective"].items())),
        "by_engine": dict(sorted(dimensions["engine"].items())),
        "by_feature_variant": dict(sorted(dimensions["feature_variant"].items())),
        "by_horizon_band": dict(sorted(dimensions["horizon_band"].items())),
        "near_misses": {
            "stage_3": sum(
                family.state == "discarded" and family.current_stage == 3
                for family in families
            ),
            "stage_4": sum(
                family.state == "discarded" and family.current_stage == 4
                for family in families
            ),
        },
    }


def _compound(values: Sequence[Decimal]) -> Decimal:
    result = Decimal(1)
    for value in values:
        result *= Decimal(1) + value
    return quantize_financial(result - Decimal(1))


def aggregate_blocks(
    base: Sequence[ProtectedSimulationResult],
    stressed_1_5x: Sequence[ProtectedSimulationResult],
    stressed_2x: Sequence[ProtectedSimulationResult],
) -> tuple[dict[str, object], tuple[str, ...]]:
    if not len(base) == len(stressed_1_5x) == len(stressed_2x) == 4:
        raise ValueError("exactly four economic blocks are required")
    half_year_returns = tuple(item.total_return for item in base)
    total_return = _compound(half_year_returns)
    return_1_5x = _compound(tuple(item.total_return for item in stressed_1_5x))
    return_2x = _compound(tuple(item.total_return for item in stressed_2x))
    maximum_drawdown = max(item.maximum_drawdown for item in base)
    evidence = FinalValidationPolicyEvidence(
        all(
            value.is_finite()
            for value in (
                total_return,
                return_1_5x,
                return_2x,
                maximum_drawdown,
                *half_year_returns,
            )
        ),
        sum(item.round_trips for item in base),
        total_return,
        half_year_returns,
        maximum_drawdown,
        return_1_5x,
        return_2x,
    )
    failed = failed_final_validation_rules(evidence)
    return (
        {
            "half_year_returns": half_year_returns,
            "round_trips": evidence.round_trips,
            "total_return": total_return,
            "stressed_1_5x_total_return": return_1_5x,
            "stressed_2x_total_return": return_2x,
            "maximum_drawdown": maximum_drawdown,
            "positive_half_years": sum(value > 0 for value in half_year_returns),
            "wins": sum(item.wins for item in base),
            "losses": sum(item.losses for item in base),
            "fees": quantize_financial(sum((item.fees for item in base), Decimal(0))),
            "slippage": quantize_financial(
                sum((item.slippage for item in base), Decimal(0))
            ),
        },
        failed,
    )


def _simulation_metrics(result: ProtectedSimulationResult) -> dict[str, object]:
    return {
        key: value
        for key, value in asdict(result).items()
        if key not in {"trades", "equity_curve"}
    }


def _benchmark_metrics(result: ProtectedBenchmarkResult) -> dict[str, object]:
    return asdict(result)


def _folds(dataset: MLExperimentDataset) -> tuple[TemporalFoldManifest, ...]:
    return tuple(
        TemporalFoldManifest(
            definition=definition,
            training_indices=np.flatnonzero(
                (dataset.boundary_time_ns >= datetime_to_ns(definition.training_start))
                & (dataset.boundary_time_ns < datetime_to_ns(definition.training_end))
            ).astype(np.int64),
            validation_indices=np.flatnonzero(
                (
                    dataset.boundary_time_ns
                    >= datetime_to_ns(definition.validation_start)
                )
                & (dataset.boundary_time_ns < datetime_to_ns(definition.validation_end))
            ).astype(np.int64),
        )
        for definition in ACCUMULATING_FOLDS
    )


def calculate_candidate_reaudit(
    *,
    dataset: MLExperimentDataset,
    candles: MLCandleColumns,
    instrument: Instrument,
    evidence: CandidateEvidence,
    seed: int = CAMPAIGN_SEED,
) -> dict[str, object]:
    if np.any(dataset.boundary_time_ns >= datetime_to_ns(DEVELOPMENT_END)):
        raise ValueError("development dataset reaches protected evidence")
    candidate = _candidate(evidence.evaluation)
    if candidate.objective is not PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION:
        raise ValueError("development re-audit supports direction classification only")
    target = dataset.targets.for_duration(
        timedelta(seconds=candidate.future_duration_seconds)
    )
    block_reports: list[dict[str, object]] = []
    base_results: list[ProtectedSimulationResult] = []
    stressed_1_5x: list[ProtectedSimulationResult] = []
    stressed_2x: list[ProtectedSimulationResult] = []
    for fold in _folds(dataset):
        eligible = dataset.feature_eligible & target.eligible
        training = fold.training_indices[eligible[fold.training_indices]]
        validation = fold.validation_indices[eligible[fold.validation_indices]]
        if not len(training) or not len(validation):
            raise ValueError(f"incomplete fold evidence: {fold.definition.name}")
        columns = _selected_columns(
            candidate, dataset.features[training], target.values[training]
        )
        labels = (target.values[training] > 0).astype(np.int64)
        if len(np.unique(labels)) < 2:
            raise ValueError(f"single-class training fold: {fold.definition.name}")
        model = fit_refinement_model(
            candidate=candidate,
            training_x=dataset.features[np.ix_(training, columns)],
            labels=labels,
            seed=seed,
        )
        training_scores = _signed_positive_scores(
            model=model, values=dataset.features[np.ix_(training, columns)]
        )
        cutoff = confidence_score_cutoff(training_scores, evidence.confidence_fraction)
        validation_scores = _signed_positive_scores(
            model=model, values=dataset.features[np.ix_(validation, columns)]
        )
        predictions: list[ProtectedPrediction] = []
        validation_start = datetime_to_ns(fold.definition.validation_start)
        validation_end = datetime_to_ns(fold.definition.validation_end)
        for row, score in zip(validation, validation_scores, strict=True):
            boundary_ns = int(dataset.boundary_time_ns[int(row)])
            if (
                not validation_start <= boundary_ns < validation_end
                or not is_decision_boundary_ns(boundary_ns)
            ):
                continue
            reference_ns = boundary_ns - 300 * 1_000_000_000
            candle_index = int(np.searchsorted(candles.open_time_ns, reference_ns))
            if (
                candle_index >= len(candles)
                or int(candles.open_time_ns[candle_index]) != reference_ns
            ):
                continue
            predictions.append(
                ProtectedPrediction(
                    ns_to_datetime(boundary_ns),
                    Decimal(str(score)),
                    Decimal(str(candles.close_price[candle_index])),
                )
            )
        if not predictions:
            raise ValueError(
                f"fold has no cadence-aligned predictions: {fold.definition.name}"
            )
        hold = timedelta(seconds=candidate.future_duration_seconds)
        results = tuple(
            simulate_fixed_horizon_long_only(
                predictions=tuple(predictions),
                instrument=instrument,
                threshold=cutoff,
                hold=hold,
                configuration=SimulationConfiguration(
                    fee_rate=CAMPAIGN_FEE_RATE * multiplier,
                    slippage_rate=CAMPAIGN_SLIPPAGE_RATE * multiplier,
                ),
            )
            for multiplier in (Decimal(1), Decimal("1.5"), Decimal(2))
        )
        benchmark = simulate_buy_and_hold(
            predictions=tuple(predictions),
            instrument=instrument,
            configuration=SimulationConfiguration(
                fee_rate=CAMPAIGN_FEE_RATE,
                slippage_rate=CAMPAIGN_SLIPPAGE_RATE,
            ),
        )
        base_results.append(results[0])
        stressed_1_5x.append(results[1])
        stressed_2x.append(results[2])
        block_reports.append(
            {
                "fold_name": fold.definition.name,
                "training_start": fold.definition.training_start,
                "training_end_exclusive": fold.definition.training_end,
                "validation_start": fold.definition.validation_start,
                "validation_end_exclusive": fold.definition.validation_end,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "prediction_count": len(predictions),
                "selected_features": tuple(
                    VARIABLE_REGISTRY[index].name for index in columns
                ),
                "confidence_score_cutoff": cutoff,
                "base": _simulation_metrics(results[0]),
                "stress_1_5x": _simulation_metrics(results[1]),
                "stress_2x": _simulation_metrics(results[2]),
                "buy_and_hold": _benchmark_metrics(benchmark),
            }
        )
    aggregate, failed = aggregate_blocks(base_results, stressed_1_5x, stressed_2x)
    return {
        "status": "completed",
        "diagnostic_only": True,
        "protected_evidence_read": False,
        "development_proxy_pass": not failed,
        "policy_rule_source": POLICY_RULE_SOURCE,
        "failed_rules": failed,
        "blocks": tuple(block_reports),
        "aggregate": aggregate,
    }


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("report datetime must be timezone-aware")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("report float must be finite")
        return str(Decimal(str(value)))
    return value


def report_json(report: Mapping[str, object]) -> str:
    return json.dumps(
        _json_value(dict(report)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(report_json(value).encode("utf-8")).hexdigest()


async def build_development_reaudit_report(
    *,
    session: AsyncSession,
    campaign_id: UUID,
    maximum_candidates: int = DEFAULT_MAXIMUM_CANDIDATES,
) -> tuple[dict[str, object], bool]:
    if isinstance(maximum_candidates, bool) or not 1 <= maximum_candidates <= 25:
        raise DevelopmentReauditError(
            "invalid_maximum_candidates", "maximum candidates must be between 1 and 25"
        )
    campaign = await session.get(MLSearchCampaignRecord, campaign_id)
    if campaign is None:
        raise DevelopmentReauditError("campaign_not_found", "campaign does not exist")
    request = campaign.normalized_request
    if (
        campaign.status != "completed"
        or campaign.families_uncovered != 0
        or campaign.families_covered != campaign.families_total
    ):
        raise DevelopmentReauditError(
            "campaign_not_complete", "campaign is not fully completed and covered"
        )
    if (
        campaign.contract_version != CAMPAIGN_CONTRACT_VERSION
        or campaign.policy_version != CAMPAIGN_POLICY_VERSION
        or campaign.classification_version != CAMPAIGN_CLASSIFICATION_VERSION
        or request.get("preset") != CAMPAIGN_PRESET
        or request.get("promotion_policy") != PROMOTION_POLICY
        or request.get("contract_version") != CAMPAIGN_CONTRACT_VERSION
        or request.get("policy_version") != CAMPAIGN_POLICY_VERSION
        or request.get("classification_version") != CAMPAIGN_CLASSIFICATION_VERSION
        or request.get("stage_trials") != list(STAGE_TRIALS)
        or request.get("seed") != CAMPAIGN_SEED
        or request.get("fee_rate") != str(CAMPAIGN_FEE_RATE)
        or request.get("slippage_rate") != str(CAMPAIGN_SLIPPAGE_RATE)
        or request.get("confidence_buckets")
        != [str(item) for item in CAMPAIGN_CONFIDENCE_BUCKETS]
    ):
        raise DevelopmentReauditError(
            "unsupported_campaign_contract", "campaign contract is incompatible"
        )
    sources = tuple(
        (
            await session.scalars(
                select(MLSearchCampaignSourceRecord)
                .where(MLSearchCampaignSourceRecord.campaign_id == campaign_id)
                .order_by(MLSearchCampaignSourceRecord.source_screening_run_id)
            )
        ).all()
    )
    if not sources or campaign.families_total != len(sources) * FAMILIES_PER_INSTRUMENT:
        raise DevelopmentReauditError(
            "campaign_source_mismatch", "campaign source coverage is inconsistent"
        )
    source_ids = {source.source_screening_run_id for source in sources}
    if request.get("source_screening_run_ids") != sorted(
        str(source_id) for source_id in source_ids
    ):
        raise DevelopmentReauditError(
            "campaign_source_mismatch",
            "normalized campaign sources do not match persisted sources",
        )
    screening = {
        row.id: row
        for row in (
            await session.scalars(
                select(MLModelScreeningRunRecord).where(
                    MLModelScreeningRunRecord.id.in_(source_ids)
                )
            )
        ).all()
    }
    if set(screening) != source_ids or any(
        screening[item.source_screening_run_id].status != "completed"
        or screening[item.source_screening_run_id].dataset_fingerprint
        != item.dataset_fingerprint
        for item in sources
    ):
        raise DevelopmentReauditError(
            "campaign_source_mismatch",
            "source ownership or fingerprint is inconsistent",
        )
    families = tuple(
        (
            await session.scalars(
                select(MLSearchCampaignFamilyRecord)
                .where(MLSearchCampaignFamilyRecord.campaign_id == campaign_id)
                .order_by(
                    MLSearchCampaignFamilyRecord.deterministic_ordinal,
                    MLSearchCampaignFamilyRecord.id,
                )
            )
        ).all()
    )
    if (
        len({family.id for family in families}) != len(families)
        or len({family.fingerprint for family in families}) != len(families)
        or len({family.deterministic_ordinal for family in families}) != len(families)
    ):
        raise DevelopmentReauditError(
            "duplicate_family_identity", "campaign family identities are not unique"
        )
    funnel = funnel_diagnosis(families, expected_total=campaign.families_total)
    if any(family.source_screening_run_id not in source_ids for family in families):
        raise DevelopmentReauditError(
            "campaign_family_source_mismatch", "family references an unknown source"
        )
    for family in families:
        source = screening[family.source_screening_run_id]
        if (
            family.dataset_fingerprint != source.dataset_fingerprint
            or family.asset != source.asset
            or family.provider != source.provider
            or family.market_type != source.market_type
            or family.interval != source.interval
        ):
            raise DevelopmentReauditError(
                "campaign_family_source_mismatch",
                "family identity does not match its screening source",
            )
        if (
            family.engine not in {item.value for item in ScreeningEngine}
            or family.objective not in {item.value for item in PredictionObjective}
            or family.feature_variant not in {item.value for item in FeatureVariant}
            or family.future_duration_seconds
            not in {int(item.total_seconds()) for item in FUTURE_DURATIONS}
        ):
            raise DevelopmentReauditError(
                "unsupported_family_contract",
                "family configuration is outside the exhaustive campaign contract",
            )
    stage_four = tuple(
        family
        for family in families
        if family.state == "discarded"
        and family.current_stage == 4
        and family.objective
        == PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION.value
        and family.best_evaluation_id is not None
    )
    evaluation_ids = {family.best_evaluation_id for family in stage_four}
    evaluations = {
        row.id: row
        for row in (
            await session.scalars(
                select(MLRefinementEvaluationRecord).where(
                    MLRefinementEvaluationRecord.id.in_(evaluation_ids)
                )
            )
        ).all()
    }
    child_rows = tuple(
        (
            await session.scalars(
                select(MLSearchCampaignChildRecord).where(
                    MLSearchCampaignChildRecord.campaign_id == campaign_id
                )
            )
        ).all()
    )
    child_runs: dict[UUID, set[UUID]] = defaultdict(set)
    for child in child_rows:
        child_runs[child.family_id].add(child.refinement_run_id)
    run_ids = {evaluation.run_id for evaluation in evaluations.values()}
    runs = {
        row.id: row
        for row in (
            await session.scalars(
                select(MLRefinementRunRecord).where(
                    MLRefinementRunRecord.id.in_(run_ids)
                )
            )
        ).all()
    }
    eligible: list[CandidateEvidence] = []
    malformed: dict[UUID, str] = {}
    for family in stage_four:
        assert family.best_evaluation_id is not None
        evaluation = evaluations.get(family.best_evaluation_id)
        if evaluation is None:
            malformed[family.id] = "best_evaluation_missing"
            continue
        run = runs.get(evaluation.run_id)
        if (
            run is None
            or evaluation.run_id not in child_runs.get(family.id, set())
            or run.source_screening_run_id != family.source_screening_run_id
        ):
            malformed[family.id] = "best_evaluation_ownership_invalid"
            continue
        try:
            evidence = candidate_evidence(family, evaluation)
        except ValueError as error:
            malformed[family.id] = str(error)
            continue
        if is_eligible(evidence):
            eligible.append(evidence)
    eligible.sort(key=candidate_rank)
    selected = tuple(eligible[:maximum_candidates])
    selection_identity: dict[str, object] = {
        "campaign_id": campaign_id,
        "dataset_fingerprints": sorted(
            source.dataset_fingerprint for source in sources
        ),
        "development_start": DEVELOPMENT_START,
        "development_end_exclusive": DEVELOPMENT_END,
        "maximum_candidates": maximum_candidates,
        "policy": normalized_selection_policy(),
        "selected": [
            {
                "family_id": item.family.id,
                "family_fingerprint": item.family.fingerprint,
                "evaluation_id": item.evaluation.id,
                "evaluation_fingerprint": item.evaluation.fingerprint,
                "confidence_fraction": item.confidence_fraction,
            }
            for item in selected
        ],
    }
    results: list[dict[str, object]] = []
    cache: dict[UUID, tuple[MLExperimentDataset, MLCandleColumns, Instrument]] = {}
    for item in selected:
        family = item.family
        base_result: dict[str, object] = {
            "family_id": family.id,
            "deterministic_ordinal": family.deterministic_ordinal,
            "evaluation_id": item.evaluation.id,
            "evaluation_fingerprint": item.evaluation.fingerprint,
            "configuration": {
                "asset": family.asset,
                "provider": family.provider,
                "market_type": family.market_type,
                "interval": family.interval,
                "engine": item.evaluation.engine,
                "future_duration_seconds": item.evaluation.future_duration_seconds,
                "objective": item.evaluation.objective,
                "hyperparameters": item.evaluation.hyperparameters,
                "feature_policy": item.evaluation.feature_policy,
            },
            "confidence_fraction": item.confidence_fraction,
            "persisted_selection_metrics": {
                "observations": item.observations,
                "positive_net_folds": item.positive_net_folds,
                "favorable_direction_folds": item.favorable_direction_folds,
                "folds_beating_benchmark": item.folds_beating_benchmark,
                "median_net_return": item.median_net_return,
                "worst_net_return": item.worst_net_return,
            },
        }
        try:
            if family.source_screening_run_id not in cache:
                definition = resolve_instrument(
                    asset=Asset(family.asset),
                    provider=Provider(family.provider),
                    market_type=MarketType(family.market_type),
                    interval=Interval(family.interval),
                    capability=InstrumentCapability.SCORING,
                )
                candle_result = await read_ml_experiment_candles(
                    session=session,
                    instrument=definition.instrument,
                    start=DEVELOPMENT_START,
                    end=DEVELOPMENT_END,
                )
                if len(candle_result.columns) == 0:
                    raise DevelopmentReauditError(
                        "missing_source_market_data",
                        "source has no development candles",
                    )
                dataset = build_ml_experiment_dataset(
                    instrument=definition.instrument,
                    columns=candle_result.columns,
                    requested_end=DEVELOPMENT_END,
                )
                cache[family.source_screening_run_id] = (
                    dataset,
                    candle_result.columns,
                    definition.instrument,
                )
            dataset, candles, instrument = cache[family.source_screening_run_id]
            calculated = calculate_candidate_reaudit(
                dataset=dataset,
                candles=candles,
                instrument=instrument,
                evidence=item,
            )
            results.append({**base_result, **calculated})
        except DevelopmentReauditError:
            raise
        except (ValueError, KeyError, TypeError) as error:
            results.append(
                {
                    **base_result,
                    "status": "invalid",
                    "diagnostic_only": True,
                    "protected_evidence_read": False,
                    "invalid_code": "candidate_evidence_invalid",
                    "invalid_message": str(error)[:240],
                }
            )
    counts = {
        "eligible": len(eligible),
        "selected": len(selected),
        "omitted_by_cap": max(0, len(eligible) - len(selected)),
        "completed": sum(item["status"] == "completed" for item in results),
        "invalid": sum(item["status"] == "invalid" for item in results),
        "proxy_passed": sum(
            item.get("development_proxy_pass") is True for item in results
        ),
        "proxy_failed": sum(
            item["status"] == "completed"
            and item.get("development_proxy_pass") is False
            for item in results
        ),
        "malformed_unselected_stage_four": len(malformed),
    }
    report: dict[str, object] = {
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "diagnostic_only": True,
        "protected_evidence_read": False,
        "campaign": {
            "id": campaign.id,
            "experiment_key": campaign.experiment_key,
            "contract_version": campaign.contract_version,
            "policy_version": campaign.policy_version,
            "classification_version": campaign.classification_version,
            "preset": request.get("preset"),
            "sources": [
                {
                    "source_screening_run_id": source.source_screening_run_id,
                    "dataset_fingerprint": source.dataset_fingerprint,
                    "asset": screening[source.source_screening_run_id].asset,
                    "provider": screening[source.source_screening_run_id].provider,
                    "market_type": screening[
                        source.source_screening_run_id
                    ].market_type,
                    "interval": screening[source.source_screening_run_id].interval,
                }
                for source in sources
            ],
        },
        "development_range": {
            "start_inclusive": DEVELOPMENT_START,
            "end_exclusive": DEVELOPMENT_END,
        },
        "folds": ACCUMULATING_FOLD_NAMES,
        "cost_multipliers": ("1", "1.5", "2"),
        "selection_policy": normalized_selection_policy(),
        "selection_fingerprint": _fingerprint(selection_identity),
        "funnel": funnel,
        "counts": counts,
        "results": results,
    }
    report["report_fingerprint"] = _fingerprint(report)
    return report, counts["invalid"] > 0

