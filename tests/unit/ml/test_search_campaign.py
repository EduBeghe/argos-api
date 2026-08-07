from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ml.campaign.contracts import (
    CAMPAIGN_CONFIDENCE_BUCKETS,
    CAMPAIGN_FEE_RATE,
    CAMPAIGN_SEED,
    CAMPAIGN_SLIPPAGE_RATE,
    ECONOMIC_POLICY_SEARCH_V2_PRESET,
    ECONOMIC_V2_CONFIDENCE_FRACTIONS,
    FAMILIES_PER_INSTRUMENT,
    STAGE_TRIALS,
    CampaignRequest,
    FeatureVariant,
    campaign_search_spaces,
)
from app.ml.campaign.coordinator import _request_for_family, select_economic_evidence
from app.ml.campaign.economic_v2_policy import strategy_policy_grid
from app.ml.campaign.models import MLSearchCampaignFamilyRecord
from app.ml.campaign.planning import SourceInstrument, build_family_universe
from app.ml.campaign.promotion import (
    EconomicEvidence,
    capped_promotions,
    economic_rank,
    is_strong_candidate,
    passes_stage_gate,
)
from app.ml.contract import FUTURE_DURATIONS
from app.ml.refinement.contracts import (
    MLRefinementRequest,
    PredictionObjective,
    RidgeSearchSpace,
)
from app.ml.refinement.models import MLRefinementEvaluationRecord
from app.ml.screening.config import ScreeningEngine


def source(name: str, fingerprint: str) -> SourceInstrument:
    return SourceInstrument(str(uuid4()), name, "binance", "spot", "5m", fingerprint)


def evidence(**overrides: object) -> EconomicEvidence:
    values: dict[str, object] = {
        "family_fingerprint": "a" * 64,
        "source_run_id": "source",
        "engine": "ridge",
        "observations": 100,
        "favorable_direction_folds": 5,
        "positive_net_folds": 5,
        "folds_beating_benchmark": 5,
        "median_net_return": Decimal("0.001"),
        "worst_net_return": Decimal("-0.0015"),
        "promising_configurations": 3,
        "complete_and_finite": True,
    }
    values.update(overrides)
    return EconomicEvidence(**values)  # type: ignore[arg-type]


def evaluation(
    fingerprint: str,
    *,
    median: str | None,
    worst: str | None,
    positive_folds: int = 3,
) -> MLRefinementEvaluationRecord:
    selected: dict[str, object] = {
        "observation_count": 100,
        "folds_with_favorable_direction": 4,
        "folds_with_positive_net_return": positive_folds,
        "median_net_return": median,
        "worst_net_return": worst,
    }
    return MLRefinementEvaluationRecord(
        id=uuid4(),
        run_id=uuid4(),
        reused_from_evaluation_id=None,
        fingerprint=fingerprint,
        engine="ridge",
        future_duration_seconds=1800,
        objective="regression",
        hyperparameters={},
        feature_policy={},
        confidence_buckets=[],
        fold_evidence=[],
        aggregate_metrics={
            "selected_confidence_bucket": selected,
            "folds_beating_benchmark": 3,
            "complete_fold_count": 7,
        },
        classification="promising_for_strategy_simulation",
        classification_version=1,
        failed_rules=[],
        elapsed_seconds=Decimal("1"),
        created_at=datetime.now(UTC),
    )


def family() -> MLSearchCampaignFamilyRecord:
    return MLSearchCampaignFamilyRecord(
        id=uuid4(),
        campaign_id=uuid4(),
        source_screening_run_id=uuid4(),
        fingerprint="f" * 64,
        dataset_fingerprint="d" * 64,
        asset="btc_usdt",
        provider="binance",
        market_type="spot",
        interval="5m",
        engine="ridge",
        future_duration_seconds=1800,
        objective="regression",
        feature_variant="all",
        state="scheduled",
        current_stage=0,
        evaluations_planned=1,
        evaluations_completed=0,
        evaluations_reused=0,
        best_evaluation_id=None,
        best_metrics=None,
        decision_reason=None,
        deterministic_ordinal=0,
        created_at=datetime.now(UTC),
    )


def test_contract_contains_exact_policy_values() -> None:
    assert len(FUTURE_DURATIONS) == 80
    assert FAMILIES_PER_INSTRUMENT == 3 * 80 * 3 * 5
    assert STAGE_TRIALS == (1, 4, 16, 64)
    assert (
        Decimal("0.05"),
        Decimal("0.10"),
        Decimal("0.20"),
    ) == CAMPAIGN_CONFIDENCE_BUCKETS
    assert (
        Decimal("0.001"),
        Decimal("0.0005"),
        1729,
    ) == (CAMPAIGN_FEE_RATE, CAMPAIGN_SLIPPAGE_RATE, CAMPAIGN_SEED)
    spaces = {item.engine: item for item in campaign_search_spaces()}
    ridge = spaces[ScreeningEngine.RIDGE]
    assert isinstance(ridge, RidgeSearchSpace)
    assert ridge.alpha.expand() == (0.01, 0.1, 1, 10, 100)
    assert ridge.fit_intercept.expand() == (False, True)


@pytest.mark.asyncio
async def test_v2_child_separates_legacy_buckets_from_economic_fractions() -> None:
    family_record = family()
    family_record.preset = ECONOMIC_POLICY_SEARCH_V2_PRESET
    family_record.search_track = "diversified"
    family_record.future_duration_seconds = 3600
    family_record.objective = PredictionObjective.FUTURE_RETURN_REGRESSION.value
    session = AsyncMock()
    session.scalar.return_value = type("SourceResult", (), {"id": uuid4()})()

    request = await _request_for_family(session, family_record, 1, 60)

    assert request.confidence_buckets == CAMPAIGN_CONFIDENCE_BUCKETS
    assert request.economic_policy_search is not None
    assert (
        request.economic_policy_search.confidence_fractions
        == ECONOMIC_V2_CONFIDENCE_FRACTIONS
    )
    assert len(strategy_policy_grid(family_record.future_duration_seconds)) == 45
    assert MLRefinementRequest.model_validate(
        request.normalized_payload()
    ) == request.model_copy(update={"notes": None})


def test_universe_is_exact_unique_deterministic_and_interleaved() -> None:
    first = source("btc_usdt", "a" * 64)
    second = source("eth_usdt", "b" * 64)
    plan = build_family_universe((second, first))
    repeated = build_family_universe((first, second))
    assert len(plan) == 2 * FAMILIES_PER_INSTRUMENT
    assert len({item.fingerprint for item in plan}) == len(plan)
    assert [item.fingerprint for item in plan] == [
        item.fingerprint for item in repeated
    ]
    assert [item.source.asset for item in plan[:6]] == [
        "btc_usdt",
        "eth_usdt",
        "btc_usdt",
        "eth_usdt",
        "btc_usdt",
        "eth_usdt",
    ]
    assert {item.engine for item in plan} == set(ScreeningEngine)
    assert {item.future_duration_seconds for item in plan} == {
        int(item.total_seconds()) for item in FUTURE_DURATIONS
    }
    assert {item.objective for item in plan} == set(PredictionObjective)
    assert {item.feature_variant for item in plan} == set(FeatureVariant)


def test_request_requires_exact_stage_one_budget_and_unique_sources() -> None:
    identifier = uuid4()
    with pytest.raises(ValidationError, match="exact minimum is 3600"):
        CampaignRequest(
            source_screening_run_ids=(identifier,), maximum_evaluations=3599
        )
    with pytest.raises(ValidationError, match="must be unique"):
        CampaignRequest(
            source_screening_run_ids=(identifier, identifier), maximum_evaluations=7200
        )


@pytest.mark.parametrize(
    ("stage", "change", "expected"),
    [
        (1, {"observations": 29}, False),
        (1, {"positive_net_folds": 1, "median_net_return": Decimal("-0.0014")}, True),
        (2, {"folds_beating_benchmark": 2}, False),
        (2, {"median_net_return": Decimal("-0.0004")}, True),
        (3, {"promising_configurations": 0}, False),
        (3, {"median_net_return": Decimal("0.0001")}, True),
    ],
)
def test_stage_gate_boundaries(
    stage: int, change: dict[str, object], expected: bool
) -> None:
    assert passes_stage_gate(stage, evidence(**change)) is expected


def test_promotion_cap_is_rounded_up_and_ranked_economically() -> None:
    candidates = tuple(
        evidence(
            family_fingerprint=f"{index:064x}", median_net_return=Decimal(index) / 1000
        )
        for index in range(5)
    )
    promoted = capped_promotions(1, candidates)
    assert len(promoted) == 2
    assert [item.median_net_return for item in promoted] == [
        Decimal("0.004"),
        Decimal("0.003"),
    ]


def test_strong_gate_requires_every_economic_requirement() -> None:
    assert is_strong_candidate(evidence())
    assert not is_strong_candidate(evidence(promising_configurations=2))
    assert not is_strong_candidate(evidence(median_net_return=Decimal("0.000999")))
    assert not is_strong_candidate(evidence(worst_net_return=Decimal("-0.001501")))
    assert not is_strong_candidate(evidence(complete_and_finite=False))


def test_economic_rank_preserves_zero_as_real_evidence() -> None:
    candidates = (
        evidence(family_fingerprint="n" * 64, median_net_return=None),
        evidence(family_fingerprint="m" * 64, median_net_return=Decimal("-0.1")),
        evidence(family_fingerprint="z" * 64, median_net_return=Decimal("0")),
        evidence(family_fingerprint="p" * 64, median_net_return=Decimal("0.1")),
    )

    assert [
        item.family_fingerprint[0] for item in sorted(candidates, key=economic_rank)
    ] == [
        "p",
        "z",
        "m",
        "n",
    ]


def test_selected_evaluation_and_metrics_come_from_the_same_best_row() -> None:
    missing = evaluation("b" * 64, median=None, worst=None)
    zero = evaluation("a" * 64, median="0", worst="0")

    selected = select_economic_evidence(family(), (missing, zero))

    assert selected.evaluation.id == zero.id
    assert selected.evidence.median_net_return == Decimal("0")
    assert selected.evidence.worst_net_return == Decimal("0")


def test_selected_evidence_ties_are_broken_by_fingerprint() -> None:
    later = evaluation("b" * 64, median="0.01", worst="-0.01")
    earlier = evaluation("a" * 64, median="0.01", worst="-0.01")

    selected = select_economic_evidence(family(), (later, earlier))

    assert selected.evaluation.id == earlier.id
