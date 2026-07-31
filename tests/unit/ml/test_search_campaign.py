from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ml.campaign.contracts import (
    CAMPAIGN_CONFIDENCE_BUCKETS,
    CAMPAIGN_FEE_RATE,
    CAMPAIGN_SEED,
    CAMPAIGN_SLIPPAGE_RATE,
    FAMILIES_PER_INSTRUMENT,
    STAGE_TRIALS,
    CampaignRequest,
    FeatureVariant,
    campaign_search_spaces,
)
from app.ml.campaign.planning import SourceInstrument, build_family_universe
from app.ml.campaign.promotion import (
    EconomicEvidence,
    capped_promotions,
    is_strong_candidate,
    passes_stage_gate,
)
from app.ml.contract import FUTURE_DURATIONS
from app.ml.refinement.contracts import PredictionObjective, RidgeSearchSpace
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

