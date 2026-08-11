from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.campaign import economic_v2_calculation
from app.ml.campaign.contracts import (
    ECONOMIC_POLICY_SEARCH_V2_PRESET,
    ECONOMIC_V2_DIVERSIFIED_FAMILIES,
    CampaignRequest,
    FeatureVariant,
    campaign_search_spaces,
    seeded_local_search_space,
)
from app.ml.campaign.economic_v2_calculation import (
    aggregate_policy_folds,
    evaluate_policy_grid_fold,
    evaluate_strategy_policy,
)
from app.ml.campaign.economic_v2_planning import (
    SeedFamily,
    build_economic_v2_universe,
)
from app.ml.campaign.economic_v2_policy import (
    StrategyPolicy,
    enforce_v2_market_boundary,
    exact_holding_candles,
    failed_qualification_rules,
    strategy_policy_grid,
)
from app.ml.campaign.economic_v2_promotion import (
    EconomicV2Evidence,
    capped_economic_v2_promotions,
    is_economic_v2_development_candidate,
    passes_economic_v2_stage,
)
from app.ml.campaign.economic_v2_qualification import (
    qualification_temporal_indices,
)
from app.ml.campaign.planning import SourceInstrument, build_family_universe
from app.ml.dataset import MLExperimentDataset
from app.ml.gaps import datetime_to_ns
from app.ml.refinement.contracts import (
    EconomicPolicySearchConfiguration,
    PredictionObjective,
)
from app.ml.refinement.planning import planned_evaluation_from_persisted_values
from app.ml.screening.config import ScreeningEngine


def _source(asset: str = "btc_usdt") -> SourceInstrument:
    return SourceInstrument(
        "00000000-0000-0000-0000-000000000001",
        asset,
        "binance",
        "spot",
        "5m",
        "a" * 64,
    )


def test_persisted_evaluation_snapshot_is_deeply_session_independent() -> None:
    hyperparameters: dict[str, object] = {"alpha": 1.0, "fit_intercept": True}
    feature_policy: dict[str, object] = {
        "strategy": "explicit",
        "variables": ["return_1", "rsi_14"],
    }
    candidate = planned_evaluation_from_persisted_values(
        fingerprint="e" * 64,
        engine="ridge",
        future_duration_seconds=3600,
        objective="future_return_regression",
        hyperparameters=hyperparameters,
        feature_policy=feature_policy,
    )
    hyperparameters["alpha"] = 100.0
    assert isinstance(feature_policy["variables"], list)
    feature_policy["variables"].append("atr_14")
    assert candidate.hyperparameters == (("alpha", 1.0), ("fit_intercept", True))
    assert candidate.explicit_variables == ("return_1", "rsi_14")


@pytest.mark.parametrize(
    ("hyperparameters", "feature_policy", "message"),
    (
        ({"alpha": "invalid"}, {"strategy": "all"}, "hyperparameter"),
        ({"alpha": 1.0}, {"strategy": "explicit", "variables": "rsi_14"}, "variables"),
        ({"alpha": 1.0}, {"strategy": "top_k", "top_k": True}, "top_k"),
    ),
)
def test_persisted_evaluation_snapshot_rejects_malformed_values(
    hyperparameters: dict[str, object],
    feature_policy: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        planned_evaluation_from_persisted_values(
            fingerprint="e" * 64,
            engine="ridge",
            future_duration_seconds=3600,
            objective="future_return_regression",
            hyperparameters=hyperparameters,
            feature_policy=feature_policy,
        )


def _evidence(**overrides: object) -> EconomicV2Evidence:
    values: dict[str, object] = {
        "family_fingerprint": "a" * 64,
        "policy_fingerprint": "b" * 64,
        "source_run_id": "source",
        "engine": "ridge",
        "objective": "future_return_regression",
        "observations": 200,
        "round_trips": 30,
        "positive_net_folds": 6,
        "folds_beating_benchmark": 5,
        "median_net_return": Decimal("0.01"),
        "worst_net_return": Decimal("-0.01"),
        "maximum_drawdown": Decimal("0.10"),
        "inactive_folds": 0,
        "compounded_return_1_5x": Decimal("0.05"),
        "compounded_return_2x": Decimal("0.02"),
        "complete_and_finite": True,
    }
    values.update(overrides)
    return EconomicV2Evidence(**values)  # type: ignore[arg-type]


def test_legacy_request_and_first_family_fingerprints_are_frozen() -> None:
    source_id = UUID("00000000-0000-0000-0000-000000000001")
    request = CampaignRequest(
        source_screening_run_ids=(source_id,),
        maximum_evaluations=3600,
        maximum_runtime_seconds=432000,
    )
    assert request.fingerprint() == (
        "e0cb92d871c266a32304f14f9f5f2b3ff92e8874b0d37197e805b64e6b20f300"
    )
    universe = build_family_universe((_source(),))
    assert len(universe) == 3600
    assert universe[0].fingerprint == (
        "fe94da440cc68a0f35007324b35da5e185dd7a1811f4cb49f64ceef639759c2e"
    )


def test_v2_diversified_universe_is_exact_unique_and_deterministic() -> None:
    first = build_economic_v2_universe((_source(),), ())
    second = build_economic_v2_universe((_source(),), ())
    assert len(first) == ECONOMIC_V2_DIVERSIFIED_FAMILIES == 432
    assert len({item.fingerprint for item in first}) == 432
    assert [item.fingerprint for item in first] == [item.fingerprint for item in second]
    assert {item.engine for item in first} == set(ScreeningEngine)
    assert {item.objective for item in first} == set(PredictionObjective)
    assert {item.feature_variant.value for item in first} == {
        "all",
        "top_k_16",
        "top_k_32",
        "top_k_64",
    }


def test_seed_offsets_are_bounded_deduplicated_and_record_provenance() -> None:
    seed = SeedFamily(
        uuid4(),
        _source().source_run_id,
        ScreeningEngine.RIDGE,
        7 * 3600,
        PredictionObjective.FUTURE_RETURN_REGRESSION,
        # This combination exists in the diversified track at some horizons.
        next(
            item.feature_variant
            for item in build_economic_v2_universe((_source(),), ())
        ),
        "e" * 64,
        (("alpha", 1.0), ("fit_intercept", True)),
    )
    universe = build_economic_v2_universe((_source(),), (seed,))
    assert len(universe) <= 600
    assert len({item.fingerprint for item in universe}) == len(universe)
    assert len({item.scientific_identity for item in universe}) == len(universe)
    assert any(seed.id in item.also_seeded_by for item in universe)
    assert all(
        3600 <= item.future_duration_seconds <= 7 * 24 * 3600 for item in universe
    )


def test_two_48_hour_seeds_add_only_nonduplicate_local_horizons() -> None:
    source = _source()
    seeds = (
        SeedFamily(
            uuid4(),
            source.source_run_id,
            ScreeningEngine.RIDGE,
            48 * 3600,
            PredictionObjective.FUTURE_RETURN_REGRESSION,
            FeatureVariant.TOP_64,
            "e" * 64,
            (("alpha", 1.0), ("fit_intercept", True)),
        ),
        SeedFamily(
            uuid4(),
            source.source_run_id,
            ScreeningEngine.CATBOOST,
            48 * 3600,
            PredictionObjective.FUTURE_RETURN_REGRESSION,
            FeatureVariant.TOP_16,
            "f" * 64,
            (),
        ),
    )

    universe = build_economic_v2_universe((source,), seeds)

    assert len(universe) == 438
    assert sum(item.search_track == "seeded_local" for item in universe) == 6
    assert len({item.fingerprint for item in universe}) == 438
    assert len({item.scientific_identity for item in universe}) == 438
    for seed in seeds:
        seeded = tuple(item for item in universe if seed.id in item.also_seeded_by)
        assert len(seeded) == 5
        assert {item.future_duration_seconds // 3600 for item in seeded} == {
            36,
            42,
            48,
            54,
            60,
        }
        assert {
            item.future_duration_seconds // 3600
            for item in seeded
            if item.search_track == "diversified"
        } == {36, 48}
        assert {
            item.future_duration_seconds // 3600
            for item in seeded
            if item.search_track == "seeded_local"
        } == {42, 54, 60}
        assert all(item.engine is seed.engine for item in seeded)
        assert all(item.objective is seed.objective for item in seeded)
        assert all(item.feature_variant is seed.feature_variant for item in seeded)
        assert all(item.source == source for item in seeded)


@pytest.mark.parametrize("engine", tuple(ScreeningEngine))
def test_seeded_local_space_uses_exact_seed_and_canonical_neighbours(
    engine: ScreeningEngine,
) -> None:
    canonical = next(item for item in campaign_search_spaces() if item.engine is engine)
    seed = {
        name: getattr(canonical, name).expand()[
            len(getattr(canonical, name).expand()) // 2
        ]
        for name in type(canonical).model_fields
        if name != "engine"
    }
    local = seeded_local_search_space(engine=engine, hyperparameters=seed)
    assert local.engine is engine
    full_size = 1
    local_size = 1
    for name, seed_value in seed.items():
        full = getattr(canonical, name).expand()
        values = getattr(local, name).expand()
        assert seed_value in values
        assert set(values).issubset(full)
        assert len(values) <= 3
        full_size *= len(full)
        local_size *= len(values)
    assert local_size < full_size
    assert local.model_dump(mode="json") == seeded_local_search_space(
        engine=engine, hyperparameters=seed
    ).model_dump(mode="json")


@pytest.mark.parametrize("engine", tuple(ScreeningEngine))
def test_seeded_local_space_clamps_at_canonical_edges(engine: ScreeningEngine) -> None:
    canonical = next(item for item in campaign_search_spaces() if item.engine is engine)
    seed = {
        name: getattr(canonical, name).expand()[0]
        for name in type(canonical).model_fields
        if name != "engine"
    }
    local = seeded_local_search_space(engine=engine, hyperparameters=seed)
    assert all(
        len(getattr(local, name).expand()) <= 2
        for name in type(local).model_fields
        if name != "engine"
    )


@pytest.mark.parametrize("horizon_seconds", (3600, 72 * 3600))
def test_qualification_temporal_selection_purges_cross_boundary_targets(
    horizon_seconds: int,
) -> None:
    boundaries = tuple(
        datetime_to_ns(value)
        for value in (
            datetime(2023, 12, 20, tzinfo=UTC),
            datetime(2023, 12, 24, 23, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 12, 31, 23, tzinfo=UTC),
        )
    )
    dataset = cast(
        MLExperimentDataset,
        SimpleNamespace(
            boundary_time_ns=np.asarray(boundaries, dtype=np.int64),
            feature_eligible=np.ones(len(boundaries), dtype=np.bool_),
        ),
    )
    pre_purge, development, qualification, targets = qualification_temporal_indices(
        dataset=dataset,
        target_eligible=np.ones(len(boundaries), dtype=np.bool_),
        horizon_seconds=horizon_seconds,
    )
    assert tuple(map(int, pre_purge)) == (0, 1)
    assert tuple(map(int, development)) == (0,)
    assert tuple(map(int, qualification)) == (2,)
    assert all(
        int(targets[index]) < datetime_to_ns(datetime(2024, 1, 1, tzinfo=UTC))
        for index in development
    )
    assert all(
        int(targets[index]) < datetime_to_ns(datetime(2025, 1, 1, tzinfo=UTC))
        for index in qualification
    )


def test_v2_request_is_discriminated_and_requires_exact_stage_one_budget() -> None:
    source_id = uuid4()
    with pytest.raises(ValidationError, match="exact minimum is 432"):
        CampaignRequest(
            source_screening_run_ids=(source_id,),
            preset=ECONOMIC_POLICY_SEARCH_V2_PRESET,
            maximum_evaluations=431,
        )
    request = CampaignRequest(
        source_screening_run_ids=(source_id,),
        preset=ECONOMIC_POLICY_SEARCH_V2_PRESET,
        maximum_evaluations=432,
    )
    assert request.normalized_payload()["preset"] == ECONOMIC_POLICY_SEARCH_V2_PRESET


def test_policy_grid_has_45_sensitive_deterministic_identities() -> None:
    policies = strategy_policy_grid(4 * 3600)
    assert len(policies) == 45
    assert len({item.fingerprint for item in policies}) == 45
    assert policies == strategy_policy_grid(4 * 3600)
    assert {item.holding_candles for item in policies} == {24, 48, 72}
    assert exact_holding_candles(3600, Decimal("0.5")) == 6
    with pytest.raises(ValueError, match="exact five-minute candles"):
        exact_holding_candles(301, Decimal("0.5"))


def test_policy_fold_evaluates_exact_configured_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = EconomicPolicySearchConfiguration(
        confidence_fractions=(Decimal("0.20"), Decimal("0.05")),
        holding_multipliers=(Decimal("1.5"), Decimal("0.5")),
        regime_policies=("trend_aligned", "all_regimes"),
        version=7,
    )
    seen: list[StrategyPolicy] = []

    def capture(**kwargs: object) -> dict[str, object]:
        policy = cast(StrategyPolicy, kwargs["policy"])
        seen.append(policy)
        return {"policy_fingerprint": policy.fingerprint}

    monkeypatch.setattr(economic_v2_calculation, "evaluate_strategy_policy", capture)
    rows = np.asarray([0], dtype=np.int64)
    result = evaluate_policy_grid_fold(
        dataset=cast(MLExperimentDataset, object()),
        training=rows,
        validation=rows,
        training_scores=np.asarray([0.1]),
        validation_scores=np.asarray([0.1]),
        horizon_seconds=3600,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        policy_configuration=configuration,
    )

    assert len(result) == len(seen) == 8
    assert tuple(seen) == strategy_policy_grid(3600, configuration)
    assert {item.version for item in seen} == {7}


@pytest.mark.parametrize("stage", (1, 2, 3))
def test_v2_stage_gates_are_objective_independent(stage: int) -> None:
    for objective in PredictionObjective:
        assert passes_economic_v2_stage(stage, _evidence(objective=objective.value))


def test_v2_development_gate_and_objective_diversity() -> None:
    assert is_economic_v2_development_candidate(_evidence())
    items = tuple(
        _evidence(
            family_fingerprint=f"{objective_index * 4 + index:064x}",
            policy_fingerprint=f"{objective_index * 4 + index + 100:064x}",
            objective=objective.value,
        )
        for objective_index, objective in enumerate(PredictionObjective)
        for index in range(4)
    )
    promoted = capped_economic_v2_promotions(1, items)
    assert {item.objective for item in promoted} == {
        item.value for item in PredictionObjective
    }


def test_qualification_rules_and_2025_boundary_guard() -> None:
    assert not failed_qualification_rules(
        round_trips=20,
        base_return=Decimal("0.05"),
        return_1_5x=Decimal("0.0001"),
        return_2x=Decimal("0.0001"),
        maximum_drawdown=Decimal("0.20"),
        quarter_returns=(
            Decimal("0.01"),
            Decimal("0.01"),
            Decimal("0.01"),
            Decimal("-0.05"),
        ),
    )
    enforce_v2_market_boundary(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="cannot read 2025"):
        enforce_v2_market_boundary(
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
        )


class _Targets:
    def __init__(self, values: np.ndarray[tuple[int], np.dtype[np.float32]]) -> None:
        self.values = values

    def for_duration(self, _duration: object) -> object:
        return SimpleNamespace(
            values=self.values,
            eligible=np.ones(len(self.values), dtype=np.bool_),
        )


def _policy_dataset(validation_volatility: float = 0.5) -> MLExperimentDataset:
    count = 12
    context = SimpleNamespace(
        close=np.full(count, 101.0),
        ema_200=np.full(count, 100.0),
        realized_volatility_12h=np.asarray(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                validation_volatility,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
        ),
    )
    return cast(
        MLExperimentDataset,
        SimpleNamespace(
            metadata=SimpleNamespace(
                instrument=Instrument(
                    Asset.BTC_USDT,
                    Provider.BINANCE,
                    MarketType.SPOT,
                    Interval.FIVE_MINUTES,
                )
            ),
            boundary_time_ns=np.arange(count, dtype=np.int64) * 30 * 60 * 1_000_000_000,
            market_context=context,
            targets=_Targets(np.full(count, 0.01, dtype=np.float32)),
        ),
    )


def test_policy_cutoff_and_volatility_bounds_are_training_only() -> None:
    policy = next(
        item
        for item in strategy_policy_grid(3600)
        if item.confidence_fraction == Decimal("0.40")
        and item.holding_multiplier == Decimal("0.5")
        and item.regime_policy == "volatility_middle_60"
    )
    training = np.arange(5, dtype=np.int64)
    validation = np.arange(5, 12, dtype=np.int64)
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.9])
    validation_scores = np.full(7, 0.8)
    first = evaluate_strategy_policy(
        dataset=_policy_dataset(0.5),
        training=training,
        validation=validation,
        training_scores=scores,
        validation_scores=validation_scores,
        horizon_seconds=3600,
        policy=policy,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    )
    mutated = evaluate_strategy_policy(
        dataset=_policy_dataset(99.0),
        training=training,
        validation=validation,
        training_scores=scores,
        validation_scores=validation_scores,
        horizon_seconds=3600,
        policy=policy,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    )
    assert first["confidence_cutoff"] == mutated["confidence_cutoff"]
    assert first["volatility_lower"] == mutated["volatility_lower"]
    assert first["volatility_upper"] == mutated["volatility_upper"]


def test_trend_policy_requires_finite_preceding_ema_and_strict_alignment() -> None:
    policy = next(
        item
        for item in strategy_policy_grid(3600)
        if item.confidence_fraction == Decimal("0.40")
        and item.holding_multiplier == Decimal("0.5")
        and item.regime_policy == "trend_aligned"
    )
    dataset = _policy_dataset()
    dataset.market_context.ema_200[5] = np.nan
    dataset.market_context.close[6] = dataset.market_context.ema_200[6]
    result = evaluate_strategy_policy(
        dataset=dataset,
        training=np.arange(5, dtype=np.int64),
        validation=np.arange(5, 12, dtype=np.int64),
        training_scores=np.asarray([0.1, 0.2, 0.3, 0.4, 0.9]),
        validation_scores=np.full(7, 0.8),
        horizon_seconds=3600,
        policy=policy,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    )
    assert result["eligible_observations"] == 5


def test_policy_aggregation_omits_duplicated_fold_payloads() -> None:
    fold = {
        "policy_fingerprint": "p" * 64,
        "strategy_policy_version": 2,
        "confidence_fraction": "0.10",
        "holding_multiplier": "1",
        "holding_candles": 12,
        "regime_policy": "all_regimes",
        "eligible_observations": 10,
        "round_trips": 2,
        "net_return": "0.01",
        "buy_and_hold_return": "0.00",
        "maximum_drawdown": "0.02",
        "return_1_5x": "0.01",
        "return_2x": "0.01",
        "trades": [{"payload": "x" * 100_000}],
        "equity_curve": [["2039-01-01T00:00:00+00:00", "1"]],
    }
    result = aggregate_policy_folds(((fold,),))
    assert len(result) == 1
    assert "folds" not in result[0]
    assert result[0]["policy_fingerprint"] == "p" * 64
    assert result[0]["round_trips"] == 2
