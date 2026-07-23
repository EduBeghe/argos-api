import argparse
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

import app.optimization.weekly_research as research_module
from app.jobs.run_weekly_strategy_research import run as run_job
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.edge_guided import (
    EdgeGuidedEvaluationResult,
    ResearchOutcome,
)
from app.optimization.edge_guided_memory import EdgeGuidedExecution
from app.optimization.feature_matrix import HistoricalFeatureMatrix
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    HistoricalDatasetRevision,
)
from app.optimization.predictive_edge_engine_v41 import V41StageCounters
from app.optimization.predictive_edges import PredictiveEdgeConfiguration
from app.optimization.targeted_refinement import TargetedRefinementConclusion
from app.optimization.targeted_refinement_schemas import (
    TargetedRefinementSummaryResponse,
)
from app.optimization.weekly_research import (
    execute_weekly_instrument_research,
    weekly_boundary,
    weekly_research_reuse_key,
)
from app.paper.config import SimulationConfiguration

INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)
IDENTITY = HistoricalDatasetIdentity(INSTRUMENT, HistoricalDatasetRange(START, END))
REVISION = HistoricalDatasetRevision(
    100,
    START,
    END - timedelta(minutes=5),
    END,
    "revision",
)


def test_weekly_boundary_is_deterministic_across_hour_and_day_rollover() -> None:
    expected = datetime(2026, 1, 4, 3, 17, tzinfo=UTC)
    assert weekly_boundary(datetime(2026, 1, 4, 3, 17, tzinfo=UTC)) == expected
    assert weekly_boundary(datetime(2026, 1, 10, 23, 59, tzinfo=UTC)) == expected
    assert weekly_boundary(datetime(2026, 1, 4, 3, 16, tzinfo=UTC)) == datetime(
        2025, 12, 28, 3, 17, tzinfo=UTC
    )


def test_reuse_identity_is_deterministic_and_includes_effective_costs() -> None:
    def key(fee_rate: str) -> str:
        return weekly_research_reuse_key(
            identity=IDENTITY,
            revision=REVISION,
            configuration=PredictiveEdgeConfiguration(),
            simulation=SimulationConfiguration(fee_rate=Decimal(fee_rate)),
            initial_cash=Decimal("10000"),
            maximum_candidates=12,
            boundary=datetime(2026, 1, 4, 3, 17, tzinfo=UTC),
        )

    first = key("0.001")
    assert first == key("0.001")
    assert first != key("0.002")


@pytest.mark.asyncio
async def test_exact_reuse_short_circuits_every_heavy_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = EdgeGuidedEvaluationResult(
        "dataset",
        1,
        1,
        12,
        3,
        (),
        ResearchOutcome.CANDIDATES_REJECTED_VALIDATION,
        "reused",
        {"edges_evaluated": 19, "unstable_edges": 19},
    )
    predictive_run_id = uuid4()
    execution = EdgeGuidedExecution(
        None, "key", True, result, 1, 0, 22, predictive_run_id
    )
    revision = AsyncMock(return_value=REVISION)
    reusable = AsyncMock(return_value=execution)
    dataset_loader = AsyncMock(side_effect=AssertionError("dataset loaded"))
    matrix_builder = Mock(side_effect=AssertionError("matrix built"))
    edge_analyzer = Mock(side_effect=AssertionError("edge analysis executed"))
    regime_analyzer = Mock(side_effect=AssertionError("v4.1 analysis executed"))
    targeted_analyzer = Mock(side_effect=AssertionError("targeted analysis executed"))
    candidate_evaluator = Mock(side_effect=AssertionError("candidate simulated"))
    diagnostics = AsyncMock(
        return_value=SimpleNamespace(
            diagnostic_conclusion=SimpleNamespace(value="completed"),
            regime_conditioned_conclusion="no_regime_conditioned_edge",
            regime_conditioned_stage_counters={"a1": {"computed": 24}},
            regime_conditioned_summary={
                "actionable_near_miss_count": 3,
                "diagnostic_near_miss_count": 5,
            },
            computed_counts={"training": 100},
            model_dump=lambda **_: {
                "diagnostic_conclusion": "no_regime_conditioned_edge",
                "computed_counts": {"training": 100},
            },
        )
    )
    targeted_run_id = uuid4()
    targeted_candidate_id = uuid4()
    targeted_reader = AsyncMock(
        return_value=TargetedRefinementSummaryResponse(
            id=targeted_run_id,
            predictive_edge_run_id=predictive_run_id,
            conclusion=TargetedRefinementConclusion.CONFIRMED,
            summary={
                "evaluations_computed": 17,
                "confirmed_candidates": 1,
            },
            confirmed_candidate_count=1,
            best_targeted_candidate_id=targeted_candidate_id,
            best_targeted_candidate="best-fingerprint",
        )
    )
    monkeypatch.setattr(research_module, "read_historical_dataset_revision", revision)
    monkeypatch.setattr(research_module, "find_reusable_weekly_research_run", reusable)
    monkeypatch.setattr(research_module, "read_historical_dataset", dataset_loader)
    monkeypatch.setattr(research_module, "get_predictive_edge_diagnostics", diagnostics)
    monkeypatch.setattr(research_module, "get_targeted_refinement", targeted_reader)
    monkeypatch.setattr(
        research_module, "build_historical_feature_matrix", matrix_builder
    )
    monkeypatch.setattr(research_module, "analyze_predictive_edges_v4", edge_analyzer)
    monkeypatch.setattr(
        research_module, "analyze_predictive_edges_v41", regime_analyzer
    )
    monkeypatch.setattr(
        research_module, "evaluate_targeted_refinement", targeted_analyzer
    )
    monkeypatch.setattr(
        research_module, "evaluate_edge_guided_candidates", candidate_evaluator
    )

    session = SimpleNamespace()
    completed = await execute_weekly_instrument_research(
        session=session,  # type: ignore[arg-type]
        identity=IDENTITY,
        configuration=PredictiveEdgeConfiguration(),
        simulation=SimulationConfiguration(),
        initial_cash=Decimal("10000"),
        maximum_candidates=12,
        boundary=datetime(2026, 1, 4, 3, 17, tzinfo=UTC),
        persist=True,
    )

    assert completed.reused is True
    assert completed.execution.result.funnel["unstable_edges"] == 19
    assert completed.work.evaluations_computed == 0
    assert completed.work.evaluations_reused == 1
    assert completed.work.simulations_avoided == 163
    assert completed.diagnostics["actionable_near_misses"] == 3
    assert completed.diagnostics["diagnostic_near_misses"] == 5
    assert completed.predictive_edge_run_id == predictive_run_id
    assert completed.diagnostics["targeted_refinement_run_id"] == str(targeted_run_id)
    assert completed.diagnostics["targeted_refinement_conclusion"] == (
        TargetedRefinementConclusion.CONFIRMED.value
    )
    assert completed.diagnostics["targeted_refinement_summary"] == {
        "evaluations_computed": 17,
        "confirmed_candidates": 1,
    }
    assert completed.diagnostics["best_targeted_candidate"] == "best-fingerprint"
    assert completed.diagnostics["best_targeted_candidate_id"] == str(
        targeted_candidate_id
    )
    assert completed.diagnostics["targeted_current_work"] == {
        "evaluations_computed": 0,
        "evaluations_reused": 0,
    }
    assert completed.diagnostics["computed_counts"] == {"training": 100}
    assert completed.diagnostics["predictive_edge_v4_conclusion"] == "completed"
    assert (
        completed.diagnostics["regime_conditioned_v41_conclusion"]
        == "no_regime_conditioned_edge"
    )
    diagnostics.assert_awaited_once_with(session, predictive_run_id)
    targeted_reader.assert_awaited_once_with(session, predictive_run_id)
    dataset_loader.assert_not_awaited()
    matrix_builder.assert_not_called()
    edge_analyzer.assert_not_called()
    regime_analyzer.assert_not_called()
    targeted_analyzer.assert_not_called()
    candidate_evaluator.assert_not_called()


@pytest.mark.asyncio
async def test_weekly_dry_run_is_catalog_driven_and_does_not_start_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyze = AsyncMock(side_effect=AssertionError("analysis started"))
    monkeypatch.setattr(
        "app.jobs.run_weekly_strategy_research.execute_weekly_instrument_research",
        analyze,
    )
    args = argparse.Namespace(
        asset=None,
        lookback_days=730,
        maximum_candidates=12,
        force_recompute=False,
        dry_run=True,
    )

    summary = await run_job(args, now=datetime(2026, 1, 4, 3, 17, tzinfo=UTC))

    assert summary["status"] == "validated"
    assert summary["instruments"] == [
        ["BTC/USDT", "binance", "spot", "5m"],
        ["ETH/USDT", "binance", "spot", "5m"],
    ]
    analyze.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_v41_hypothesis_reaches_edge_guided_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = V41StageCounters(generated=1, computed=1, reused=0, retained=1)
    v4_hypothesis = SimpleNamespace(hypothesis_id="v4")
    v41_hypothesis = SimpleNamespace(hypothesis_id="v41")
    v4 = SimpleNamespace(
        hypotheses=(v4_hypothesis,),
        training_counters=counter,
        validation_counters=counter,
        test_counters=counter,
        confirmed_edges=(),
        diagnostic_only_edges=(),
        near_misses=(),
        rejection_counts={},
        bottleneck_stage="test",
        strongest_observed_signal=None,
        conclusion=SimpleNamespace(value="completed"),
        outcome_explanation="v4",
    )
    v41 = SimpleNamespace(
        hypotheses=(v41_hypothesis,),
        stage_a1=counter,
        stage_a2=counter,
        stage_a3=counter,
        stage_b=counter,
        stage_c=counter,
        confirmed_candidates=(object(),),
        actionable_near_misses=(),
        diagnostic_near_misses=(),
        conclusion=SimpleNamespace(value="regime_conditioned_edge_confirmed"),
    )
    generated = Mock(return_value=())
    evaluated = Mock(
        return_value=EdgeGuidedEvaluationResult(
            "dataset",
            0,
            0,
            12,
            3,
            (),
            ResearchOutcome.NO_PREDICTIVE_EDGE,
            "evaluated",
            {},
        )
    )
    monkeypatch.setattr(
        research_module,
        "read_historical_dataset_revision",
        AsyncMock(return_value=REVISION),
    )
    monkeypatch.setattr(
        research_module,
        "read_historical_dataset",
        AsyncMock(
            return_value=SimpleNamespace(fingerprint="dataset", actual_candle_count=0)
        ),
    )
    monkeypatch.setattr(
        research_module,
        "build_historical_feature_matrix",
        Mock(return_value=HistoricalFeatureMatrix("dataset", ())),
    )
    monkeypatch.setattr(
        research_module, "build_predictive_windows", Mock(return_value=())
    )
    monkeypatch.setattr(
        research_module, "analyze_predictive_edges_v4", Mock(return_value=v4)
    )
    monkeypatch.setattr(
        research_module, "analyze_predictive_edges_v41", Mock(return_value=v41)
    )
    monkeypatch.setattr(research_module, "generate_edge_guided_candidates", generated)
    monkeypatch.setattr(research_module, "evaluate_edge_guided_candidates", evaluated)
    monkeypatch.setattr(
        research_module, "load_reusable_edge_guided_stages", AsyncMock(return_value={})
    )

    session = SimpleNamespace(new=set(), dirty=set(), deleted=set())
    completed = await execute_weekly_instrument_research(
        session=session,  # type: ignore[arg-type]
        identity=IDENTITY,
        configuration=PredictiveEdgeConfiguration(),
        simulation=SimulationConfiguration(),
        initial_cash=Decimal("10000"),
        maximum_candidates=12,
        boundary=datetime(2026, 1, 4, 3, 17, tzinfo=UTC),
        persist=False,
    )

    assert generated.call_args.kwargs["hypotheses"] == (
        v4_hypothesis,
        v41_hypothesis,
    )
    assert evaluated.call_args.kwargs["hypotheses"] == (
        v4_hypothesis,
        v41_hypothesis,
    )
    assert (
        completed.diagnostics["regime_conditioned_v41_conclusion"]
        == "regime_conditioned_edge_confirmed"
    )
