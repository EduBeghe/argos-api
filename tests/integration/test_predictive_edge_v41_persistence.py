from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.optimization.predictive_edge_memory import (
    PredictiveEdgeNearMissRecord,
    PredictiveEdgeRunDiagnosticRecord,
    PredictiveEdgeRunRecord,
)
from app.optimization.predictive_edge_read import (
    get_predictive_edge_diagnostics,
    list_predictive_edge_near_misses,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
KEY = "v41-persistence-" + "1" * 46


@pytest_asyncio.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(get_settings().database_url)
    value = async_sessionmaker(engine, expire_on_commit=False)
    async with value() as session:
        await session.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == KEY
            )
        )
        await session.commit()
    yield value
    async with value() as session:
        await session.execute(
            delete(PredictiveEdgeRunRecord).where(
                PredictiveEdgeRunRecord.experiment_key == KEY
            )
        )
        await session.commit()
    await engine.dispose()


def _run(run_id: UUID) -> PredictiveEdgeRunRecord:
    started = datetime(2032, 1, 1, tzinfo=UTC)
    return PredictiveEdgeRunRecord(
        id=run_id,
        experiment_key=KEY,
        status="completed",
        asset="BTC/USDT",
        provider="binance",
        market_type="spot",
        interval="5m",
        dataset_fingerprint="d" * 64,
        range_start=started,
        range_end=started + timedelta(days=730),
        algorithm_version=7,
        score_model_name="baseline_technical",
        score_model_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        neutral_threshold=Decimal("0"),
        decision_interval_seconds=1800,
        training_days=180,
        validation_days=60,
        test_days=60,
        step_days=60,
        maximum_windows=6,
        bin_count=5,
        minimum_observations=8,
        minimum_windows=3,
        minimum_stable_direction_ratio=Decimal("0.6"),
        minimum_absolute_correlation=Decimal("0.05"),
        minimum_directional_hit_rate=Decimal("0.55"),
        minimum_hit_rate_improvement_over_benchmark=Decimal("0.01"),
        windows_available=3,
        windows_required=3,
        conclusion_reason=None,
        windows_generated=3,
        aggregate_metrics_computed=0,
        metrics_reused=0,
        compute_avoided_estimate=0,
        reuse_count=0,
        started_at=started,
        completed_at=started + timedelta(hours=1),
        failed_at=None,
        failure_reason=None,
    )


def _evidence(*, economic: bool, stable: bool) -> dict[str, object]:
    return {
        "observations": 42,
        "gross_movement": "0.002",
        "aligned_gross_movement": "0.002",
        "net_movement": "-0.001",
        "directional_hit_rate": "0.57",
        "benchmark": "0.54",
        "benchmark_improvement": "0.03",
        "standard_error": "0.0002",
        "stable": stable,
        "economic": economic,
        "passes": stable and economic,
    }


async def test_v41_diagnostics_and_near_misses_survive_session_restart(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    run_id = uuid4()
    counters = {
        "a1": {"generated": 16, "computed": 48, "reused": 0, "retained": 24},
        "a2": {"generated": 5, "computed": 15, "reused": 0, "retained": 5},
        "a3": {"generated": 192, "computed": 60, "reused": 0, "retained": 60},
        "validation": {"generated": 40, "computed": 40, "reused": 0, "retained": 0},
        "test": {"generated": 0, "computed": 0, "reused": 0, "retained": 0},
    }
    async with factory() as session:
        session.add(_run(run_id))
        session.add(
            PredictiveEdgeRunDiagnosticRecord(
                id=uuid4(),
                run_id=run_id,
                contract_version=4,
                effective_parameters=[],
                rejection_counts_by_reason={},
                rejection_counts_by_predictor={},
                rejection_counts_by_horizon={},
                rejection_counts_by_condition={},
                rejection_counts_by_neutral_policy={},
                rejection_counts_by_cadence={},
                strongest_observed_signal=None,
                bottleneck_stage="validation",
                diagnostic_conclusion="no_detectable_relationship",
                outcome_explanation="No deployable Strategy.",
                stage_timings={},
                computed_counts={"cheap_relationships": 10},
                reused_counts={},
                regime_conditioned_conclusion="context_improves_signal_but_not_economic",
                regime_conditioned_stage_counters=counters,
                regime_conditioned_summary={
                    "contract_version": 42,
                    "confirmed_interaction_count": 0,
                    "actionable_near_miss_count": 1,
                    "diagnostic_near_miss_count": 1,
                    "hypotheses_generated": 0,
                },
            )
        )
        common = dict(
            run_id=run_id,
            cadence_seconds=1800,
            horizon_seconds=14400,
            bin_count=3,
            condition="all",
            neutral_policy="training_noise",
            profile="balanced_v41",
            passed_requirements=[],
            requirement_margins=[],
            usable_observations=42,
            stable_direction_ratio=Decimal("0.57"),
            expected_net_movement=Decimal("-0.001"),
            normalized_missing_distance=Decimal("1.2"),
            engine="v4.1",
            engine_contract_version=42,
        )
        session.add(
            PredictiveEdgeNearMissRecord(
                id=uuid4(),
                rank=1,
                category_rank=1,
                kind="actionable",
                edge_fingerprint="a" * 64,
                predictor_name="return_12",
                directional_predictor="return_12",
                directional_predictor_version=1,
                context_predictor="atr_14_relative",
                context_predictor_version=1,
                context_regime="high",
                selected_bin=1,
                directional_lower_bound=None,
                directional_upper_bound=Decimal("-0.01"),
                context_lower_bound=Decimal("0.02"),
                context_upper_bound=None,
                direction_mapping="positive",
                suggested_action="long_entry",
                classification="regime_conditioned_near_miss",
                primary_rejection_reason="context_improves_signal_but_not_economic",
                failed_requirements=["movement_does_not_cover_costs"],
                training_evidence=_evidence(economic=True, stable=True),
                directional_baseline_evidence=_evidence(economic=False, stable=True),
                validation_evidence=_evidence(economic=False, stable=True),
                test_evidence=None,
                expected_gross_movement=Decimal("0.002"),
                expected_aligned_movement=Decimal("0.002"),
                directional_hit_rate=Decimal("0.57"),
                benchmark=Decimal("0.54"),
                benchmark_improvement=Decimal("0.03"),
                observations=42,
                context_improvement=Decimal("0.0004"),
                usable_window_count=3,
                confirmed_window_count=0,
                **common,
            )
        )
        session.add(
            PredictiveEdgeNearMissRecord(
                id=uuid4(),
                rank=2,
                category_rank=1,
                kind="diagnostic",
                edge_fingerprint="b" * 64,
                predictor_name="relative_volume_20",
                directional_predictor=None,
                context_predictor="relative_volume_20",
                context_regime=None,
                classification="diagnostic_context_predictor",
                primary_rejection_reason="non_directional_predictor",
                failed_requirements=["non_directional_predictor"],
                **common,
            )
        )
        await session.commit()

    async with factory() as restarted:
        diagnostics = await get_predictive_edge_diagnostics(restarted, run_id)
        actionable = await list_predictive_edge_near_misses(
            restarted, run_id=run_id, kind="actionable"
        )
        diagnostic = await list_predictive_edge_near_misses(
            restarted, run_id=run_id, kind="diagnostic"
        )

    assert diagnostics.regime_conditioned_stage_counters == counters
    assert diagnostics.regime_conditioned_summary is not None
    assert actionable.items[0].engine == "v4.1"
    assert actionable.items[0].validation_evidence is not None
    assert actionable.items[0].validation_evidence.net_movement == Decimal("-0.001")
    assert actionable.items[0].context_improvement == Decimal("0.0004")
    assert diagnostic.items[0].kind == "diagnostic"
    assert diagnostic.items[0].context_predictor == "relative_volume_20"
