from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ml.refinement.calculation import RefinementEvaluationResult
from app.ml.refinement.contracts import MLRefinementRequest, RefinementClassification
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)
from app.ml.refinement.persistence import (
    complete_run,
    experiment_key,
    find_completed_run,
    persist_evaluation,
    resolve_source,
    start_run,
)
from app.ml.refinement.planning import build_refinement_plan
from app.ml.refinement.service import prepare_refinement
from app.ml.screening.models import (
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
PREFIX = "ml-refinement-test-"
NOW = datetime(2039, 1, 1, tzinfo=UTC)


def request(run_id: UUID, result_id: UUID) -> MLRefinementRequest:
    return MLRefinementRequest.model_validate(
        {
            "source_screening_run_id": run_id,
            "source_screening_result_id": result_id,
            "engines": ["ridge"],
            "future_durations_seconds": [1800],
            "search_space": [
                {
                    "engine": "ridge",
                    "alpha": {"values": [1.0]},
                    "fit_intercept": {"values": [True]},
                }
            ],
            "feature_selection": {"strategies": [{"strategy": "all"}]},
            "prediction_objectives": ["future_return_regression"],
            "confidence_buckets": ["0.10"],
            "fee_rate": "0.001",
            "slippage_rate": "0.0005",
            "maximum_trials": 1,
            "maximum_runtime_seconds": 60,
            "deterministic_seed": 1729,
        }
    )


def source_run(run_id: UUID) -> MLModelScreeningRunRecord:
    return MLModelScreeningRunRecord(
        id=run_id,
        experiment_key=(PREFIX + "source").ljust(64, "s"),
        status="completed",
        screening_version=1,
        experiment_contract_version=2,
        variable_registry_version=1,
        missing_data_policy_version=1,
        dataset_fingerprint="d" * 64,
        asset="BTC/USDT",
        provider="binance",
        market_type="spot",
        interval="5m",
        requested_engines=["ridge"],
        requested_future_durations=[1800],
        library_versions={"scikit-learn": "1.9.0"},
        deterministic_seed=1729,
        configuration={},
        started_at=NOW,
        completed_at=NOW,
        failure_type=None,
        failure_message=None,
        evaluations_computed=7,
        evaluations_reused=0,
        results_generated=1,
        pareto_retained_count=1,
        elapsed_seconds=Decimal("1"),
        created_at=NOW,
    )


def source_result(run_id: UUID, result_id: UUID) -> MLModelScreeningResultRecord:
    return MLModelScreeningResultRecord(
        id=result_id,
        run_id=run_id,
        engine="ridge",
        future_duration_seconds=1800,
        folds_evaluated=7,
        total_validation_samples=700,
        median_mae_improvement_over_zero=Decimal("0.1"),
        worst_fold_mae_improvement_over_zero=Decimal("0.01"),
        median_directional_accuracy=Decimal("0.55"),
        worst_fold_directional_accuracy=Decimal("0.5"),
        median_pearson_correlation=Decimal("0.1"),
        median_spearman_correlation=Decimal("0.1"),
        folds_beating_zero=7,
        folds_with_positive_spearman=7,
        total_training_time_seconds=Decimal("1"),
        pareto_retained=True,
        display_rank=1,
        created_at=NOW,
    )


@pytest_asyncio.fixture
async def refinement_factory() -> AsyncIterator[
    tuple[async_sessionmaker[AsyncSession], UUID, UUID]
]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    run_id, result_id = uuid4(), uuid4()
    async with factory() as session:
        prior_source_runs = select(MLModelScreeningRunRecord.id).where(
            MLModelScreeningRunRecord.experiment_key.like(f"{PREFIX}%")
        )
        await session.execute(
            delete(MLRefinementRunRecord).where(
                MLRefinementRunRecord.source_screening_run_id.in_(prior_source_runs)
            )
        )
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.experiment_key.like(f"{PREFIX}%")
            )
        )
        session.add(source_run(run_id))
        await session.flush()
        session.add(source_result(run_id, result_id))
        await session.commit()
    yield factory, run_id, result_id
    async with factory() as session:
        await session.execute(
            delete(MLRefinementRunRecord).where(
                MLRefinementRunRecord.source_screening_run_id == run_id
            )
        )
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.experiment_key.like(f"{PREFIX}%")
            )
        )
        await session.commit()
    await engine.dispose()


async def test_persistence_reuse_and_read_only_api(
    client: TestClient,
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    plan = build_refinement_plan(refinement_request)
    versions = {"scikit-learn": "1.9.0", "lightgbm": "4", "catboost": "1"}
    key = experiment_key(
        request=refinement_request, dataset_fingerprint="d" * 64, versions=versions
    )
    async with factory() as session:
        count_before_dry_run = await session.scalar(
            select(func.count(MLRefinementRunRecord.id))
        )
        dry_run = await prepare_refinement(
            session=session,
            request=refinement_request,
            dataset_fingerprint="d" * 64,
        )
        count_after_dry_run = await session.scalar(
            select(func.count(MLRefinementRunRecord.id))
        )
        assert dry_run.experiment_key == key
        assert count_after_dry_run == count_before_dry_run == 0
        await resolve_source(session, refinement_request)
        run = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key=key,
            dataset_fingerprint="d" * 64,
            versions=versions,
            now=NOW,
        )
        candidate = plan.evaluations[0]
        result = RefinementEvaluationResult(
            candidate,
            (),
            {"fold_count": 0},
            RefinementClassification.INCOMPLETE,
            ("incomplete_fold_count",),
            0.123456789012345678,
        )
        await persist_evaluation(
            session=session,
            run_id=run.id,
            fingerprint=candidate.fingerprint,
            result=result,
            reused=None,
            request=refinement_request,
            now=NOW,
        )
        await complete_run(
            session,
            run.id,
            computed=1,
            reused=0,
            omitted=0,
            budget_exhausted=False,
            reason="trial_plan_completed",
            elapsed=1.25,
            now=NOW,
        )
        assert (await find_completed_run(session, key)).id == run.id  # type: ignore[union-attr]

    listing = client.get("/optimization/ml-refinement-runs", params={"limit": 1})
    assert listing.status_code == 200
    assert listing.json()["total"] >= 1
    detail = client.get(f"/optimization/ml-refinement-runs/{run.id}")
    assert detail.status_code == 200
    assert detail.json()["classification_counts"]["incomplete"] == 1
    evaluations = client.get(
        f"/optimization/ml-refinement-runs/{run.id}/evaluations",
        params={"classification": "incomplete", "limit": 1, "offset": 0},
    )
    assert evaluations.status_code == 200
    assert evaluations.json()["total"] == 1
    evaluation_id = evaluations.json()["items"][0]["id"]
    evidence = client.get(
        f"/optimization/ml-refinement-runs/{run.id}/evaluations/{evaluation_id}"
    )
    assert evidence.status_code == 200
    assert evidence.json()["elapsed_seconds"] == "0.123456789012345680"
    assert client.get(f"/optimization/ml-refinement-runs/{uuid4()}").status_code == 404
    assert client.post("/optimization/ml-refinement-runs").status_code == 405

    async with factory() as session:
        count = await session.scalar(select(func.count(MLRefinementRunRecord.id)))
        assert count == 1
        evaluation_count = await session.scalar(
            select(func.count(MLRefinementEvaluationRecord.id))
        )
        assert evaluation_count == 1
