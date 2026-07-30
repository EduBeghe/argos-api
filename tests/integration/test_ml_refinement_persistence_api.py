import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.contract import DEVELOPMENT_START, FINAL_TEST_END, FINAL_TEST_START
from app.ml.dataset import MLDatasetMetadata, build_ml_experiment_dataset
from app.ml.reader import candle_columns
from app.ml.refinement import service as refinement_service
from app.ml.refinement.calculation import RefinementEvaluationResult
from app.ml.refinement.contracts import MLRefinementRequest, RefinementClassification
from app.ml.refinement.identity import DatasetFingerprintChanged
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)
from app.ml.refinement.persistence import (
    complete_run,
    effective_evaluation_fingerprint,
    existing_run_evaluation,
    experiment_key,
    fail_run,
    find_completed_run,
    persist_evaluation,
    resolve_source,
    start_run,
)
from app.ml.refinement.planning import build_refinement_plan
from app.ml.refinement.queue import (
    QueueDisposition,
    cancel_refinement,
    claim_next_run,
    enqueue_refinement,
    fail_stale_runs,
    retry_refinement,
)
from app.ml.refinement.service import prepare_refinement
from app.ml.screening.engines import library_versions
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
    versions = library_versions()
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
        evaluation_fingerprint = effective_evaluation_fingerprint(
            dataset_fingerprint="d" * 64,
            candidate_fingerprint=candidate.fingerprint,
            engine=candidate.engine.value,
            versions=versions,
            deterministic_seed=refinement_request.deterministic_seed,
        )
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
            fingerprint=evaluation_fingerprint,
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
    assert client.post("/optimization/ml-refinement-runs").status_code == 422

    async with factory() as session:
        count = await session.scalar(select(func.count(MLRefinementRunRecord.id)))
        assert count == 1
        evaluation_count = await session.scalar(
            select(func.count(MLRefinementEvaluationRecord.id))
        )
        assert evaluation_count == 1


async def test_failed_run_retains_complete_evaluations_for_retry(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    payload = request(source_run_id, source_result_id).model_dump(mode="json")
    payload["search_space"][0]["alpha"] = {"values": [1.0, 2.0]}
    payload["maximum_trials"] = 2
    refinement_request = MLRefinementRequest.model_validate(payload)
    plan = build_refinement_plan(refinement_request)
    versions = library_versions()
    key = experiment_key(
        request=refinement_request, dataset_fingerprint="r" * 64, versions=versions
    )
    fingerprints = [
        effective_evaluation_fingerprint(
            dataset_fingerprint="r" * 64,
            candidate_fingerprint=candidate.fingerprint,
            engine=candidate.engine.value,
            versions=versions,
            deterministic_seed=refinement_request.deterministic_seed,
        )
        for candidate in plan.evaluations
    ]
    async with factory() as session:
        run = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key=key,
            dataset_fingerprint="r" * 64,
            versions=versions,
            now=NOW,
        )
        first_result = RefinementEvaluationResult(
            plan.evaluations[0],
            (),
            {"fold_count": 0},
            RefinementClassification.INCOMPLETE,
            ("incomplete_fold_count",),
            0.1,
        )
        await persist_evaluation(
            session=session,
            run_id=run.id,
            fingerprint=fingerprints[0],
            result=first_result,
            reused=None,
            request=refinement_request,
            now=NOW,
        )
        await fail_run(session, run.id, RuntimeError("simulated"), NOW)

        recovered = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key=key,
            dataset_fingerprint="r" * 64,
            versions=versions,
            now=NOW,
        )
        assert recovered.id == run.id
        assert await existing_run_evaluation(session, run.id, fingerprints[0])
        assert await existing_run_evaluation(session, run.id, fingerprints[1]) is None
        second_result = RefinementEvaluationResult(
            plan.evaluations[1],
            (),
            {"fold_count": 0},
            RefinementClassification.INCOMPLETE,
            ("incomplete_fold_count",),
            0.2,
        )
        await persist_evaluation(
            session=session,
            run_id=run.id,
            fingerprint=fingerprints[1],
            result=second_result,
            reused=None,
            request=refinement_request,
            now=NOW,
        )
        await complete_run(
            session,
            run.id,
            computed=1,
            reused=1,
            omitted=0,
            budget_exhausted=False,
            reason="trial_plan_completed",
            elapsed=1,
            now=NOW,
        )
        await session.refresh(recovered)
        count = await session.scalar(
            select(func.count(MLRefinementEvaluationRecord.id)).where(
                MLRefinementEvaluationRecord.run_id == run.id
            )
        )
        assert count == 2
        assert recovered.status == "completed"
        assert recovered.evaluations_computed == 1
        assert recovered.evaluations_reused == 1


async def test_queue_api_location_cancel_retry_and_claim_lifecycle(
    client: TestClient,
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    payload = request(source_run_id, source_result_id).model_dump(mode="json")

    response = client.post("/optimization/ml-refinement-runs", json=payload)
    assert response.status_code == 202
    body = response.json()
    assert body["disposition"] == "queued"
    assert response.headers["location"] == body["location"]
    run_id = UUID(body["run"]["id"])

    duplicate = client.post("/optimization/ml-refinement-runs", json=payload)
    assert duplicate.status_code == 202
    assert duplicate.json()["disposition"] == "already_queued"
    assert duplicate.json()["run"]["id"] == str(run_id)

    cancelled = client.post(f"/optimization/ml-refinement-runs/{run_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["started_at"] is None

    retry = client.post(f"/optimization/ml-refinement-runs/{run_id}/retry")
    assert retry.status_code == 202
    assert retry.json()["disposition"] == "requeued_cancelled"
    assert retry.headers["location"] == retry.json()["location"]

    async with factory() as session:
        claimed = await claim_next_run(session, worker_id="worker-a", now=NOW)
        assert claimed is not None and claimed.id == run_id
        running_cancel = await cancel_refinement(
            session, run_id, now=NOW + timedelta(seconds=1)
        )
        assert running_cancel.status == "running"
        assert running_cancel.cancellation_requested_at is not None


async def test_zero_progress_fingerprint_failure_retries_the_same_run_and_advances(
    client: TestClient,
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    response = client.post(
        "/optimization/ml-refinement-runs",
        json=refinement_request.model_dump(mode="json"),
    )
    assert response.status_code == 202
    run_id = UUID(response.json()["run"]["id"])

    async with factory() as session:
        claimed = await claim_next_run(session, worker_id="worker-old", now=NOW)
        assert claimed is not None and claimed.id == run_id
        await fail_run(
            session,
            run_id,
            DatasetFingerprintChanged("pre-fix boundary mismatch"),
            NOW + timedelta(seconds=1),
            worker_id="worker-old",
        )

    retry = client.post(f"/optimization/ml-refinement-runs/{run_id}/retry")
    assert retry.status_code == 202
    assert UUID(retry.json()["run"]["id"]) == run_id

    plan = build_refinement_plan(refinement_request)
    instrument = Instrument(
        Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
    )
    dataset = build_ml_experiment_dataset(
        instrument=instrument,
        columns=candle_columns(
            open_times=[],
            close_times=[],
            open_prices=[],
            high_prices=[],
            low_prices=[],
            close_prices=[],
            volumes=[],
        ),
        requested_end=FINAL_TEST_START,
    )
    async with factory() as session:
        claimed = await claim_next_run(
            session, worker_id="worker-new", now=NOW + timedelta(seconds=2)
        )
        assert claimed is not None and claimed.id == run_id
        result = await refinement_service.execute_refinement(
            session=session,
            identity_metadata=cast(
                MLDatasetMetadata,
                SimpleNamespace(
                    fingerprint="d" * 64,
                    instrument=instrument,
                    requested_start=DEVELOPMENT_START,
                    requested_end=FINAL_TEST_END,
                    candle_count=55_000,
                ),
            ),
            development_dataset=dataset,
            request=refinement_request,
            claimed_run=claimed,
            worker_id="worker-new",
            heartbeat_interval_seconds=1,
            now=NOW + timedelta(seconds=2),
        )
        await session.refresh(claimed)
        assert result.run_id == run_id
        assert result.evaluations_computed == len(plan.evaluations) == 1
        assert claimed.status == "completed"
        assert claimed.failure_type is None
        assert (
            await session.scalar(
                select(func.count(MLRefinementEvaluationRecord.id)).where(
                    MLRefinementEvaluationRecord.run_id == run_id
                )
            )
            == 1
        )


async def test_concurrent_enqueue_skip_locked_and_stale_recovery(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)

    async def enqueue_once() -> tuple[UUID, QueueDisposition]:
        async with factory() as session:
            result = await enqueue_refinement(session, refinement_request, now=NOW)
            return result.run.id, result.disposition

    first, second = await asyncio.gather(enqueue_once(), enqueue_once())
    assert first[0] == second[0]
    assert {first[1], second[1]} == {
        QueueDisposition.QUEUED,
        QueueDisposition.ALREADY_QUEUED,
    }

    async with factory() as session_one, factory() as session_two:
        claimed_one, claimed_two = await asyncio.gather(
            claim_next_run(session_one, worker_id="worker-one", now=NOW),
            claim_next_run(session_two, worker_id="worker-two", now=NOW),
        )
        assert sum(item is not None for item in (claimed_one, claimed_two)) == 1

    async with factory() as session:
        expired = await fail_stale_runs(
            session, stale_after=timedelta(seconds=300), now=NOW + timedelta(hours=1)
        )
        assert expired == 1
        record = await session.get(MLRefinementRunRecord, first[0])
        assert record is not None
        assert record.status == "failed"
        assert record.failure_type == "WorkerLeaseExpired"
        evaluation_count = await session.scalar(
            select(func.count(MLRefinementEvaluationRecord.id)).where(
                MLRefinementEvaluationRecord.run_id == record.id
            )
        )
        retried = await retry_refinement(session, record.id)
        assert retried.disposition is QueueDisposition.REQUEUED_FAILED
        assert retried.run.status == "queued"
        assert (
            await session.scalar(
                select(func.count(MLRefinementEvaluationRecord.id)).where(
                    MLRefinementEvaluationRecord.run_id == record.id
                )
            )
            == evaluation_count
        )
