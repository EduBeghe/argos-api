from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ml.contract import temporal_fold_definitions
from app.ml.screening.models import (
    MLModelScreeningEvaluationRecord,
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
EXPERIMENT_PREFIX = "ml-read-api-"
BASE_TIME = datetime(2038, 1, 1, tzinfo=UTC)


def _run(
    run_id: UUID,
    *,
    suffix: str,
    status: str = "completed",
    asset: str = "BTC/USDT",
    started_offset: int = 0,
) -> MLModelScreeningRunRecord:
    started_at = BASE_TIME + timedelta(hours=started_offset)
    return MLModelScreeningRunRecord(
        id=run_id,
        experiment_key=(EXPERIMENT_PREFIX + suffix).ljust(64, "x"),
        status=status,
        screening_version=1,
        experiment_contract_version=2,
        variable_registry_version=1,
        missing_data_policy_version=1,
        dataset_fingerprint=("dataset-" + suffix).ljust(64, "d"),
        asset=asset,
        provider="binance",
        market_type="spot",
        interval="5m",
        requested_engines=["ridge", "lightgbm", "catboost"],
        requested_future_durations=[1800, 3600],
        library_versions={"scikit-learn": "1.9.0"},
        deterministic_seed=1729,
        configuration={"thread_count": 2},
        started_at=started_at,
        completed_at=started_at + timedelta(minutes=1)
        if status == "completed"
        else None,
        failure_type="RuntimeError" if status == "failed" else None,
        failure_message="safe failure" if status == "failed" else None,
        evaluations_computed=70,
        evaluations_reused=0,
        results_generated=10,
        pareto_retained_count=2,
        elapsed_seconds=Decimal("60.123456789012345678"),
        created_at=started_at,
    )


def _result(
    run_id: UUID,
    *,
    engine: str,
    duration: int,
    rank: int,
    classification: str,
    pareto: bool = False,
) -> MLModelScreeningResultRecord:
    values: dict[str, object] = {
        "folds_evaluated": 7,
        "median_mae_improvement_over_zero": Decimal("0.010000000000000001"),
        "worst_fold_mae_improvement_over_zero": Decimal("0.001"),
        "median_directional_accuracy": Decimal("0.510000000000000001"),
        "worst_fold_directional_accuracy": Decimal("0.5"),
        "median_pearson_correlation": Decimal("0.01"),
        "median_spearman_correlation": Decimal("0.01"),
        "folds_beating_zero": 7,
        "folds_with_positive_spearman": 7,
    }
    if classification == "insufficient":
        values["folds_beating_zero"] = 6
    elif classification == "discarded":
        values.update(
            median_mae_improvement_over_zero=Decimal("-0.01"),
            median_directional_accuracy=Decimal("0.49"),
            median_pearson_correlation=Decimal("-0.01"),
            median_spearman_correlation=Decimal("-0.01"),
            folds_beating_zero=0,
            folds_with_positive_spearman=0,
        )
    elif classification == "incomplete":
        values.update(folds_evaluated=6, median_pearson_correlation=None)
    return MLModelScreeningResultRecord(
        id=uuid4(),
        run_id=run_id,
        engine=engine,
        future_duration_seconds=duration,
        total_validation_samples=700,
        total_training_time_seconds=Decimal("12.345678901234567890"),
        pareto_retained=pareto,
        display_rank=rank,
        created_at=BASE_TIME,
        **values,
    )


def _evaluation(
    run_id: UUID, result: MLModelScreeningResultRecord, fold_name: str, ordinal: int
) -> MLModelScreeningEvaluationRecord:
    return MLModelScreeningEvaluationRecord(
        id=uuid4(),
        run_id=run_id,
        engine=result.engine,
        future_duration_seconds=result.future_duration_seconds,
        fold_name=fold_name,
        training_row_count=100,
        validation_row_count=20,
        sample_count=20,
        mean_absolute_error=Decimal("0.123456789012345678"),
        root_mean_squared_error=Decimal("0.2"),
        mean_prediction=Decimal("0.01"),
        mean_actual_return=Decimal("0.02"),
        pearson_correlation=Decimal("0.1"),
        spearman_correlation=Decimal("0.1"),
        directional_accuracy=Decimal("0.55"),
        zero_prediction_mae=Decimal("0.2"),
        mae_improvement_over_zero=Decimal("0.076543210987654322"),
        positive_prediction_count=11,
        negative_prediction_count=9,
        elapsed_seconds=Decimal("1.000000000000000001"),
        created_at=BASE_TIME + timedelta(seconds=ordinal),
    )


@pytest_asyncio.fixture
async def screening_data() -> AsyncIterator[dict[str, object]]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    run_id = uuid4()
    other_run_id = uuid4()
    running_id = uuid4()
    failed_id = uuid4()
    async with factory() as session:
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.experiment_key.like(f"{EXPERIMENT_PREFIX}%")
            )
        )
        runs = (
            _run(run_id, suffix="completed", started_offset=3),
            _run(other_run_id, suffix="older", started_offset=0),
            _run(running_id, suffix="running", status="running", started_offset=5),
            _run(failed_id, suffix="failed", status="failed", started_offset=4),
        )
        session.add_all(runs)
        await session.flush()
        classifications = (
            "promising",
            "insufficient",
            "discarded",
            "incomplete",
            "promising",
            "insufficient",
            "discarded",
            "incomplete",
            "promising",
            "insufficient",
        )
        results = [
            _result(
                run_id,
                engine=("ridge", "lightgbm", "catboost")[index % 3],
                duration=1800 + index * 1800,
                rank=index + 1,
                classification=classification,
                pareto=index in (0, 4),
            )
            for index, classification in enumerate(classifications)
        ]
        foreign_result = _result(
            other_run_id,
            engine="ridge",
            duration=1800,
            rank=1,
            classification="promising",
        )
        session.add_all([*results, foreign_result])
        await session.flush()
        folds = temporal_fold_definitions()
        session.add_all(
            [
                _evaluation(run_id, results[0], fold.name, index)
                for index, fold in enumerate(reversed(folds))
            ]
        )
        await session.commit()
    yield {
        "run_id": run_id,
        "other_run_id": other_run_id,
        "result_id": results[0].id,
        "foreign_result_id": foreign_result.id,
        "fold_names": [fold.name for fold in folds],
    }
    async with factory() as session:
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.experiment_key.like(f"{EXPERIMENT_PREFIX}%")
            )
        )
        await session.commit()
    await engine.dispose()


async def test_run_list_filters_pagination_and_latest(
    client: TestClient, screening_data: dict[str, object]
) -> None:
    response = client.get(
        "/optimization/ml-screening-runs",
        params={
            "asset": "BTC/USDT",
            "provider": "binance",
            "market_type": "spot",
            "interval": "5m",
            "status": "completed",
            "limit": 1,
            "offset": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert response.json()["limit"] == 1
    assert response.json()["offset"] == 1
    assert response.json()["has_more"] is False

    latest = client.get(
        "/optimization/ml-screening-runs/latest",
        params={
            "asset": "BTC/USDT",
            "provider": "binance",
            "market_type": "spot",
            "interval": "5m",
        },
    )
    assert latest.status_code == 200
    assert latest.json()["id"] == str(screening_data["run_id"])
    unknown = client.get(
        "/optimization/ml-screening-runs/latest",
        params={
            "asset": "ETH/USDT",
            "provider": "binance",
            "market_type": "spot",
            "interval": "5m",
        },
    )
    assert unknown.status_code == 404


async def test_run_detail_counts(
    client: TestClient, screening_data: dict[str, object]
) -> None:
    response = client.get(f"/optimization/ml-screening-runs/{screening_data['run_id']}")
    assert response.status_code == 200
    assert response.json()["classification_counts"] == {
        "promising": 3,
        "insufficient": 3,
        "discarded": 2,
        "incomplete": 2,
    }
    assert response.json()["engine_counts"] == {
        "ridge": 4,
        "lightgbm": 3,
        "catboost": 3,
    }


async def test_result_pagination_combined_filters_and_order(
    client: TestClient, screening_data: dict[str, object]
) -> None:
    base = f"/optimization/ml-screening-runs/{screening_data['run_id']}/results"
    first = client.get(base, params={"limit": 4, "offset": 0}).json()
    second = client.get(base, params={"limit": 4, "offset": 4}).json()
    assert first["total"] == 10
    assert first["has_more"] is True
    assert len(first["items"] + second["items"]) == 8
    priorities = {"promising": 0, "insufficient": 1, "discarded": 2, "incomplete": 3}
    keys = [
        (
            priorities[item["evidence_classification"]],
            not item["pareto_retained"],
            item["display_rank"],
        )
        for item in first["items"] + second["items"]
    ]
    assert keys == sorted(keys)

    filtered = client.get(
        base,
        params={
            "engine": "lightgbm",
            "pareto_retained": True,
            "evidence_classification": "promising",
            "minimum_future_duration_seconds": 9000,
            "maximum_future_duration_seconds": 9000,
        },
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    invalid = client.get(
        base,
        params={
            "minimum_future_duration_seconds": 3600,
            "maximum_future_duration_seconds": 1800,
        },
    )
    assert invalid.status_code == 422


async def test_result_detail_fold_scope_order_and_decimal_precision(
    client: TestClient, screening_data: dict[str, object]
) -> None:
    path = (
        f"/optimization/ml-screening-runs/{screening_data['run_id']}"
        f"/results/{screening_data['result_id']}"
    )
    response = client.get(path)
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["fold_evaluations"]) == 7
    assert [
        item["fold_name"] for item in payload["fold_evaluations"]
    ] == screening_data["fold_names"]
    assert payload["result"]["median_directional_accuracy"] == "0.510000000000000001"
    assert (
        payload["fold_evaluations"][0]["mean_absolute_error"] == "0.123456789012345678"
    )


async def test_result_and_run_ownership_404s(
    client: TestClient, screening_data: dict[str, object]
) -> None:
    run_id = screening_data["run_id"]
    assert client.get(f"/optimization/ml-screening-runs/{uuid4()}").status_code == 404
    assert (
        client.get(
            f"/optimization/ml-screening-runs/{run_id}/results/{uuid4()}"
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/optimization/ml-screening-runs/{run_id}/results/{screening_data['foreign_result_id']}"
        ).status_code
        == 404
    )
