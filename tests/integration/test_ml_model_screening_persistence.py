from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ml.screening.models import (
    MLModelScreeningEvaluationRecord,
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)
from app.ml.screening.persistence import fail_run, find_completed_run

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
EXPERIMENT_KEY = "m" * 64


@pytest_asyncio.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(get_settings().database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.experiment_key == EXPERIMENT_KEY
            )
        )
        await session.commit()
    yield sessions
    async with sessions() as session:
        await session.execute(
            delete(MLModelScreeningRunRecord).where(
                MLModelScreeningRunRecord.experiment_key == EXPERIMENT_KEY
            )
        )
        await session.commit()
    await engine.dispose()


def _run(run_id: UUID, *, status: str = "completed") -> MLModelScreeningRunRecord:
    now = datetime(2036, 1, 1, tzinfo=UTC)
    return MLModelScreeningRunRecord(
        id=run_id,
        experiment_key=EXPERIMENT_KEY,
        status=status,
        screening_version=1,
        experiment_contract_version=2,
        variable_registry_version=1,
        missing_data_policy_version=1,
        dataset_fingerprint="d" * 64,
        asset="BTC/USDT",
        provider="Binance",
        market_type="Spot",
        interval="5m",
        requested_engines=["ridge"],
        requested_future_durations=[1800],
        library_versions={"scikit-learn": "1.9.0"},
        deterministic_seed=1729,
        configuration={"thread_count": 2},
        started_at=now,
        completed_at=now if status == "completed" else None,
        failure_type=None,
        failure_message=None,
        evaluations_computed=1,
        evaluations_reused=0,
        results_generated=1,
        pareto_retained_count=1,
        elapsed_seconds=Decimal("1"),
        created_at=now,
    )


def _evaluation(run_id: UUID) -> MLModelScreeningEvaluationRecord:
    return MLModelScreeningEvaluationRecord(
        id=uuid4(),
        run_id=run_id,
        engine="ridge",
        future_duration_seconds=1800,
        fold_name="fold",
        training_row_count=10,
        validation_row_count=5,
        sample_count=5,
        mean_absolute_error=Decimal("0.1"),
        root_mean_squared_error=Decimal("0.1"),
        mean_prediction=Decimal("0"),
        mean_actual_return=Decimal("0"),
        pearson_correlation=None,
        spearman_correlation=None,
        directional_accuracy=Decimal("0.5"),
        zero_prediction_mae=Decimal("0.2"),
        mae_improvement_over_zero=Decimal("0.1"),
        positive_prediction_count=2,
        negative_prediction_count=3,
        elapsed_seconds=Decimal("1"),
        created_at=datetime(2036, 1, 1, tzinfo=UTC),
    )


async def test_unique_evaluation_and_result_constraints(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    run_id = uuid4()
    async with factory() as session:
        session.add(_run(run_id))
        await session.commit()
        session.add(_evaluation(run_id))
        await session.commit()
        session.add(_evaluation(run_id))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        now = datetime(2036, 1, 1, tzinfo=UTC)
        values = {
            "engine": "ridge",
            "future_duration_seconds": 1800,
            "folds_evaluated": 1,
            "total_validation_samples": 5,
            "median_mae_improvement_over_zero": Decimal("0.1"),
            "worst_fold_mae_improvement_over_zero": Decimal("0.1"),
            "median_directional_accuracy": Decimal("0.5"),
            "worst_fold_directional_accuracy": Decimal("0.5"),
            "median_pearson_correlation": None,
            "median_spearman_correlation": None,
            "folds_beating_zero": 1,
            "folds_with_positive_spearman": 0,
            "total_training_time_seconds": Decimal("1"),
            "pareto_retained": True,
            "display_rank": 1,
            "created_at": now,
        }
        session.add(MLModelScreeningResultRecord(id=uuid4(), run_id=run_id, **values))
        await session.commit()
        session.add(MLModelScreeningResultRecord(id=uuid4(), run_id=run_id, **values))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


async def test_only_completed_runs_are_reused_and_failure_is_terminal(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    async with factory() as session:
        running = _run(uuid4(), status="running")
        failed = _run(uuid4(), status="failed")
        session.add_all((running, failed))
        await session.commit()
        assert await find_completed_run(session, EXPERIMENT_KEY) is None

        await fail_run(
            session,
            running.id,
            RuntimeError("native engine failed"),
            datetime(2036, 1, 2, tzinfo=UTC),
        )
        await session.refresh(running)
        assert running.status == "failed"
        assert running.failure_type == "RuntimeError"

        completed = _run(uuid4())
        session.add(completed)
        await session.commit()
        reused = await find_completed_run(session, EXPERIMENT_KEY)
        assert reused is not None
        assert reused.id == completed.id
