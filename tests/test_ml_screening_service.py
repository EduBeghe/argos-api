import logging
import tracemalloc
from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.market.catalog import MARKET_INSTRUMENTS
from app.ml.screening import service
from app.ml.screening.calculation import FoldEvaluation
from app.ml.screening.config import DEFAULT_SCREENING_CONFIGURATION, ScreeningEngine
from app.ml.screening.metrics import EvaluationMetrics


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(
            fingerprint="d" * 64,
            instrument=MARKET_INSTRUMENTS[0].instrument,
        ),
        folds=(object(),),
    )


def _evaluation() -> FoldEvaluation:
    return FoldEvaluation(
        engine=ScreeningEngine.RIDGE,
        future_duration_seconds=1800,
        fold_name="fold",
        training_row_count=10,
        validation_row_count=5,
        metrics=EvaluationMetrics(
            sample_count=5,
            mean_absolute_error=0.1,
            root_mean_squared_error=0.1,
            mean_prediction=0.0,
            mean_actual_return=0.0,
            pearson_correlation=None,
            spearman_correlation=None,
            directional_accuracy=0.5,
            zero_prediction_mae=0.2,
            mae_improvement_over_zero=0.1,
            positive_prediction_count=2,
            negative_prediction_count=3,
        ),
        elapsed_seconds=0.01,
    )


@pytest.mark.asyncio
async def test_completed_run_is_reused_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    existing = SimpleNamespace(
        id=uuid4(),
        evaluations_computed=7,
        results_generated=2,
        pareto_retained_count=1,
    )
    monkeypatch.setattr(service, "library_versions", lambda: {"x": "1"})
    monkeypatch.setattr(service, "experiment_key", lambda **kwargs: "key")
    monkeypatch.setattr(service, "find_completed_run", AsyncMock(return_value=existing))
    create = AsyncMock()
    monkeypatch.setattr(service, "create_run", create)
    with caplog.at_level(logging.INFO):
        result = await service.screen_models(
            session=AsyncMock(),
            dataset=_dataset(),  # type: ignore[arg-type]
            engines=(ScreeningEngine.RIDGE,),
            durations=(timedelta(minutes=30),),
            configuration=DEFAULT_SCREENING_CONFIGURATION,
        )
    assert result.reused_existing_run is True
    assert result.evaluations_reused == 7
    create.assert_not_awaited()
    fields = cast(dict[str, object], vars(caplog.records[-1])["structured_fields"])
    assert fields["reused_existing_run"] is True
    assert fields["compute_avoided"] == 7


@pytest.mark.asyncio
async def test_force_recompute_bypasses_reuse_and_cleans_tracemalloc(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(service, "library_versions", lambda: {"x": "1"})
    monkeypatch.setattr(service, "experiment_key", lambda **kwargs: "key")
    find = AsyncMock()
    monkeypatch.setattr(service, "find_completed_run", find)
    monkeypatch.setattr(service, "create_run", AsyncMock(return_value=uuid4()))
    monkeypatch.setattr(service, "persist_evaluations", AsyncMock())
    monkeypatch.setattr(service, "complete_run", AsyncMock())
    monkeypatch.setattr(
        service, "evaluate_combination", lambda **kwargs: (_evaluation(),)
    )
    monkeypatch.setattr(service, "approximate_peak_resident_bytes", lambda value: value)
    with caplog.at_level(logging.INFO):
        result = await service.screen_models(
            session=AsyncMock(),
            dataset=_dataset(),  # type: ignore[arg-type]
            engines=(ScreeningEngine.RIDGE,),
            durations=(timedelta(minutes=30),),
            configuration=DEFAULT_SCREENING_CONFIGURATION,
            force_recompute=True,
        )
    assert result.reused_existing_run is False
    find.assert_not_awaited()
    assert not tracemalloc.is_tracing()
    fields = cast(dict[str, object], vars(caplog.records[-1])["structured_fields"])
    assert fields["reused_existing_run"] is False
    assert fields["evaluations_computed"] == 1


@pytest.mark.asyncio
async def test_partial_engine_failure_marks_failed_and_cleans_tracemalloc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "library_versions", lambda: {"x": "1"})
    monkeypatch.setattr(service, "experiment_key", lambda **kwargs: "key")
    monkeypatch.setattr(service, "find_completed_run", AsyncMock(return_value=None))
    monkeypatch.setattr(service, "create_run", AsyncMock(return_value=uuid4()))
    failed = AsyncMock()
    monkeypatch.setattr(service, "fail_run", failed)

    def fail(**kwargs: object) -> tuple[FoldEvaluation, ...]:
        raise RuntimeError("engine failed")

    monkeypatch.setattr(service, "evaluate_combination", fail)
    with pytest.raises(RuntimeError, match="engine failed"):
        await service.screen_models(
            session=AsyncMock(),
            dataset=_dataset(),  # type: ignore[arg-type]
            engines=(ScreeningEngine.RIDGE,),
            durations=(timedelta(minutes=30),),
            configuration=DEFAULT_SCREENING_CONFIGURATION,
        )
    failed.assert_awaited_once()
    assert not tracemalloc.is_tracing()
