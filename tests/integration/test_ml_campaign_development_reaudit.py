from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.ml.campaign import development_reaudit
from app.ml.campaign.contracts import CampaignRequest
from app.ml.campaign.development_reaudit import (
    DevelopmentReauditError,
    build_development_reaudit_report,
    report_json,
)
from app.ml.campaign.models import (
    MLSearchCampaignChildRecord,
    MLSearchCampaignFamilyRecord,
    MLSearchCampaignRecord,
    MLSearchCampaignSourceRecord,
)
from app.ml.protected.models import MLFinalValidationRecord
from app.ml.reader import MLCandleColumns
from app.ml.refinement.contracts import MLRefinementRequest
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)
from app.ml.refinement.persistence import effective_evaluation_fingerprint
from app.ml.refinement.planning import build_refinement_plan
from app.ml.screening.models import (
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)
from app.paper.deployment_models import PaperDeploymentRecord
from app.paper.persistence import PaperTradingRunRecord

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
NOW = datetime(2040, 1, 1, tzinfo=UTC)
PREFIX = "development-reaudit-test-"


async def cleanup(session: AsyncSession) -> None:
    source_ids = select(MLModelScreeningRunRecord.id).where(
        MLModelScreeningRunRecord.experiment_key.like(f"{PREFIX}%")
    )
    campaign_ids = select(MLSearchCampaignRecord.id).where(
        MLSearchCampaignRecord.experiment_key.like(f"{PREFIX}%")
    )
    await session.execute(
        delete(MLSearchCampaignRecord).where(
            MLSearchCampaignRecord.id.in_(campaign_ids)
        )
    )
    await session.execute(
        delete(MLRefinementRunRecord).where(
            MLRefinementRunRecord.source_screening_run_id.in_(source_ids)
        )
    )
    await session.execute(
        delete(MLModelScreeningRunRecord).where(
            MLModelScreeningRunRecord.id.in_(source_ids)
        )
    )
    await session.commit()


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        await cleanup(database_session)
        yield database_session
        await database_session.rollback()
        await cleanup(database_session)
    await engine.dispose()


async def seed_source(session: AsyncSession) -> tuple[UUID, UUID]:
    source_id, result_id = uuid4(), uuid4()
    session.add(
        MLModelScreeningRunRecord(
            id=source_id,
            experiment_key=f"{PREFIX}{source_id}"[:64],
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
            library_versions={},
            deterministic_seed=1729,
            configuration={},
            started_at=NOW,
            completed_at=NOW,
            failure_type=None,
            failure_message=None,
            evaluations_computed=1,
            evaluations_reused=0,
            results_generated=1,
            pareto_retained_count=1,
            elapsed_seconds=Decimal("1"),
            created_at=NOW,
        )
    )
    await session.flush()
    session.add(
        MLModelScreeningResultRecord(
            id=result_id,
            run_id=source_id,
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
    )
    await session.flush()
    return source_id, result_id


async def seed_campaign(
    session: AsyncSession, *, selected: bool
) -> tuple[UUID, UUID | None]:
    source_id, result_id = await seed_source(session)
    campaign_id, family_id = uuid4(), uuid4()
    request = CampaignRequest(source_screening_run_ids=(source_id,))
    session.add(
        MLSearchCampaignRecord(
            id=campaign_id,
            experiment_key=f"{PREFIX}{campaign_id}"[:64],
            status="completed",
            outcome="no_strong_candidate",
            contract_version=1,
            policy_version=1,
            classification_version=1,
            normalized_request=request.normalized_payload(),
            maximum_evaluations=request.maximum_evaluations,
            maximum_runtime_seconds=request.maximum_runtime_seconds,
            families_total=1,
            families_covered=1,
            families_uncovered=0,
            evaluations_computed=1,
            evaluations_reused=0,
            current_stage=4,
            cancellation_requested_at=None,
            first_claimed_at=NOW,
            completed_at=NOW,
            failed_at=None,
            failure_type=None,
            failure_message=None,
            created_at=NOW,
        )
    )
    await session.flush()
    session.add(
        MLSearchCampaignSourceRecord(
            campaign_id=campaign_id,
            source_screening_run_id=source_id,
            dataset_fingerprint="d" * 64,
        )
    )
    await session.flush()

    if not selected:
        session.add(
            MLSearchCampaignFamilyRecord(
                id=family_id,
                campaign_id=campaign_id,
                source_screening_run_id=source_id,
                fingerprint="f" * 64,
                dataset_fingerprint="d" * 64,
                asset="BTC/USDT",
                provider="binance",
                market_type="spot",
                interval="5m",
                engine="ridge",
                future_duration_seconds=1800,
                objective="future_return_regression",
                feature_variant="all",
                state="discarded",
                current_stage=1,
                evaluations_planned=1,
                evaluations_completed=1,
                evaluations_reused=0,
                best_evaluation_id=None,
                best_metrics=None,
                decision_reason="failed_stage_1",
                deterministic_ordinal=1,
                created_at=NOW,
            )
        )
        await session.commit()
        return campaign_id, None

    refinement_request = MLRefinementRequest.model_validate(
        {
            "source_screening_run_id": str(source_id),
            "source_screening_result_id": str(result_id),
            "engines": ["ridge"],
            "future_durations_seconds": [1800],
            "search_space": [
                {
                    "engine": "ridge",
                    "alpha": {"values": [1]},
                    "fit_intercept": {"values": [True]},
                }
            ],
            "feature_selection": {"strategies": [{"strategy": "all"}]},
            "prediction_objectives": ["future_direction_classification"],
            "confidence_buckets": ["0.05", "0.10", "0.20"],
            "fee_rate": "0.001",
            "slippage_rate": "0.0005",
            "maximum_trials": 64,
            "maximum_runtime_seconds": 3600,
            "deterministic_seed": 1729,
        }
    )
    planned = build_refinement_plan(refinement_request).evaluations[0]
    run_id, evaluation_id = uuid4(), uuid4()
    versions = {"scikit-learn": "1.0"}
    evaluation_fingerprint = effective_evaluation_fingerprint(
        dataset_fingerprint="d" * 64,
        candidate_fingerprint=planned.fingerprint,
        engine=planned.engine.value,
        versions=versions,
        deterministic_seed=1729,
    )
    session.add(
        MLRefinementRunRecord(
            id=run_id,
            experiment_key=f"{PREFIX}run-{run_id}"[:64],
            source_screening_run_id=source_id,
            source_screening_result_id=result_id,
            status="completed",
            contract_version=2,
            classification_version=2,
            dataset_fingerprint="d" * 64,
            variable_registry_version=1,
            missing_data_policy_version=1,
            library_versions=versions,
            normalized_configuration=refinement_request.normalized_payload(),
            notes=None,
            maximum_trials=64,
            maximum_runtime_seconds=3600,
            combinations_planned=1,
            evaluations_computed=1,
            evaluations_reused=0,
            evaluations_omitted=0,
            evaluations_selected=1,
            budget_exhausted=False,
            termination_reason="completed",
            started_at=NOW,
            completed_at=NOW,
            failed_at=None,
            failure_type=None,
            failure_message=None,
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
            cancellation_requested_at=None,
            elapsed_seconds=Decimal("1"),
            created_at=NOW,
        )
    )
    await session.flush()
    session.add(
        MLRefinementEvaluationRecord(
            id=evaluation_id,
            run_id=run_id,
            reused_from_evaluation_id=None,
            fingerprint=evaluation_fingerprint,
            engine="ridge",
            future_duration_seconds=1800,
            objective="future_direction_classification",
            hyperparameters=dict(planned.hyperparameters),
            feature_policy={"strategy": "all", "variables": None, "top_k": None},
            confidence_buckets=["0.05", "0.10", "0.20"],
            fold_evidence=[{"fold_name": f"fold-{index}"} for index in range(4)],
            aggregate_metrics={
                "selected_confidence_bucket": {"fraction": "0.050000000000000000"}
            },
            classification="promising_for_strategy_simulation",
            classification_version=2,
            failed_rules=[],
            elapsed_seconds=Decimal("1"),
            created_at=NOW,
        )
    )
    await session.flush()
    session.add(
        MLSearchCampaignFamilyRecord(
            id=family_id,
            campaign_id=campaign_id,
            source_screening_run_id=source_id,
            fingerprint="f" * 64,
            dataset_fingerprint="d" * 64,
            asset="BTC/USDT",
            provider="binance",
            market_type="spot",
            interval="5m",
            engine="ridge",
            future_duration_seconds=1800,
            objective="future_direction_classification",
            feature_variant="all",
            state="discarded",
            current_stage=4,
            evaluations_planned=64,
            evaluations_completed=1,
            evaluations_reused=0,
            best_evaluation_id=evaluation_id,
            best_metrics={
                "complete_and_finite": True,
                "observations": 500,
                "positive_net_folds": 5,
                "favorable_direction_folds": 5,
                "folds_beating_benchmark": 4,
                "median_net_return": "0.001",
                "worst_net_return": "-0.02",
            },
            decision_reason="failed_stage_4_economic_gate",
            deterministic_ordinal=1,
            created_at=NOW,
        )
    )
    await session.flush()
    session.add(
        MLSearchCampaignChildRecord(
            id=uuid4(),
            campaign_id=campaign_id,
            family_id=family_id,
            refinement_run_id=run_id,
            stage=4,
            reserved_evaluations=64,
            reuses_existing_run=False,
            created_at=NOW,
        )
    )
    await session.commit()
    return campaign_id, family_id


async def table_counts(session: AsyncSession) -> tuple[int, ...]:
    models = (
        MLSearchCampaignRecord,
        MLSearchCampaignFamilyRecord,
        MLSearchCampaignChildRecord,
        MLRefinementRunRecord,
        MLRefinementEvaluationRecord,
        MLFinalValidationRecord,
        PaperDeploymentRecord,
        PaperTradingRunRecord,
    )
    counts: list[int] = []
    for model in models:
        count = await session.scalar(select(func.count()).select_from(model))
        counts.append(int(count or 0))
    return tuple(counts)


async def test_empty_cohort_is_deterministic_and_read_only(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_id, _ = await seed_campaign(session, selected=False)
    monkeypatch.setattr(development_reaudit, "FAMILIES_PER_INSTRUMENT", 1)
    before = await table_counts(session)
    first, first_invalid = await build_development_reaudit_report(
        session=session, campaign_id=campaign_id
    )
    second, second_invalid = await build_development_reaudit_report(
        session=session, campaign_id=campaign_id
    )
    assert not first_invalid and not second_invalid
    assert first["counts"]["selected"] == 0  # type: ignore[index]
    assert report_json(first) == report_json(second)
    assert before == await table_counts(session)


async def test_selected_candidate_uses_valid_postgresql_graph_and_is_read_only(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_id, family_id = await seed_campaign(session, selected=True)
    assert family_id is not None
    monkeypatch.setattr(development_reaudit, "FAMILIES_PER_INSTRUMENT", 1)
    requested_ranges: list[tuple[datetime, datetime]] = []
    columns = MLCandleColumns(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0]),
    )

    async def read_candles(**kwargs: object) -> SimpleNamespace:
        start, end = kwargs["start"], kwargs["end"]
        assert isinstance(start, datetime) and isinstance(end, datetime)
        requested_ranges.append((start, end))
        return SimpleNamespace(columns=columns)

    monkeypatch.setattr(development_reaudit, "read_ml_experiment_candles", read_candles)
    dataset = object()
    monkeypatch.setattr(
        development_reaudit, "build_ml_experiment_dataset", lambda **_: dataset
    )
    monkeypatch.setattr(
        development_reaudit,
        "calculate_candidate_reaudit",
        lambda **_: {
            "status": "completed",
            "diagnostic_only": True,
            "protected_evidence_read": False,
            "development_proxy_pass": False,
            "policy_rule_source": "final_validation_policy_v1",
            "failed_rules": ("minimum_round_trips",),
            "blocks": (),
            "aggregate": {},
        },
    )
    family_before = await session.get(MLSearchCampaignFamilyRecord, family_id)
    assert family_before is not None
    persisted_metrics = dict(family_before.best_metrics or {})
    before = await table_counts(session)
    first, has_invalid = await build_development_reaudit_report(
        session=session, campaign_id=campaign_id
    )
    second, second_invalid = await build_development_reaudit_report(
        session=session, campaign_id=campaign_id
    )

    assert not has_invalid and not second_invalid
    assert first["counts"]["selected"] == 1  # type: ignore[index]
    assert first["results"][0]["family_id"] == family_id  # type: ignore[index]
    assert report_json(first) == report_json(second)
    assert requested_ranges == [
        (
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
        (
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
    ]
    assert all(end <= datetime(2024, 1, 1, tzinfo=UTC) for _, end in requested_ranges)
    assert before == await table_counts(session)
    family_after = await session.get(MLSearchCampaignFamilyRecord, family_id)
    assert family_after is not None
    assert family_after.best_metrics == persisted_metrics


async def test_structural_identity_failure_precedes_market_loading(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_id, family_id = await seed_campaign(session, selected=True)
    assert family_id is not None
    monkeypatch.setattr(development_reaudit, "FAMILIES_PER_INSTRUMENT", 1)
    family_record = await session.get(MLSearchCampaignFamilyRecord, family_id)
    assert family_record is not None and family_record.best_evaluation_id is not None
    evaluation = await session.get(
        MLRefinementEvaluationRecord, family_record.best_evaluation_id
    )
    assert evaluation is not None
    evaluation.engine = "lightgbm"
    await session.commit()
    market_loaded = False

    async def forbidden_market_load(**_: object) -> None:
        nonlocal market_loaded
        market_loaded = True
        raise AssertionError("market data must not load after identity corruption")

    monkeypatch.setattr(
        development_reaudit, "read_ml_experiment_candles", forbidden_market_load
    )
    with pytest.raises(DevelopmentReauditError) as raised:
        await build_development_reaudit_report(session=session, campaign_id=campaign_id)
    assert raised.value.code == "candidate_identity_mismatch"
    assert str(family_id) in str(raised.value)
    assert "engine" in str(raised.value)
    assert not market_loaded

