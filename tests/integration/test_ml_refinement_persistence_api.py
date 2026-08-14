import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.jobs import process_ml_refinement_queue
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.campaign.contracts import (
    CAMPAIGN_CONFIDENCE_BUCKETS,
    ECONOMIC_V2_CONFIDENCE_FRACTIONS,
    ECONOMIC_V2_HOLDING_MULTIPLIERS,
    ECONOMIC_V2_REGIME_POLICIES,
    ECONOMIC_V2_STRATEGY_POLICY_VERSION,
)
from app.ml.campaign.models import MLSearchCampaignFamilyRecord, MLSearchCampaignRecord
from app.ml.contract import DEVELOPMENT_START, FINAL_TEST_END, FINAL_TEST_START
from app.ml.dataset import (
    MLDatasetMetadata,
    MLExperimentDataset,
    build_ml_experiment_dataset,
)
from app.ml.protected.models import MLFinalValidationRecord, MLTemporalManifestRecord
from app.ml.reader import candle_columns
from app.ml.refinement import service as refinement_service
from app.ml.refinement.calculation import (
    RefinementEvaluationResult,
    RefinementFoldResult,
)
from app.ml.refinement.contracts import (
    EconomicPolicySearchConfiguration,
    MLRefinementRequest,
    RefinementClassification,
)
from app.ml.refinement.identity import DatasetFingerprintChanged
from app.ml.refinement.models import (
    MLRefinementEvaluationRecord,
    MLRefinementRunRecord,
)
from app.ml.refinement.persistence import (
    MAX_PERSISTED_AGGREGATE_METRICS_BYTES,
    MLRefinementError,
    MLRefinementEvidenceTooLargeError,
    complete_run,
    effective_evaluation_fingerprint,
    existing_run_evaluation,
    experiment_key,
    fail_run,
    find_completed_run,
    persist_evaluation,
    requeue_incomplete_completed_run,
    resolve_source,
    start_run,
)
from app.ml.refinement.planning import RefinementPlan, build_refinement_plan
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


def minimal_development_dataset() -> MLExperimentDataset:
    candle_open = DEVELOPMENT_START
    return build_ml_experiment_dataset(
        instrument=Instrument(
            Asset.BTC_USDT,
            Provider.BINANCE,
            MarketType.SPOT,
            Interval.FIVE_MINUTES,
        ),
        columns=candle_columns(
            open_times=[candle_open],
            close_times=[candle_open + timedelta(minutes=5, microseconds=-1)],
            open_prices=[100.0],
            high_prices=[101.0],
            low_prices=[99.0],
            close_prices=[100.5],
            volumes=[10.0],
        ),
        requested_end=FINAL_TEST_START,
    )


def completed_manifest(evaluation_id: UUID) -> MLTemporalManifestRecord:
    return MLTemporalManifestRecord(
        id=uuid4(),
        evaluation_id=evaluation_id,
        source="persisted_at_execution",
        status="completed",
        manifest_version=1,
        split_policy_version=1,
        dataset_fingerprint="d" * 64,
        manifest_fingerprint="m" * 64,
        manifest={},
        fingerprint_verified=True,
        counts_verified=True,
        worker_id=None,
        claimed_at=None,
        heartbeat_at=None,
        cancellation_requested_at=None,
        failure_code=None,
        failure_message=None,
        created_at=NOW,
        completed_at=NOW,
    )


def protected_campaign_graph(
    source_run_id: UUID,
) -> tuple[MLSearchCampaignRecord, MLSearchCampaignFamilyRecord]:
    campaign_id = uuid4()
    campaign = MLSearchCampaignRecord(
        id=campaign_id,
        experiment_key=f"{PREFIX}protected-{campaign_id}"[:64],
        status="completed",
        outcome="no_strong_candidate",
        contract_version=1,
        policy_version=1,
        classification_version=1,
        normalized_request={},
        maximum_evaluations=1,
        maximum_runtime_seconds=60,
        families_total=1,
        families_covered=1,
        families_uncovered=0,
        evaluations_computed=1,
        evaluations_reused=0,
        current_stage=1,
        cancellation_requested_at=None,
        first_claimed_at=NOW,
        completed_at=NOW,
        failed_at=None,
        failure_type=None,
        failure_message=None,
        created_at=NOW,
    )
    family_id = uuid4()
    family = MLSearchCampaignFamilyRecord(
        id=family_id,
        campaign_id=campaign_id,
        source_screening_run_id=source_run_id,
        fingerprint=f"{family_id.hex:0<64}"[:64],
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
        selected_policy=None,
        selected_policy_fingerprint=None,
        decision_reason="test",
        deterministic_ordinal=0,
        created_at=NOW,
    )
    return campaign, family


async def persisted_incomplete_run(
    session: AsyncSession,
    refinement_request: MLRefinementRequest,
    *,
    out_of_plan_ordinals: set[int],
) -> tuple[
    MLRefinementRunRecord,
    RefinementPlan,
    tuple[MLRefinementEvaluationRecord, ...],
]:
    plan = build_refinement_plan(refinement_request)
    versions = library_versions()
    run = await start_run(
        session=session,
        request=refinement_request,
        plan=plan,
        key=experiment_key(
            request=refinement_request,
            dataset_fingerprint="d" * 64,
            versions=versions,
        ),
        dataset_fingerprint="d" * 64,
        versions=versions,
        now=NOW,
    )
    for candidate in plan.evaluations:
        result = RefinementEvaluationResult(
            candidate,
            (),
            {"fold_count": 0},
            RefinementClassification.INCOMPLETE,
            ("incomplete_fold_count",),
            0.1,
        )
        await persist_evaluation(
            session=session,
            run_id=run.id,
            fingerprint=effective_evaluation_fingerprint(
                dataset_fingerprint="d" * 64,
                candidate_fingerprint=candidate.fingerprint,
                engine=candidate.engine.value,
                versions=versions,
                deterministic_seed=refinement_request.deterministic_seed,
            ),
            result=result,
            reused=None,
            request=refinement_request,
            now=NOW,
        )
    rows = tuple(
        (
            await session.scalars(
                select(MLRefinementEvaluationRecord)
                .where(MLRefinementEvaluationRecord.run_id == run.id)
                .order_by(MLRefinementEvaluationRecord.created_at)
            )
        ).all()
    )
    for ordinal in out_of_plan_ordinals:
        rows[ordinal].fingerprint = f"{ordinal + 1}" * 64
    await complete_run(
        session,
        run.id,
        computed=0,
        reused=0,
        omitted=0,
        budget_exhausted=False,
        reason="interrupted_with_stale_evidence",
        elapsed=1.0,
        now=NOW,
    )
    return run, plan, rows


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
        prior_refinement_runs = select(MLRefinementRunRecord.id).where(
            MLRefinementRunRecord.source_screening_run_id.in_(prior_source_runs)
        )
        await session.execute(
            delete(MLTemporalManifestRecord).where(
                MLTemporalManifestRecord.evaluation_id.in_(
                    select(MLRefinementEvaluationRecord.id).where(
                        MLRefinementEvaluationRecord.run_id.in_(prior_refinement_runs)
                    )
                )
            )
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
        refinement_runs = select(MLRefinementRunRecord.id).where(
            MLRefinementRunRecord.source_screening_run_id == run_id
        )
        await session.execute(
            delete(MLTemporalManifestRecord).where(
                MLTemporalManifestRecord.evaluation_id.in_(
                    select(MLRefinementEvaluationRecord.id).where(
                        MLRefinementEvaluationRecord.run_id.in_(refinement_runs)
                    )
                )
            )
        )
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


async def test_evaluation_parent_precedes_manifest_and_rollback_retry_is_atomic(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    plan = build_refinement_plan(refinement_request)
    dataset = minimal_development_dataset()
    fingerprint = "f" * 64
    target = dataset.targets.for_duration(timedelta(minutes=30))
    eligible = dataset.feature_eligible & target.eligible
    result = RefinementEvaluationResult(
        plan.evaluations[0],
        tuple(
            RefinementFoldResult(
                fold.definition.name,
                int(eligible[fold.training_indices].sum()),
                int(eligible[fold.validation_indices].sum()),
                (),
                {},
                (),
                0.0,
                "insufficient_training_rows",
            )
            for fold in dataset.folds
        ),
        {"fold_count": len(dataset.folds)},
        RefinementClassification.INCOMPLETE,
        ("incomplete_fold_count",),
        0.1,
    )
    async with factory() as session:
        run = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key="a" * 64,
            dataset_fingerprint="d" * 64,
            versions=library_versions(),
            now=NOW,
        )
        run_id = run.id

        def fail_after_parent_flush(**_kwargs: object) -> object:
            raise RuntimeError("forced dependent evidence failure")

        with monkeypatch.context() as patch:
            patch.setattr(
                "app.ml.refinement.persistence.reconstruct_temporal_manifest",
                fail_after_parent_flush,
            )
            with pytest.raises(RuntimeError, match="forced dependent evidence failure"):
                await persist_evaluation(
                    session=session,
                    run_id=run_id,
                    fingerprint=fingerprint,
                    result=result,
                    reused=None,
                    request=refinement_request,
                    now=NOW,
                    development_dataset=dataset,
                    canonical_dataset_fingerprint="d" * 64,
                )
        await session.rollback()

    async with factory() as verification:
        assert (
            await verification.scalar(
                select(func.count(MLRefinementEvaluationRecord.id)).where(
                    MLRefinementEvaluationRecord.run_id == run_id
                )
            )
            == 0
        )
        assert (
            await verification.scalar(
                select(func.count(MLTemporalManifestRecord.id))
                .join(
                    MLRefinementEvaluationRecord,
                    MLRefinementEvaluationRecord.id
                    == MLTemporalManifestRecord.evaluation_id,
                )
                .where(MLRefinementEvaluationRecord.run_id == run_id)
            )
            == 0
        )

    async with factory() as session:
        await persist_evaluation(
            session=session,
            run_id=run_id,
            fingerprint=fingerprint,
            result=result,
            reused=None,
            request=refinement_request,
            now=NOW,
            development_dataset=dataset,
            canonical_dataset_fingerprint="d" * 64,
        )
        retained = await existing_run_evaluation(session, run_id, fingerprint)
        assert retained is not None

    async with factory() as verification:
        evaluation_ids = tuple(
            await verification.scalars(
                select(MLRefinementEvaluationRecord.id).where(
                    MLRefinementEvaluationRecord.run_id == run_id
                )
            )
        )
        assert len(evaluation_ids) == 1
        manifests = tuple(
            await verification.scalars(
                select(MLTemporalManifestRecord).where(
                    MLTemporalManifestRecord.evaluation_id == evaluation_ids[0]
                )
            )
        )
        assert len(manifests) == 1
        assert manifests[0].status == "completed"
        assert manifests[0].dataset_fingerprint == "d" * 64


async def test_invalid_evaluation_source_writes_no_rows(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    plan = build_refinement_plan(refinement_request)
    async with factory() as session:
        run = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key="i" * 64,
            dataset_fingerprint="d" * 64,
            versions=library_versions(),
            now=NOW,
        )
        with pytest.raises(
            ValueError, match="exactly one evaluation source is required"
        ):
            await persist_evaluation(
                session=session,
                run_id=run.id,
                fingerprint="n" * 64,
                result=None,
                reused=None,
                request=refinement_request,
                now=NOW,
            )
        assert (
            await session.scalar(
                select(func.count(MLRefinementEvaluationRecord.id)).where(
                    MLRefinementEvaluationRecord.run_id == run.id
                )
            )
            == 0
        )


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


async def test_persistence_canonicalizes_new_bucket_identity_but_copies_legacy_reuse(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    payload = request(source_run_id, source_result_id).model_dump(mode="json")
    payload["confidence_buckets"] = ["0.05", "0.10", "0.20"]
    refinement_request = MLRefinementRequest.model_validate(payload)
    plan = build_refinement_plan(refinement_request)
    versions = library_versions()
    async with factory() as session:
        first_run = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key="c" * 64,
            dataset_fingerprint="d" * 64,
            versions=versions,
            now=NOW,
        )
        result = RefinementEvaluationResult(
            plan.evaluations[0],
            (
                RefinementFoldResult(
                    "fold-1",
                    100,
                    50,
                    (),
                    {"unrelated_metric": 0.12345678901234567},
                    (
                        {
                            "fraction": 0.05,
                            "coverage": 0.12345678901234567,
                        },
                    ),
                    0.003,
                ),
            ),
            {
                "confidence_bucket_aggregates": (
                    {"fraction": 0.1, "median_net_return": 0.012345678901234567},
                ),
                "selected_confidence_bucket": {
                    "fraction": 0.2,
                    "median_net_return": 0.012345678901234567,
                },
                "unrelated_metric": 0.012345678901234567,
            },
            RefinementClassification.INCOMPLETE,
            ("incomplete_fold_count",),
            0.1,
        )
        await persist_evaluation(
            session=session,
            run_id=first_run.id,
            fingerprint="p" * 64,
            result=result,
            reused=None,
            request=refinement_request,
            now=NOW,
        )
        persisted = await session.scalar(
            select(MLRefinementEvaluationRecord).where(
                MLRefinementEvaluationRecord.run_id == first_run.id
            )
        )
        assert persisted is not None
        assert persisted.fold_evidence[0]["confidence_metrics"] == [
            {"fraction": "0.05", "coverage": "0.123456789012345660"}
        ]
        aggregates = persisted.aggregate_metrics["confidence_bucket_aggregates"]
        assert isinstance(aggregates, list)
        assert aggregates[0]["fraction"] == "0.10"
        selected = persisted.aggregate_metrics["selected_confidence_bucket"]
        assert isinstance(selected, dict)
        assert selected["fraction"] == "0.20"
        assert persisted.aggregate_metrics["unrelated_metric"] == (
            "0.012345678901234567"
        )

        selected["fraction"] = "0.200000000000000000"
        legacy_metrics = dict(persisted.aggregate_metrics)
        legacy_metrics["selected_confidence_bucket"] = selected
        persisted.aggregate_metrics = legacy_metrics
        await session.commit()
        second_run = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key="r" * 64,
            dataset_fingerprint="d" * 64,
            versions=versions,
            now=NOW,
        )
        await persist_evaluation(
            session=session,
            run_id=second_run.id,
            fingerprint="unused-for-reuse",
            result=None,
            reused=persisted,
            request=refinement_request,
            now=NOW,
        )
        copied = await session.scalar(
            select(MLRefinementEvaluationRecord).where(
                MLRefinementEvaluationRecord.run_id == second_run.id
            )
        )
        assert copied is not None
        copied_selected = copied.aggregate_metrics["selected_confidence_bucket"]
        assert isinstance(copied_selected, dict)
        assert copied_selected["fraction"] == "0.200000000000000000"


async def test_v2_first_evaluation_persists_with_separate_confidence_contracts(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    base = request(source_run_id, source_result_id)
    refinement_request = base.model_copy(
        update={
            "confidence_buckets": CAMPAIGN_CONFIDENCE_BUCKETS,
            "economic_policy_search": EconomicPolicySearchConfiguration(
                confidence_fractions=ECONOMIC_V2_CONFIDENCE_FRACTIONS,
                holding_multipliers=ECONOMIC_V2_HOLDING_MULTIPLIERS,
                regime_policies=ECONOMIC_V2_REGIME_POLICIES,
                version=ECONOMIC_V2_STRATEGY_POLICY_VERSION,
            ),
        }
    )
    plan = build_refinement_plan(refinement_request)
    result = RefinementEvaluationResult(
        plan.evaluations[0],
        (
            RefinementFoldResult(
                "fold-1",
                100,
                50,
                (),
                {},
                tuple(
                    {"fraction": float(fraction), "coverage": 0.5}
                    for fraction in CAMPAIGN_CONFIDENCE_BUCKETS
                ),
                0.01,
            ),
        ),
        {
            "selected_confidence_bucket": {
                "fraction": 0.2,
                "median_net_return": 0.01,
            }
        },
        RefinementClassification.INCOMPLETE,
        ("incomplete_fold_count",),
        0.1,
    )
    async with factory() as session:
        run = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key="v" * 64,
            dataset_fingerprint="d" * 64,
            versions=library_versions(),
            now=NOW,
        )
        await persist_evaluation(
            session=session,
            run_id=run.id,
            fingerprint="w" * 64,
            result=result,
            reused=None,
            request=refinement_request,
            now=NOW,
        )
        persisted = await session.scalar(
            select(MLRefinementEvaluationRecord).where(
                MLRefinementEvaluationRecord.run_id == run.id
            )
        )
        assert persisted is not None
        confidence_metrics = persisted.fold_evidence[0]["confidence_metrics"]
        assert isinstance(confidence_metrics, list)
        assert [
            item["fraction"] for item in confidence_metrics if isinstance(item, dict)
        ] == ["0.05", "0.10", "0.20"]
        assert refinement_request.economic_policy_search is not None
        assert refinement_request.economic_policy_search.confidence_fractions == (
            Decimal("0.05"),
            Decimal("0.10"),
            Decimal("0.20"),
            Decimal("0.30"),
            Decimal("0.40"),
        )


class SentinelRefinementError(RuntimeError):
    pass


async def test_worker_failure_uses_captured_run_id_across_expiring_commit(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    engine = cast(AsyncEngine, factory.kw["bind"])
    expiring_factory = async_sessionmaker(engine, expire_on_commit=True)
    async with expiring_factory() as session:
        await enqueue_refinement(session, refinement_request, now=NOW)
        run_id = await session.scalar(
            select(MLRefinementRunRecord.id).where(
                MLRefinementRunRecord.source_screening_result_id == source_result_id
            )
        )
        assert run_id is not None

    monkeypatch.setattr(
        process_ml_refinement_queue,
        "fail_stale_protected_work",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "claim_protected_work",
        AsyncMock(side_effect=[None, None, None, None]),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "fail_stale_qualifications",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "claim_next_qualification",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "coordinate_next_campaign",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "fail_stale_runs",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue, "validate_engine_dependencies", lambda: None
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "load_refinement_identity_metadata",
        AsyncMock(
            return_value=SimpleNamespace(
                instrument=Instrument(
                    Asset.BTC_USDT,
                    Provider.BINANCE,
                    MarketType.SPOT,
                    Interval.FIVE_MINUTES,
                )
            )
        ),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "validate_dataset_identity",
        lambda **_kwargs: SimpleNamespace(structured_fields=lambda: {}),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "load_ml_development_dataset",
        AsyncMock(return_value=object()),
    )
    monkeypatch.setattr(
        process_ml_refinement_queue,
        "execute_refinement",
        AsyncMock(side_effect=SentinelRefinementError("sentinel primary failure")),
    )
    caplog.set_level(logging.ERROR)

    assert await process_ml_refinement_queue.process_once(
        expiring_factory,
        worker_id="expire-worker",
        stale_after=timedelta(hours=2),
    )
    async with expiring_factory() as verification:
        persisted = await verification.get(MLRefinementRunRecord, run_id)
        assert persisted is not None
        assert persisted.status == "failed"
        assert persisted.failure_type == "SentinelRefinementError"
        assert persisted.failure_message == "sentinel primary failure"
        assert persisted.worker_id is None
        assert persisted.claimed_at is None
        assert persisted.heartbeat_at is None
    failure_record = next(
        item
        for item in caplog.records
        if item.message == "ML refinement queue item failed"
    )
    assert failure_record.structured_fields == {  # type: ignore[attr-defined]
        "run_id": str(run_id),
        "worker_id": "expire-worker",
        "failure_type": "SentinelRefinementError",
    }
    assert not await process_ml_refinement_queue.process_once(
        expiring_factory,
        worker_id="expire-worker",
        stale_after=timedelta(hours=2),
    )


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
    dataset = minimal_development_dataset()
    instrument = dataset.metadata.instrument
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


async def test_persistence_compacts_economic_policy_folds_and_enforces_size_limit(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    plan = build_refinement_plan(refinement_request)
    versions = library_versions()
    async with factory() as session:
        run = await start_run(
            session=session,
            request=refinement_request,
            plan=plan,
            key="z" * 64,
            dataset_fingerprint="d" * 64,
            versions=versions,
            now=NOW,
        )
        candidate = plan.evaluations[0]
        duplicated_fold = {
            "policy_fingerprint": "p" * 64,
            "net_return": "0.01",
            "trades": [{"payload": "x" * 600_000}],
            "equity_curve": [["2039-01-01T00:00:00+00:00", "1"]] * 20_000,
        }
        aggregate_metrics = {
            "fold_count": 7,
            "economic_policy_v2_aggregates": [
                {
                    "policy_fingerprint": "p" * 64,
                    "positive_net_folds": 7,
                    "folds": [duplicated_fold, duplicated_fold],
                }
            ],
            "selected_economic_policy_v2": {
                "policy_fingerprint": "p" * 64,
                "positive_net_folds": 7,
                "folds": [duplicated_fold, duplicated_fold],
            },
        }
        result = RefinementEvaluationResult(
            candidate,
            (),
            aggregate_metrics,
            RefinementClassification.PROMISING,
            (),
            0.1,
        )
        await persist_evaluation(
            session=session,
            run_id=run.id,
            fingerprint="e" * 64,
            result=result,
            reused=None,
            request=refinement_request,
            now=NOW,
        )
        persisted = await existing_run_evaluation(session, run.id, "e" * 64)
        assert persisted is not None
        aggregates = persisted.aggregate_metrics["economic_policy_v2_aggregates"]
        assert isinstance(aggregates, list)
        assert "folds" not in aggregates[0]
        selected = persisted.aggregate_metrics["selected_economic_policy_v2"]
        assert isinstance(selected, dict)
        assert "folds" not in selected
        assert (
            len(
                json.dumps(
                    persisted.aggregate_metrics,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            <= MAX_PERSISTED_AGGREGATE_METRICS_BYTES
        )

        oversized = RefinementEvaluationResult(
            candidate,
            (),
            {"unrelated_payload": "x" * (MAX_PERSISTED_AGGREGATE_METRICS_BYTES + 1)},
            RefinementClassification.INCOMPLETE,
            ("incomplete_fold_count",),
            0.1,
        )
        with pytest.raises(
            MLRefinementEvidenceTooLargeError,
            match="compacted aggregate metrics exceed",
        ):
            await persist_evaluation(
                session=session,
                run_id=run.id,
                fingerprint="o" * 64,
                result=oversized,
                reused=None,
                request=refinement_request,
                now=NOW,
            )
        assert await existing_run_evaluation(session, run.id, "o" * 64) is None


async def test_incomplete_completed_enqueue_requeues_same_run_concurrently(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    async with factory() as session:
        queued = await enqueue_refinement(session, refinement_request, now=NOW)
        run_id = queued.run.id
        queued.run.status = "completed"
        queued.run.started_at = NOW
        queued.run.evaluations_computed = queued.run.evaluations_selected
        queued.run.completed_at = NOW
        await session.commit()

    async def enqueue_once() -> tuple[UUID, QueueDisposition]:
        async with factory() as session:
            result = await enqueue_refinement(session, refinement_request, now=NOW)
            return result.run.id, result.disposition

    first, second = await asyncio.gather(enqueue_once(), enqueue_once())
    assert first[0] == second[0] == run_id
    assert {first[1], second[1]} == {
        QueueDisposition.REQUEUED_INCOMPLETE,
        QueueDisposition.ALREADY_QUEUED,
    }
    async with factory() as session:
        recovered = await session.get(MLRefinementRunRecord, run_id)
        assert recovered is not None
        assert recovered.status == "queued"
        assert recovered.completed_at is None
        assert recovered.evaluations_computed == 0
        assert recovered.evaluations_reused == 0


async def test_incomplete_recovery_preserves_compatible_evidence_and_rolls_back(
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
        request=refinement_request,
        dataset_fingerprint="d" * 64,
        versions=versions,
    )
    async with factory() as session:
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
        fingerprint = effective_evaluation_fingerprint(
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
            0.1,
        )
        await persist_evaluation(
            session=session,
            run_id=run.id,
            fingerprint=fingerprint,
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
            reason="interrupted_after_partial_persistence",
            elapsed=1.0,
            now=NOW,
        )

    async with factory() as session:
        locked = await session.scalar(
            select(MLRefinementRunRecord)
            .where(MLRefinementRunRecord.id == run.id)
            .with_for_update()
        )
        assert locked is not None
        recovery = await requeue_incomplete_completed_run(
            session=session,
            run=locked,
            request=refinement_request,
            plan=plan,
            dataset_fingerprint="d" * 64,
            versions=versions,
            caller_context="rollback_test",
        )
        assert recovery.preserved_count == 1
        assert recovery.deleted_incompatible_count == 0
        await session.rollback()

    async with factory() as session:
        unchanged = await session.get(MLRefinementRunRecord, run.id)
        assert unchanged is not None
        assert unchanged.status == "completed"
        assert unchanged.evaluations_computed == 1
        assert (
            await session.scalar(
                select(func.count(MLRefinementEvaluationRecord.id)).where(
                    MLRefinementEvaluationRecord.run_id == run.id
                )
            )
            == 1
        )

        recovered = await enqueue_refinement(session, refinement_request, now=NOW)
        assert recovered.disposition is QueueDisposition.REQUEUED_INCOMPLETE
        assert recovered.run.id == run.id
        assert recovered.run.evaluations_selected == 2
        assert recovered.run.evaluations_computed == 0
        assert recovered.run.evaluations_reused == 0
        assert (
            await session.scalar(
                select(func.count(MLRefinementEvaluationRecord.id)).where(
                    MLRefinementEvaluationRecord.run_id == run.id
                )
            )
            == 1
        )

async def test_recovery_deletes_only_unprotected_out_of_plan_evidence(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    payload = request(source_run_id, source_result_id).model_dump(mode="json")
    payload["search_space"][0]["alpha"] = {"values": [1.0, 2.0]}
    payload["maximum_trials"] = 2
    refinement_request = MLRefinementRequest.model_validate(payload)
    async with factory() as session:
        run, plan, rows = await persisted_incomplete_run(
            session, refinement_request, out_of_plan_ordinals={1}
        )
        compatible_id, out_of_plan_id = rows[0].id, rows[1].id

    async with factory() as session:
        locked = await session.scalar(
            select(MLRefinementRunRecord)
            .where(MLRefinementRunRecord.id == run.id)
            .with_for_update()
        )
        assert locked is not None
        recovery = await requeue_incomplete_completed_run(
            session=session,
            run=locked,
            request=refinement_request,
            plan=plan,
            dataset_fingerprint="d" * 64,
            versions=library_versions(),
            caller_context="out_of_plan_test",
        )
        assert recovery.deleted_out_of_plan_count == 1
        assert recovery.preserved_count == 1
        await session.commit()

    async with factory() as session:
        recovered = await session.get(MLRefinementRunRecord, run.id)
        assert recovered is not None and recovered.status == "queued"
        assert (
            await session.get(MLRefinementEvaluationRecord, compatible_id) is not None
        )
        assert await session.get(MLRefinementEvaluationRecord, out_of_plan_id) is None


async def test_temporal_manifest_protects_recovery_candidate_losslessly(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    async with factory() as session:
        run, plan, rows = await persisted_incomplete_run(
            session, refinement_request, out_of_plan_ordinals={0}
        )
        manifest = completed_manifest(rows[0].id)
        session.add(manifest)
        await session.commit()
        evaluation_id, manifest_id = rows[0].id, manifest.id

    async with factory() as session:
        locked = await session.scalar(
            select(MLRefinementRunRecord)
            .where(MLRefinementRunRecord.id == run.id)
            .with_for_update()
        )
        assert locked is not None
        with pytest.raises(
            MLRefinementError,
            match="incomplete refinement evidence is protected",
        ):
            await requeue_incomplete_completed_run(
                session=session,
                run=locked,
                request=refinement_request,
                plan=plan,
                dataset_fingerprint="d" * 64,
                versions=library_versions(),
                caller_context="manifest_test",
            )
        await session.rollback()

    async with factory() as session:
        unchanged = await session.get(MLRefinementRunRecord, run.id)
        assert unchanged is not None and unchanged.status == "completed"
        assert (
            await session.get(MLRefinementEvaluationRecord, evaluation_id) is not None
        )
        assert await session.get(MLTemporalManifestRecord, manifest_id) is not None


async def test_final_validation_protects_complete_audit_graph(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    campaign, family = protected_campaign_graph(source_run_id)
    async with factory() as session:
        run, plan, rows = await persisted_incomplete_run(
            session, refinement_request, out_of_plan_ordinals={0}
        )
        manifest = completed_manifest(rows[0].id)
        validation = MLFinalValidationRecord(
            id=uuid4(),
            experiment_key=f"{PREFIX}validation-{uuid4()}"[:64],
            campaign_id=campaign.id,
            family_id=family.id,
            evaluation_id=rows[0].id,
            temporal_manifest_id=manifest.id,
            status="completed",
            classification="rejected",
            frozen_inputs={},
            execution_manifest={},
            scenario_metrics={},
            failed_rules=[],
            protected_data_read_at=NOW,
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
            cancellation_requested_at=None,
            failure_code=None,
            failure_message=None,
            created_at=NOW,
            completed_at=NOW,
        )
        session.add(campaign)
        await session.flush()
        session.add(family)
        await session.flush()
        session.add(manifest)
        await session.flush()
        session.add(validation)
        await session.commit()
        evaluation_id, manifest_id, validation_id = (
            rows[0].id,
            manifest.id,
            validation.id,
        )

    try:
        async with factory() as session:
            locked = await session.scalar(
                select(MLRefinementRunRecord)
                .where(MLRefinementRunRecord.id == run.id)
                .with_for_update()
            )
            assert locked is not None
            with pytest.raises(
                MLRefinementError,
                match="incomplete refinement evidence is protected",
            ):
                await requeue_incomplete_completed_run(
                    session=session,
                    run=locked,
                    request=refinement_request,
                    plan=plan,
                    dataset_fingerprint="d" * 64,
                    versions=library_versions(),
                    caller_context="final_validation_test",
                )
            await session.rollback()

        async with factory() as session:
            unchanged = await session.get(MLRefinementRunRecord, run.id)
            assert unchanged is not None and unchanged.status == "completed"
            assert (
                await session.get(MLRefinementEvaluationRecord, evaluation_id)
                is not None
            )
            assert await session.get(MLTemporalManifestRecord, manifest_id) is not None
            assert await session.get(MLFinalValidationRecord, validation_id) is not None
    finally:
        async with factory() as session:
            await session.execute(
                delete(MLFinalValidationRecord).where(
                    MLFinalValidationRecord.id == validation_id
                )
            )
            await session.execute(
                delete(MLSearchCampaignRecord).where(
                    MLSearchCampaignRecord.id == campaign.id
                )
            )
            await session.commit()


async def test_one_protected_candidate_blocks_all_recovery_deletions(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    payload = request(source_run_id, source_result_id).model_dump(mode="json")
    payload["search_space"][0]["alpha"] = {"values": [1.0, 2.0]}
    payload["maximum_trials"] = 2
    refinement_request = MLRefinementRequest.model_validate(payload)
    async with factory() as session:
        run, plan, rows = await persisted_incomplete_run(
            session, refinement_request, out_of_plan_ordinals={0, 1}
        )
        manifest = completed_manifest(rows[1].id)
        session.add(manifest)
        await session.commit()
        evaluation_ids = {row.id for row in rows}

    async with factory() as session:
        locked = await session.scalar(
            select(MLRefinementRunRecord)
            .where(MLRefinementRunRecord.id == run.id)
            .with_for_update()
        )
        assert locked is not None
        with pytest.raises(MLRefinementError, match="evidence is protected"):
            await requeue_incomplete_completed_run(
                session=session,
                run=locked,
                request=refinement_request,
                plan=plan,
                dataset_fingerprint="d" * 64,
                versions=library_versions(),
                caller_context="mixed_test",
            )
        await session.rollback()

    async with factory() as session:
        unchanged = await session.get(MLRefinementRunRecord, run.id)
        persisted_ids = set(
            await session.scalars(
                select(MLRefinementEvaluationRecord.id).where(
                    MLRefinementEvaluationRecord.run_id == run.id
                )
            )
        )
        assert unchanged is not None and unchanged.status == "completed"
        assert persisted_ids == evaluation_ids


async def test_protected_reference_insert_serializes_with_recovery(
    refinement_factory: tuple[async_sessionmaker[AsyncSession], UUID, UUID],
) -> None:
    factory, source_run_id, source_result_id = refinement_factory
    refinement_request = request(source_run_id, source_result_id)
    async with factory() as session:
        run, plan, rows = await persisted_incomplete_run(
            session, refinement_request, out_of_plan_ordinals={0}
        )
        evaluation_id = rows[0].id

    async with factory() as reference_session:
        manifest = completed_manifest(evaluation_id)
        reference_session.add(manifest)
        await reference_session.flush()

        async def recover_concurrently() -> None:
            async with factory() as recovery_session:
                locked = await recovery_session.scalar(
                    select(MLRefinementRunRecord)
                    .where(MLRefinementRunRecord.id == run.id)
                    .with_for_update()
                )
                assert locked is not None
                with pytest.raises(
                    MLRefinementError,
                    match="incomplete refinement evidence is protected",
                ):
                    await requeue_incomplete_completed_run(
                        session=recovery_session,
                        run=locked,
                        request=refinement_request,
                        plan=plan,
                        dataset_fingerprint="d" * 64,
                        versions=library_versions(),
                        caller_context="concurrency_test",
                    )
                await recovery_session.rollback()

        recovery_task = asyncio.create_task(recover_concurrently())
        await asyncio.sleep(0.05)
        assert not recovery_task.done()
        await reference_session.commit()
        await recovery_task

    async with factory() as session:
        unchanged = await session.get(MLRefinementRunRecord, run.id)
        assert unchanged is not None and unchanged.status == "completed"
        assert (
            await session.get(MLRefinementEvaluationRecord, evaluation_id) is not None
        )
        assert await session.get(MLTemporalManifestRecord, manifest.id) is not None
