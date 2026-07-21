from collections.abc import AsyncIterator
from unittest.mock import ANY, AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.database.session import get_db_session
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.experience import EvidenceClassification
from app.optimization.read_service import OptimizationReadNotFoundError
from app.optimization.schemas import (
    CandidatePageResponse,
    EvaluationPageResponse,
    WalkForwardCandidatePageResponse,
    WalkForwardWindowEvaluationPageResponse,
)
from app.optimization.walk_forward import WalkForwardClassification
from app.optimization.walk_forward_read import WalkForwardReadNotFoundError
from main import app


async def fake_session() -> AsyncIterator[object]:
    yield object()


def test_run_collection_passes_typed_filters_and_bounded_pagination(
    client: TestClient,
) -> None:
    reader = AsyncMock(return_value=())
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.optimization.router.read_refinement_runs", reader):
            response = client.get(
                "/optimization/refinement-runs",
                params={
                    "asset": "BTC/USDT",
                    "provider": "binance",
                    "market_type": "spot",
                    "interval": "5m",
                    "status": "completed",
                    "recommendation_status": "recommended_and_test_confirmed",
                    "limit": 25,
                    "offset": 50,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []
    awaited = reader.await_args
    assert awaited is not None
    query = awaited.args[1]
    assert query.asset is Asset.BTC_USDT
    assert query.provider is Provider.BINANCE
    assert query.market_type is MarketType.SPOT
    assert query.interval is Interval.FIVE_MINUTES
    assert query.limit == 25
    assert query.offset == 50


def test_candidate_collection_uses_versioned_classification_filter(
    client: TestClient,
) -> None:
    reader = AsyncMock(
        return_value=CandidatePageResponse(items=(), has_more=False, next_cursor=None)
    )
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.optimization.router.read_candidates", reader):
            response = client.get(
                "/optimization/candidates",
                params={
                    "evidence_classification": "robust_candidate",
                    "minimum_validation_runs": 3,
                    "limit": 10,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"items": [], "has_more": False, "next_cursor": None}
    awaited = reader.await_args
    assert awaited is not None
    query = awaited.args[1]
    assert query.evidence_classification is EvidenceClassification.ROBUST_CANDIDATE
    assert query.minimum_validation_runs == 3


def test_collection_limit_is_explicitly_bounded(client: TestClient) -> None:
    response = client.get("/optimization/refinement-runs", params={"limit": 201})
    assert response.status_code == 422


def test_unknown_run_and_candidate_return_clear_404(client: TestClient) -> None:
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch(
            "app.optimization.router.read_refinement_run",
            AsyncMock(
                side_effect=OptimizationReadNotFoundError("refinement run not found")
            ),
        ):
            run_response = client.get(f"/optimization/refinement-runs/{uuid4()}")
        with patch(
            "app.optimization.router.read_candidate",
            AsyncMock(
                side_effect=OptimizationReadNotFoundError(
                    "optimization candidate not found"
                )
            ),
        ):
            candidate_response = client.get("/optimization/candidates/missing")
    finally:
        app.dependency_overrides.clear()

    assert run_response.status_code == 404
    assert run_response.json() == {"detail": "refinement run not found"}
    assert candidate_response.status_code == 404
    assert candidate_response.json() == {"detail": "optimization candidate not found"}


def test_candidate_history_pagination_is_forwarded(client: TestClient) -> None:
    reader = AsyncMock(
        side_effect=OptimizationReadNotFoundError("optimization candidate not found")
    )
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.optimization.router.read_candidate", reader):
            response = client.get(
                "/optimization/candidates/missing",
                params={"limit": 20, "cursor": "cursor-value"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    reader.assert_awaited_once_with(ANY, "missing", limit=20, cursor="cursor-value")


def test_run_evaluations_use_bounded_cursor_pagination(client: TestClient) -> None:
    run_id = uuid4()
    cursor = uuid4()
    reader = AsyncMock(
        return_value=EvaluationPageResponse(items=(), has_more=False, next_cursor=None)
    )
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.optimization.router.read_run_evaluations", reader):
            response = client.get(
                f"/optimization/refinement-runs/{run_id}/evaluations",
                params={"limit": 25, "cursor": str(cursor)},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"items": [], "has_more": False, "next_cursor": None}
    reader.assert_awaited_once_with(ANY, run_id, limit=25, cursor=str(cursor))

    too_large = client.get(
        f"/optimization/refinement-runs/{run_id}/evaluations",
        params={"limit": 201},
    )
    assert too_large.status_code == 422


def test_walk_forward_candidate_filters_and_cursor_are_forwarded(
    client: TestClient,
) -> None:
    run_id = uuid4()
    reader = AsyncMock(
        return_value=WalkForwardCandidatePageResponse(
            items=(), has_more=False, next_cursor=None
        )
    )
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.optimization.router.read_walk_forward_candidates", reader):
            response = client.get(
                f"/optimization/walk-forward-runs/{run_id}/candidates",
                params={
                    "classification": "stable_candidate",
                    "limit": 25,
                    "cursor": "2:candidate",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"items": [], "has_more": False, "next_cursor": None}
    reader.assert_awaited_once_with(
        ANY,
        run_id,
        classification=WalkForwardClassification.STABLE_CANDIDATE,
        limit=25,
        cursor="2:candidate",
    )


def test_walk_forward_window_evidence_is_bounded_and_unknown_is_404(
    client: TestClient,
) -> None:
    run_id = uuid4()
    page_reader = AsyncMock(
        return_value=WalkForwardWindowEvaluationPageResponse(
            items=(), has_more=False, next_cursor=None
        )
    )
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch(
            "app.optimization.router.read_walk_forward_candidate_windows",
            page_reader,
        ):
            response = client.get(
                f"/optimization/walk-forward-runs/{run_id}/candidates/candidate/windows",
                params={"limit": 2, "cursor": "3"},
            )
        with patch(
            "app.optimization.router.read_walk_forward_run",
            AsyncMock(side_effect=WalkForwardReadNotFoundError("not found")),
        ):
            missing = client.get(f"/optimization/walk-forward-runs/{uuid4()}")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    page_reader.assert_awaited_once_with(ANY, run_id, "candidate", limit=2, cursor="3")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "not found"}

    too_large = client.get(
        f"/optimization/walk-forward-runs/{run_id}/candidates/candidate/windows",
        params={"limit": 201},
    )
    assert too_large.status_code == 422

