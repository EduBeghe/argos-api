from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.core.database.session import get_db_session
from app.optimization.edge_guided_read import EdgeGuidedReadNotFoundError
from app.optimization.edge_guided_schemas import (
    EdgeGuidedCandidatePageResponse,
    EdgeGuidedRunPageResponse,
)
from main import create_app


async def empty_session() -> AsyncIterator[object]:
    yield object()


def test_openapi_exposes_only_the_four_read_only_routes() -> None:
    paths = TestClient(create_app()).get("/openapi.json").json()["paths"]
    selected = {
        path: methods
        for path, methods in paths.items()
        if path.startswith("/optimization/edge-guided")
    }
    assert set(selected) == {
        "/optimization/edge-guided/runs",
        "/optimization/edge-guided/runs/{run_id}",
        "/optimization/edge-guided/candidates",
        "/optimization/edge-guided/candidates/{candidate_evaluation_id}",
    }
    assert all(set(methods) == {"get"} for methods in selected.values())


def test_empty_pages_and_query_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = AsyncMock(
        return_value=EdgeGuidedRunPageResponse(items=(), total=0, limit=50, offset=0)
    )
    candidates = AsyncMock(
        return_value=EdgeGuidedCandidatePageResponse(
            items=(), total=0, limit=50, offset=0
        )
    )
    monkeypatch.setattr("app.optimization.router.list_edge_guided_runs", runs)
    monkeypatch.setattr(
        "app.optimization.router.list_edge_guided_candidates", candidates
    )
    app = create_app()
    app.dependency_overrides[get_db_session] = empty_session
    client = TestClient(app)
    try:
        assert client.get("/optimization/edge-guided/runs").json()["items"] == []
        assert client.get("/optimization/edge-guided/candidates").json()["items"] == []
        for path in (
            "/optimization/edge-guided/runs?limit=0",
            "/optimization/edge-guided/runs?limit=101",
            "/optimization/edge-guided/runs?offset=-1",
            "/optimization/edge-guided/runs?provider=invalid",
            "/optimization/edge-guided/candidates?action=diagnostic_only",
            "/optimization/edge-guided/candidates?status=invalid",
        ):
            assert client.get(path).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_invalid_uuid_and_stable_not_found_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_run = AsyncMock(
        side_effect=EdgeGuidedReadNotFoundError("edge-guided run was not found")
    )
    missing_candidate = AsyncMock(
        side_effect=EdgeGuidedReadNotFoundError("edge-guided candidate was not found")
    )
    monkeypatch.setattr("app.optimization.router.get_edge_guided_run", missing_run)
    monkeypatch.setattr(
        "app.optimization.router.get_edge_guided_candidate", missing_candidate
    )
    app = create_app()
    app.dependency_overrides[get_db_session] = empty_session
    client = TestClient(app)
    try:
        assert (
            client.get("/optimization/edge-guided/runs/not-a-uuid").status_code == 422
        )
        unknown = UUID("00000000-0000-0000-0000-000000000099")
        response = client.get(f"/optimization/edge-guided/runs/{unknown}")
        assert response.status_code == 404
        assert response.json()["detail"] == "edge-guided run was not found"
        response = client.get(f"/optimization/edge-guided/candidates/{unknown}")
        assert response.status_code == 404
        assert response.json()["detail"] == "edge-guided candidate was not found"
    finally:
        app.dependency_overrides.clear()
