from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.paper.evaluation_queries import latest_snapshot_and_count


@pytest.mark.asyncio
async def test_snapshot_count_is_bounded_by_selected_authoritative_snapshot() -> None:
    boundary = datetime(2026, 7, 19, 12, tzinfo=UTC)
    run_id = uuid4()
    snapshot = SimpleNamespace(id=uuid4(), run_id=run_id, scheduled_boundary=boundary)
    scalar = AsyncMock(side_effect=(snapshot, 1))
    session = cast(AsyncSession, SimpleNamespace(scalar=scalar))

    selected, count = await latest_snapshot_and_count(session, run_id=run_id)

    assert selected is not None and selected.id == snapshot.id
    assert count == 1
    count_statement = scalar.await_args_list[1].args[0]
    compiled = count_statement.compile()
    assert "paper_portfolio_snapshots.run_id =" in str(compiled)
    assert "paper_portfolio_snapshots.scheduled_boundary <=" in str(compiled)
    assert run_id in compiled.params.values()
    assert boundary in compiled.params.values()
    assert boundary + timedelta(minutes=30) not in compiled.params.values()


@pytest.mark.asyncio
async def test_missing_snapshot_does_not_issue_independent_count_query() -> None:
    scalar = AsyncMock(return_value=None)
    session = cast(AsyncSession, SimpleNamespace(scalar=scalar))

    assert await latest_snapshot_and_count(session, run_id=uuid4()) == (None, 0)
    scalar.assert_awaited_once()
