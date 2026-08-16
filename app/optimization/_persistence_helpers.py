"""Shared, behavior-preserving persistence plumbing for the optimization variants.

This module exists to remove infrastructure duplication that accumulated across
`composite_search_memory.py`, `targeted_refinement_memory.py`, and the read services
built on top of them — not to unify the search variants themselves. Each variant's
research algorithm, gate thresholds, and reuse-eligibility rules remain independent
and must stay that way; only the mechanical plumbing below was ever byte-for-byte
identical across files.

`edge_guided_memory.py` has its own `_json` helper (built on
`json.dumps(value, default=str)`) that is *not* equivalent to `to_json_safe` below —
it formats datetimes and enums differently — so it intentionally keeps its own
implementation rather than importing this one.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase


def to_json_safe(value: Any) -> Any:
    """Recursively coerce a value into JSON-serializable primitives for JSONB columns.

    Decimal -> str, datetime -> isoformat, Enum -> `.value`, timedelta -> whole
    seconds, tuple/list -> list, dict -> dict with coerced values and string keys.
    Equivalent to the identical private `_json` helpers this replaces in
    `composite_search_memory.py` and `targeted_refinement_memory.py`.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (tuple, list)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "total_seconds"):
        return int(value.total_seconds())
    return value


async def replace_by_column[ModelT: DeclarativeBase](
    session: AsyncSession,
    model: type[ModelT],
    column: Any,
    value: UUID | str,
) -> None:
    """Delete any existing row where `column == value` so a fresh insert can replace it.

    Implements the idempotent "find the row keyed by this identity, delete it if
    present, flush" sequence that `persist_composite_search_result` and
    `persist_targeted_refinement_result` both performed independently, keyed on
    `predictive_edge_run_id` in both callers. Callers that need extra concurrency
    hardening (row locking, integrity-error retry) — `edge_guided_memory.py` does —
    should not use this helper.
    """
    existing = await session.scalar(select(model).where(column == value))
    if existing is not None:
        await session.execute(delete(model).where(model.id == existing.id))  # type: ignore[attr-defined]
        await session.flush()


def require[RecordT](value: RecordT | None, error: Exception) -> RecordT:
    """Return `value` unchanged, or raise `error` if it is `None`.

    A tiny, deliberately generic replacement for the repeated
    `if record is None: raise SomeNotFoundError(...)` guard found in every
    `*_read.py` module in this package.
    """
    if value is None:
        raise error
    return value
