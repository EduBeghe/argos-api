"""Shared offset-pagination plumbing for the optimization search variants.

Only the mechanical "count rows matching these filters" query is extracted here.
Ordering, filter construction, and the response envelope shape (`total`+`offset` vs.
`next_cursor`+`has_more` vs. keyset cursors) remain genuinely different between
variants and are deliberately left to each `*_read.py` module to build.
"""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase


async def count_matching[ModelT: DeclarativeBase](
    session: AsyncSession, model: type[ModelT], *filters: Any
) -> int:
    """Return the number of `model` rows matching `filters` (0 when none match).

    Equivalent to the repeated `int(await session.scalar(select(func.count())
    .select_from(model).where(*filters)) or 0)` block in `composite_search_read.py`
    and `targeted_refinement_read.py`.
    """
    total = await session.scalar(
        select(func.count()).select_from(model).where(*filters)
    )
    return int(total or 0)
