from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.persistence import MarketCandleRecord, to_record
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        await database_session.execute(delete(MarketCandleRecord))
        await database_session.commit()
        yield database_session
        await database_session.rollback()
        await database_session.execute(delete(MarketCandleRecord))
        await database_session.commit()
    await engine.dispose()


async def test_duplicate_natural_identity_is_rejected(session: AsyncSession) -> None:
    session.add(to_record(make_candle()))
    await session.commit()
    session.add(to_record(make_candle()))

    with pytest.raises(IntegrityError):
        await session.commit()


async def test_invalid_database_value_is_rejected(session: AsyncSession) -> None:
    record = to_record(make_candle())
    record.open_price = Decimal("0")
    session.add(record)

    with pytest.raises(IntegrityError):
        await session.commit()


async def test_invalid_categorical_value_is_rejected(session: AsyncSession) -> None:
    record = to_record(make_candle())
    record.provider = "unsupported"
    session.add(record)

    with pytest.raises(IntegrityError):
        await session.commit()
