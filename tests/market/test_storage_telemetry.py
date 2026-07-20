from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.storage_telemetry import read_market_storage_telemetry


@pytest.mark.asyncio
async def test_storage_telemetry_explicitly_omits_non_postgresql() -> None:
    session = MagicMock(spec=AsyncSession)
    session.get_bind.return_value.dialect.name = "sqlite"
    result = await read_market_storage_telemetry(session)
    assert result.supported is False
    session.execute.assert_not_called()
