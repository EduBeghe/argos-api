from typing import cast

import httpx
import pytest

from app.core.config import Settings
from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.errors import (
    BinanceHTTPError,
    BinanceRequestError,
    BinanceResponseError,
    BinanceTimeoutError,
    BinanceTransportError,
)
from app.market.enums import Asset, Interval

pytestmark = pytest.mark.asyncio

KLINE: list[int | str] = [
    1499040000000,
    "0.01634790",
    "0.80000000",
    "0.01575800",
    "0.01577100",
    "148976.11427815",
    1499644799999,
    "2434.19055334",
    308,
    "1756.87402397",
    "28.46694368",
    "0",
]


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "DATABASE_URL": "postgresql+asyncpg://user:secret@localhost/argos",
        "BINANCE_BASE_URL": "https://binance.test",
        **overrides,
    }
    return Settings.model_validate(values)


async def test_get_klines_sends_complete_contract_and_returns_raw_data() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v3/klines"
        assert dict(request.url.params) == {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "startTime": "1000",
            "endTime": "2000",
            "limit": "1000",
        }
        assert "X-MBX-APIKEY" not in request.headers
        return httpx.Response(200, json=[KLINE])

    async with BinanceClient(
        make_settings(), transport=httpx.MockTransport(handler)
    ) as client:
        response = await client.get_klines(
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            start_time=1000,
            end_time=2000,
            limit=1000,
        )

    assert response == [KLINE]


async def test_get_klines_omits_optional_parameters() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == {"symbol": "ETHUSDT", "interval": "5m"}
        return httpx.Response(200, json=[])

    async with BinanceClient(
        make_settings(), transport=httpx.MockTransport(handler)
    ) as client:
        response = await client.get_klines(
            asset=Asset.ETH_USDT,
            interval=Interval.FIVE_MINUTES,
        )

    assert response == []


async def test_get_klines_uses_separate_configured_timeouts() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        timeout = cast(dict[str, float], request.extensions["timeout"])
        assert timeout["connect"] == 2.0
        assert timeout["read"] == 7.0
        return httpx.Response(200, json=[])

    settings = make_settings(
        BINANCE_CONNECT_TIMEOUT_SECONDS=2,
        BINANCE_READ_TIMEOUT_SECONDS=7,
    )
    async with BinanceClient(
        settings, transport=httpx.MockTransport(handler)
    ) as client:
        await client.get_klines(
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"start_time": -1}, "start_time"),
        ({"end_time": -1}, "end_time"),
        ({"start_time": 2, "end_time": 1}, "start_time"),
        ({"limit": 0}, "limit"),
        ({"limit": 1001}, "limit"),
    ],
)
async def test_get_klines_rejects_invalid_request(
    arguments: dict[str, int], message: str
) -> None:
    async with BinanceClient(
        make_settings(), transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as client:
        with pytest.raises(BinanceRequestError, match=message):
            await client.get_klines(
                asset=Asset.BTC_USDT,
                interval=Interval.FIVE_MINUTES,
                **arguments,
            )


async def test_get_klines_exposes_binance_http_error() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
    )
    async with BinanceClient(make_settings(), transport=transport) as client:
        with pytest.raises(BinanceHTTPError) as captured:
            await client.get_klines(
                asset=Asset.BTC_USDT,
                interval=Interval.FIVE_MINUTES,
            )

    assert captured.value.status_code == 400
    assert captured.value.error_code == -1121
    assert captured.value.message == "Invalid symbol."


async def test_get_klines_exposes_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    async with BinanceClient(
        make_settings(), transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(BinanceTimeoutError):
            await client.get_klines(
                asset=Asset.BTC_USDT,
                interval=Interval.FIVE_MINUTES,
            )


async def test_get_klines_exposes_transport_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    async with BinanceClient(
        make_settings(), transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(BinanceTransportError):
            await client.get_klines(
                asset=Asset.BTC_USDT,
                interval=Interval.FIVE_MINUTES,
            )


@pytest.mark.parametrize(
    "payload", [{}, [KLINE[:-1]], [[*KLINE[:8], "308", *KLINE[9:]]]]
)
async def test_get_klines_rejects_malformed_response(payload: object) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    async with BinanceClient(make_settings(), transport=transport) as client:
        with pytest.raises(BinanceResponseError):
            await client.get_klines(
                asset=Asset.BTC_USDT,
                interval=Interval.FIVE_MINUTES,
            )


async def test_get_klines_rejects_invalid_json() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"not-json"))
    async with BinanceClient(make_settings(), transport=transport) as client:
        with pytest.raises(BinanceResponseError, match="invalid JSON"):
            await client.get_klines(
                asset=Asset.BTC_USDT,
                interval=Interval.FIVE_MINUTES,
            )
