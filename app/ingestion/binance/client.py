from types import TracebackType
from typing import Self, TypeGuard

import httpx

from app.core.config import Settings
from app.ingestion.binance.errors import (
    BinanceHTTPError,
    BinanceRequestError,
    BinanceResponseError,
    BinanceTimeoutError,
    BinanceTransportError,
)
from app.ingestion.binance.types import BinanceKline, BinanceKlineResponse
from app.market.enums import Asset, Interval

_ASSET_SYMBOLS: dict[Asset, str] = {
    Asset.BTC_USDT: "BTCUSDT",
    Asset.ETH_USDT: "ETHUSDT",
}
_KLINE_LENGTH = 12
_INTEGER_FIELDS = frozenset({0, 6, 8})


class BinanceClient:
    """Minimal asynchronous client for public Binance Spot market data."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        timeout = httpx.Timeout(
            settings.binance_read_timeout_seconds,
            connect=settings.binance_connect_timeout_seconds,
        )
        self._client = httpx.AsyncClient(
            base_url=settings.binance_base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    async def get_klines(
        self,
        *,
        asset: Asset,
        interval: Interval,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int | None = None,
    ) -> BinanceKlineResponse:
        """Retrieve raw Binance Spot kline data."""

        self._validate_request(start_time=start_time, end_time=end_time, limit=limit)
        params: dict[str, str | int] = {
            "symbol": _ASSET_SYMBOLS[asset],
            "interval": interval.value,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit

        try:
            response = await self._client.get("/api/v3/klines", params=params)
        except httpx.TimeoutException as exc:
            raise BinanceTimeoutError("Binance kline request timed out") from exc
        except httpx.RequestError as exc:
            raise BinanceTransportError("Binance kline request failed") from exc

        if not response.is_success:
            raise self._http_error(response)

        try:
            payload: object = response.json()
        except ValueError as exc:
            raise BinanceResponseError("Binance returned invalid JSON") from exc
        if not _is_kline_response(payload):
            raise BinanceResponseError("Binance returned an invalid kline response")
        return payload

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""

        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    @staticmethod
    def _validate_request(
        *, start_time: int | None, end_time: int | None, limit: int | None
    ) -> None:
        if start_time is not None and start_time < 0:
            raise BinanceRequestError("start_time must be non-negative")
        if end_time is not None and end_time < 0:
            raise BinanceRequestError("end_time must be non-negative")
        if start_time is not None and end_time is not None and start_time > end_time:
            raise BinanceRequestError("start_time must not be after end_time")
        if limit is not None and not 1 <= limit <= 1000:
            raise BinanceRequestError("limit must be between 1 and 1000")

    @staticmethod
    def _http_error(response: httpx.Response) -> BinanceHTTPError:
        error_code: int | None = None
        message = response.text or "unsuccessful response"
        try:
            payload: object = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            code = payload.get("code")
            provider_message = payload.get("msg")
            if isinstance(code, int):
                error_code = code
            if isinstance(provider_message, str):
                message = provider_message
        return BinanceHTTPError(
            status_code=response.status_code,
            error_code=error_code,
            message=message,
        )


def _is_kline_response(payload: object) -> TypeGuard[BinanceKlineResponse]:
    if not isinstance(payload, list):
        return False
    return all(_is_kline(row) for row in payload)


def _is_kline(row: object) -> TypeGuard[BinanceKline]:
    if not isinstance(row, list) or len(row) != _KLINE_LENGTH:
        return False
    return all(
        isinstance(value, int) and not isinstance(value, bool)
        if index in _INTEGER_FIELDS
        else isinstance(value, str)
        for index, value in enumerate(row)
    )
