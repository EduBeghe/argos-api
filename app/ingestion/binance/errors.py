class BinanceClientError(Exception):
    """Base error for Binance client failures."""


class BinanceRequestError(BinanceClientError):
    """Raised when a Binance request is invalid before it is sent."""


class BinanceTimeoutError(BinanceClientError):
    """Raised when Binance does not respond within the configured timeout."""


class BinanceTransportError(BinanceClientError):
    """Raised when the Binance HTTP request cannot be completed."""


class BinanceHTTPError(BinanceClientError):
    """Raised when Binance returns an unsuccessful HTTP response."""

    def __init__(
        self,
        *,
        status_code: int,
        error_code: int | None,
        message: str,
    ) -> None:
        super().__init__(f"Binance returned HTTP {status_code}: {message}")
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


class BinanceResponseError(BinanceClientError):
    """Raised when Binance returns malformed kline data."""
