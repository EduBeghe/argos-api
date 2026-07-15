class BinanceKlineNormalizationError(ValueError):
    """Base error for Binance kline normalization failures."""


class BinanceKlineStructureError(BinanceKlineNormalizationError):
    """Raised when a raw kline does not have Binance's positional structure."""


class BinanceKlineFieldError(BinanceKlineNormalizationError):
    """Raised when a raw kline field has an invalid provider value."""

    def __init__(self, *, index: int, field: str, message: str) -> None:
        super().__init__(f"Invalid Binance kline {field} at index {index}: {message}")
        self.index = index
        self.field = field


class BinanceKlineDomainError(BinanceKlineNormalizationError):
    """Raised when normalized values violate the MarketCandle contract."""
