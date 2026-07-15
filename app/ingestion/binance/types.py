"""Provider-specific Binance response types."""

type BinanceKlineValue = int | str
type BinanceKline = list[BinanceKlineValue]
type BinanceKlineResponse = list[BinanceKline]
