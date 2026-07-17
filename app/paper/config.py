from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

CONFIGURATION_VERSION = 1
DEFAULT_INITIAL_CASH = Decimal("10000.00")
DEFAULT_ACCOUNT_CURRENCY = "USDT"
FEE_RATE = Decimal("0.001")
SLIPPAGE_RATE = Decimal("0.0005")
FINANCIAL_QUANTUM = Decimal("0.000000000000000001")
MAX_INITIAL_CASH = Decimal("10000000000000000000")


def quantize_financial(value: Decimal) -> Decimal:
    """Quantize persisted financial values to NUMERIC(38, 18), half-even."""
    return value.quantize(FINANCIAL_QUANTUM, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True, slots=True)
class SimulationConfiguration:
    version: int = CONFIGURATION_VERSION
    account_currency: str = DEFAULT_ACCOUNT_CURRENCY
    fee_rate: Decimal = FEE_RATE
    slippage_rate: Decimal = SLIPPAGE_RATE

    def __post_init__(self) -> None:
        if self.version <= 0 or not self.account_currency.strip():
            raise ValueError("configuration version and currency must be valid")
        for rate in (self.fee_rate, self.slippage_rate):
            if not rate.is_finite() or not Decimal("0") <= rate < Decimal("1"):
                raise ValueError("simulation rates must be finite and in [0, 1)")


VERSION_1 = SimulationConfiguration()


def validate_initial_cash(value: Decimal) -> Decimal:
    if not value.is_finite() or value <= 0 or value > MAX_INITIAL_CASH:
        raise ValueError(
            "initial cash must be finite, positive, and within the safe maximum"
        )
    return quantize_financial(value)
