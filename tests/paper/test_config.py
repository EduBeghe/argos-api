from decimal import Decimal

import pytest

from app.paper.config import MAX_INITIAL_CASH, validate_initial_cash


@pytest.mark.parametrize(
    "value",
    [
        Decimal("0"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("Infinity"),
        MAX_INITIAL_CASH + 1,
    ],
)
def test_invalid_initial_cash_is_rejected(value: Decimal) -> None:
    with pytest.raises(ValueError):
        validate_initial_cash(value)


def test_initial_cash_is_explicitly_quantized() -> None:
    assert validate_initial_cash(Decimal("10000.1234567890123456789")) == Decimal(
        "10000.123456789012345679"
    )
