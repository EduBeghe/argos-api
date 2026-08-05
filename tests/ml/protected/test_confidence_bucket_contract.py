from decimal import Decimal

import pytest

from app.ml.protected.contracts import (
    SUPPORTED_CONFIDENCE_BUCKET_FRACTIONS,
    ConfidenceBucketContractError,
    canonicalize_confidence_bucket,
    legacy_confidence_bucket_representation,
)

CONFIGURED = ["0.05", "0.10", "0.20"]


@pytest.mark.parametrize("expected", sorted(SUPPORTED_CONFIDENCE_BUCKET_FRACTIONS))
def test_canonical_and_exact_legacy_representations_resolve_to_configured_identity(
    expected: Decimal,
) -> None:
    assert canonicalize_confidence_bucket(str(expected), CONFIGURED) == expected
    assert (
        canonicalize_confidence_bucket(
            legacy_confidence_bucket_representation(expected), CONFIGURED
        )
        == expected
    )


@pytest.mark.parametrize(
    "selected",
    [
        None,
        True,
        False,
        0.05,
        Decimal("0.05"),
        "",
        "not-a-number",
        "NaN",
        "Infinity",
        "-Infinity",
        "0",
        "-0.05",
        "0.15",
        "0.050000000000000001",
        "0.049999999999999999",
    ],
)
def test_invalid_or_unknown_selected_representations_fail_closed(
    selected: object,
) -> None:
    with pytest.raises(ConfidenceBucketContractError):
        canonicalize_confidence_bucket(selected, CONFIGURED)


@pytest.mark.parametrize(
    "configured",
    [
        None,
        [],
        ["0.05", "0.05"],
        ["0.05", "0.050000000000000000"],
        ["0.05", "0.15"],
        ["0.05", "NaN"],
        ["0.05", True],
        [Decimal("0.05")],
    ],
)
def test_invalid_unsupported_or_duplicate_configuration_fails_closed(
    configured: object,
) -> None:
    with pytest.raises(ConfidenceBucketContractError):
        canonicalize_confidence_bucket("0.05", configured)


def test_result_is_deterministic_independent_of_configured_order() -> None:
    legacy = legacy_confidence_bucket_representation(Decimal("0.10"))
    assert canonicalize_confidence_bucket(
        legacy, CONFIGURED
    ) == canonicalize_confidence_bucket(legacy, list(reversed(CONFIGURED)))
