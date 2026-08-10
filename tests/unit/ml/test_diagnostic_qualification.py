from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ml.campaign.contracts import (
    DIAGNOSTIC_QUALIFICATION_PROVENANCE,
    DiagnosticQualificationRequest,
)
from app.ml.campaign.diagnostic_qualification import (
    _stage_three_only_benchmark_miss,
)
from app.ml.campaign.economic_v2_identity import qualification_identity


def _eligible_metrics() -> dict[str, object]:
    return {
        "observations": 100,
        "round_trips": 10,
        "positive_net_folds": 4,
        "folds_beating_benchmark": 2,
        "median_net_return": str(Decimal("0.0001")),
        "worst_net_return": str(Decimal("-0.05")),
        "inactive_folds": 2,
        "complete_and_finite": True,
    }


def test_diagnostic_request_preserves_order_and_canonicalizes_identity() -> None:
    first, second = uuid4(), uuid4()
    request = DiagnosticQualificationRequest(family_ids=(second, first))

    assert request.family_ids == (second, first)
    assert request.canonical_family_ids() == tuple(sorted((first, second), key=str))
    assert request.request_version == 1


@pytest.mark.parametrize("count", [0, 11])
def test_diagnostic_request_enforces_cardinality(count: int) -> None:
    with pytest.raises(ValidationError):
        DiagnosticQualificationRequest(family_ids=tuple(uuid4() for _ in range(count)))


def test_diagnostic_request_rejects_duplicate_families() -> None:
    family_id = uuid4()
    with pytest.raises(ValidationError, match="unique UUIDs"):
        DiagnosticQualificationRequest(family_ids=(family_id, family_id))


def test_stage_three_near_miss_fails_only_benchmark_fold_requirement() -> None:
    metrics = _eligible_metrics()
    assert _stage_three_only_benchmark_miss(metrics)

    for name, value in (
        ("folds_beating_benchmark", 1),
        ("positive_net_folds", 3),
        ("round_trips", 9),
        ("median_net_return", "0"),
        ("worst_net_return", "-0.051"),
        ("inactive_folds", 3),
    ):
        incompatible = dict(metrics)
        incompatible[name] = value
        assert not _stage_three_only_benchmark_miss(incompatible)


def test_diagnostic_provenance_participates_in_identity() -> None:
    automatic = {"qualification_version": 1, "evaluation_fingerprint": "a" * 64}
    diagnostic = {
        **automatic,
        "qualification_provenance": DIAGNOSTIC_QUALIFICATION_PROVENANCE,
    }

    assert qualification_identity(automatic) != qualification_identity(diagnostic)
