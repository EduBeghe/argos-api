from app.ml.refinement.persistence import (
    _canonical_aggregate_metrics,
    _canonical_confidence_metrics,
)

CONFIGURED = ["0.05", "0.10", "0.20"]


def test_semantic_fraction_fields_are_canonical_without_rewriting_other_metrics() -> (
    None
):
    folds = _canonical_confidence_metrics(
        [
            {
                "fraction": 0.05,
                "coverage": 0.12345678901234567,
                "observation_count": 7,
            }
        ],
        CONFIGURED,
    )
    assert folds == [
        {
            "fraction": "0.05",
            "coverage": "0.123456789012345660",
            "observation_count": 7,
        }
    ]

    aggregate = _canonical_aggregate_metrics(
        {
            "confidence_bucket_aggregates": [
                {"fraction": 0.1, "median_net_return": 0.012345678901234567}
            ],
            "selected_confidence_bucket": {
                "fraction": 0.2,
                "median_net_return": 0.012345678901234567,
            },
            "unrelated_metric": 0.012345678901234567,
        },
        CONFIGURED,
    )
    assert aggregate["confidence_bucket_aggregates"] == [
        {"fraction": "0.10", "median_net_return": "0.012345678901234567"}
    ]
    assert aggregate["selected_confidence_bucket"] == {
        "fraction": "0.20",
        "median_net_return": "0.012345678901234567",
    }
    assert aggregate["unrelated_metric"] == "0.012345678901234567"
