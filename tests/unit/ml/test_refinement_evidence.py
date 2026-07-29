
import pytest

from app.ml.refinement.contracts import RefinementClassification
from app.ml.refinement.evidence import RefinementEvidence, classify_refinement_evidence


def test_promising_requires_consistent_economic_evidence() -> None:
    classification, failed = classify_refinement_evidence(
        RefinementEvidence(7, 5, 5, 0.001, 50)
    )
    assert classification is RefinementClassification.PROMISING
    assert failed == ()


@pytest.mark.parametrize(
    "evidence",
    [
        RefinementEvidence(7, 3, 5, 0.001, 50),
        RefinementEvidence(7, 5, 3, 0.001, 50),
        RefinementEvidence(7, 5, 5, 0, 50),
        RefinementEvidence(7, 5, 5, 0.001, 29),
    ],
)
def test_each_promising_gate_is_centralized(evidence: RefinementEvidence) -> None:
    classification, failed = classify_refinement_evidence(evidence)
    assert classification is not RefinementClassification.PROMISING
    assert failed


def test_incomplete_fold_evidence_is_incomplete() -> None:
    classification, _ = classify_refinement_evidence(
        RefinementEvidence(6, 6, 6, 1, 100)
    )
    assert classification is RefinementClassification.INCOMPLETE
