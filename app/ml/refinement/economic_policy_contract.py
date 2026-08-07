from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .contracts import EconomicPolicySearchConfiguration, MLRefinementRequest


@dataclass(frozen=True, slots=True)
class EconomicPolicyCompatibility:
    compatible: bool
    reason_code: str
    expected_contract: str
    found_contract: str


def canonical_economic_policy_configuration(
    configuration: EconomicPolicySearchConfiguration,
) -> dict[str, object]:
    """Return the order-independent scientific identity of one policy grid."""

    return {
        "version": configuration.version,
        "confidence_fractions": sorted(
            str(value) for value in configuration.confidence_fractions
        ),
        "holding_multipliers": sorted(
            str(value) for value in configuration.holding_multipliers
        ),
        "regime_policies": sorted(configuration.regime_policies),
    }


def _decimal(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def economic_policy_evidence_compatibility(
    *,
    aggregate_metrics: dict[str, object],
    request: MLRefinementRequest,
) -> EconomicPolicyCompatibility:
    configuration = request.economic_policy_search
    selected = aggregate_metrics.get("selected_economic_policy_v2")
    if configuration is None:
        if selected is not None:
            return EconomicPolicyCompatibility(
                False,
                "legacy_request_has_v2_evidence",
                "legacy",
                "economic_policy_v2",
            )
        return EconomicPolicyCompatibility(True, "compatible", "legacy", "legacy")

    expected = f"economic_policy_v{configuration.version}"
    if not isinstance(selected, dict) or not selected:
        return EconomicPolicyCompatibility(
            False,
            "missing_selected_economic_policy",
            expected,
            "missing",
        )
    version = selected.get("strategy_policy_version")
    found = f"economic_policy_v{version}" if version is not None else "unversioned"
    if version != configuration.version:
        return EconomicPolicyCompatibility(
            False, "economic_policy_version_mismatch", expected, found
        )
    confidence = _decimal(selected.get("confidence_fraction"))
    multiplier = _decimal(selected.get("holding_multiplier"))
    regime = selected.get("regime_policy")
    if (
        confidence not in configuration.confidence_fractions
        or multiplier not in configuration.holding_multipliers
        or not isinstance(regime, str)
        or regime not in configuration.regime_policies
    ):
        return EconomicPolicyCompatibility(
            False, "selected_economic_policy_outside_grid", expected, found
        )
    fingerprint = selected.get("policy_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return EconomicPolicyCompatibility(
            False, "invalid_selected_economic_policy", expected, found
        )
    return EconomicPolicyCompatibility(True, "compatible", expected, found)
