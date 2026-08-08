import hashlib
import json
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


def _fingerprint(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def economic_policy_search_contract(
    configuration: EconomicPolicySearchConfiguration,
) -> dict[str, object]:
    canonical = canonical_economic_policy_configuration(configuration)
    return {"configuration": canonical, "fingerprint": _fingerprint(canonical)}


def economic_policy_fingerprint(
    *,
    confidence_fraction: Decimal,
    holding_multiplier: Decimal,
    regime_policy: str,
    holding_candles: int,
    version: int,
) -> str:
    return _fingerprint(
        {
            "confidence_fraction": str(confidence_fraction),
            "holding_candles": holding_candles,
            "holding_multiplier": str(holding_multiplier),
            "regime_policy": regime_policy,
            "version": version,
        }
    )


def _decimal(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _policy_identity(item: object) -> tuple[object, ...] | None:
    if not isinstance(item, dict):
        return None
    confidence = _decimal(item.get("confidence_fraction"))
    multiplier = _decimal(item.get("holding_multiplier"))
    regime = item.get("regime_policy")
    version = item.get("strategy_policy_version")
    candles = item.get("holding_candles")
    fingerprint = item.get("policy_fingerprint")
    if (
        confidence is None
        or multiplier is None
        or not isinstance(regime, str)
        or not isinstance(version, int)
        or not isinstance(candles, int)
        or not isinstance(fingerprint, str)
    ):
        return None
    expected = economic_policy_fingerprint(
        confidence_fraction=confidence,
        holding_multiplier=multiplier,
        regime_policy=regime,
        holding_candles=candles,
        version=version,
    )
    if fingerprint != expected:
        return None
    return (fingerprint, confidence, multiplier, regime, candles, version)


def _expected_policy_identities(
    configuration: EconomicPolicySearchConfiguration,
    future_duration_seconds: int,
) -> set[tuple[object, ...]]:
    identities: set[tuple[object, ...]] = set()
    for confidence in configuration.confidence_fractions:
        for multiplier in configuration.holding_multipliers:
            candles = Decimal(future_duration_seconds) * multiplier / Decimal(300)
            integral = candles.to_integral_value()
            if candles != integral or integral <= 0:
                return set()
            for regime in configuration.regime_policies:
                holding_candles = int(integral)
                fingerprint = economic_policy_fingerprint(
                    confidence_fraction=confidence,
                    holding_multiplier=multiplier,
                    regime_policy=regime,
                    holding_candles=holding_candles,
                    version=configuration.version,
                )
                identities.add(
                    (
                        fingerprint,
                        confidence,
                        multiplier,
                        regime,
                        holding_candles,
                        configuration.version,
                    )
                )
    return identities


def economic_policy_evidence_compatibility(
    *,
    aggregate_metrics: dict[str, object],
    request: MLRefinementRequest,
    future_duration_seconds: int | None = None,
) -> EconomicPolicyCompatibility:
    configuration = request.economic_policy_search
    selected = aggregate_metrics.get("selected_economic_policy_v2")
    provenance = aggregate_metrics.get("economic_policy_search_contract")
    if configuration is None:
        if selected is not None or provenance is not None:
            return EconomicPolicyCompatibility(
                False,
                "legacy_request_has_v2_evidence",
                "legacy",
                "economic_policy_v2",
            )
        return EconomicPolicyCompatibility(True, "compatible", "legacy", "legacy")

    expected_contract = economic_policy_search_contract(configuration)
    expected_fingerprint = str(expected_contract["fingerprint"])
    if not isinstance(provenance, dict):
        return EconomicPolicyCompatibility(
            False,
            "missing_economic_policy_contract",
            expected_fingerprint,
            "missing",
        )
    found_fingerprint = provenance.get("fingerprint")
    found_configuration = provenance.get("configuration")
    if not isinstance(found_configuration, dict) or not isinstance(
        found_fingerprint, str
    ):
        return EconomicPolicyCompatibility(
            False,
            "invalid_economic_policy_contract",
            expected_fingerprint,
            str(found_fingerprint or "invalid"),
        )
    if (
        _fingerprint(found_configuration) != found_fingerprint
        or provenance != expected_contract
    ):
        return EconomicPolicyCompatibility(
            False,
            "economic_policy_contract_mismatch",
            expected_fingerprint,
            found_fingerprint,
        )

    aggregates = aggregate_metrics.get("economic_policy_v2_aggregates")
    if not isinstance(aggregates, (list, tuple)):
        return EconomicPolicyCompatibility(
            False,
            "invalid_economic_policy_coverage",
            expected_fingerprint,
            found_fingerprint,
        )
    identities = tuple(_policy_identity(item) for item in aggregates)
    expected_identities = (
        _expected_policy_identities(configuration, future_duration_seconds)
        if future_duration_seconds is not None
        else set()
    )
    if (
        not expected_identities
        or any(item is None for item in identities)
        or len(identities) != len(expected_identities)
        or set(identities) != expected_identities
    ):
        return EconomicPolicyCompatibility(
            False,
            "invalid_economic_policy_coverage",
            expected_fingerprint,
            found_fingerprint,
        )
    assert all(item is not None for item in identities)
    selected_identity = _policy_identity(selected)
    if selected_identity is None or selected_identity not in identities:
        return EconomicPolicyCompatibility(
            False,
            "invalid_selected_economic_policy",
            expected_fingerprint,
            found_fingerprint,
        )
    return EconomicPolicyCompatibility(
        True, "compatible", expected_fingerprint, found_fingerprint
    )
