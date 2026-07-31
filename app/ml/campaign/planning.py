import hashlib
import json
from dataclasses import dataclass

from app.ml.contract import FUTURE_DURATIONS
from app.ml.refinement.contracts import PredictionObjective
from app.ml.screening.config import ScreeningEngine

from .contracts import (
    CAMPAIGN_CONTRACT_VERSION,
    CAMPAIGN_FEE_RATE,
    CAMPAIGN_POLICY_VERSION,
    CAMPAIGN_SEED,
    CAMPAIGN_SLIPPAGE_RATE,
    FAMILIES_PER_INSTRUMENT,
    FeatureVariant,
)


@dataclass(frozen=True, slots=True)
class SourceInstrument:
    source_run_id: str
    asset: str
    provider: str
    market_type: str
    interval: str
    dataset_fingerprint: str

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return self.asset, self.provider, self.market_type, self.interval


@dataclass(frozen=True, slots=True)
class CampaignFamily:
    fingerprint: str
    source: SourceInstrument
    engine: ScreeningEngine
    future_duration_seconds: int
    objective: PredictionObjective
    feature_variant: FeatureVariant


def _family_fingerprint(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_family_universe(
    sources: tuple[SourceInstrument, ...],
) -> tuple[CampaignFamily, ...]:
    if len({item.identity for item in sources}) != len(sources):
        raise ValueError("source instrument identities must be unique")
    ordered_sources = tuple(
        sorted(sources, key=lambda item: (*item.identity, item.source_run_id))
    )
    by_source: list[list[CampaignFamily]] = []
    for source in ordered_sources:
        families: list[CampaignFamily] = []
        for engine in ScreeningEngine:
            for duration in FUTURE_DURATIONS:
                seconds = int(duration.total_seconds())
                for objective in PredictionObjective:
                    for feature in FeatureVariant:
                        payload: dict[str, object] = {
                            "contract_version": CAMPAIGN_CONTRACT_VERSION,
                            "policy_version": CAMPAIGN_POLICY_VERSION,
                            "source_identity": source.identity,
                            "source_run_id": source.source_run_id,
                            "dataset_fingerprint": source.dataset_fingerprint,
                            "engine": engine.value,
                            "future_duration_seconds": seconds,
                            "objective": objective.value,
                            "feature_variant": feature.value,
                            "fee_rate": str(CAMPAIGN_FEE_RATE),
                            "slippage_rate": str(CAMPAIGN_SLIPPAGE_RATE),
                            "seed": CAMPAIGN_SEED,
                        }
                        families.append(
                            CampaignFamily(
                                _family_fingerprint(payload),
                                source,
                                engine,
                                seconds,
                                objective,
                                feature,
                            )
                        )
        assert len(families) == FAMILIES_PER_INSTRUMENT
        by_source.append(families)
    # Breadth-first deterministic interleaving prevents instrument starvation.
    return tuple(
        family
        for ordinal in range(FAMILIES_PER_INSTRUMENT)
        for families in by_source
        for family in (families[ordinal],)
    )
