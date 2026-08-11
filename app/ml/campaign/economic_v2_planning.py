import hashlib
import json
from dataclasses import dataclass
from uuid import UUID

from app.ml.refinement.contracts import PredictionObjective
from app.ml.screening.config import ScreeningEngine

from .contracts import (
    CAMPAIGN_FEE_RATE,
    CAMPAIGN_SLIPPAGE_RATE,
    ECONOMIC_V2_CONTRACT_VERSION,
    ECONOMIC_V2_DIVERSIFIED_FAMILIES,
    ECONOMIC_V2_FEATURE_VARIANTS,
    ECONOMIC_V2_HORIZONS_SECONDS,
    ECONOMIC_V2_MAX_FAMILIES_PER_SOURCE,
    ECONOMIC_V2_POLICY_VERSION,
    ECONOMIC_V2_SEED,
    FeatureVariant,
)
from .planning import SourceInstrument


@dataclass(frozen=True, slots=True)
class SeedFamily:
    id: UUID
    source_run_id: str
    engine: ScreeningEngine
    future_duration_seconds: int
    objective: PredictionObjective
    feature_variant: FeatureVariant
    evaluation_fingerprint: str
    hyperparameters: tuple[tuple[str, bool | int | float], ...]


@dataclass(frozen=True, slots=True)
class EconomicV2Family:
    fingerprint: str
    source: SourceInstrument
    search_track: str
    seed_family_id: UUID | None
    also_seeded_by: tuple[UUID, ...]
    engine: ScreeningEngine
    future_duration_seconds: int
    objective: PredictionObjective
    feature_variant: FeatureVariant

    @property
    def scientific_identity(self) -> tuple[object, ...]:
        return (
            self.source.source_run_id,
            self.engine.value,
            self.future_duration_seconds,
            self.objective.value,
            self.feature_variant.value,
        )


def _fingerprint(
    *,
    source: SourceInstrument,
    engine: ScreeningEngine,
    duration: int,
    objective: PredictionObjective,
    feature: FeatureVariant,
    track: str,
    seed_id: UUID | None,
) -> str:
    payload = {
        "contract_version": ECONOMIC_V2_CONTRACT_VERSION,
        "policy_version": ECONOMIC_V2_POLICY_VERSION,
        "preset": "economic_policy_search_v2",
        "source_identity": source.identity,
        "source_run_id": source.source_run_id,
        "dataset_fingerprint": source.dataset_fingerprint,
        "engine": engine.value,
        "future_duration_seconds": duration,
        "objective": objective.value,
        "feature_variant": feature.value,
        "fee_rate": str(CAMPAIGN_FEE_RATE),
        "slippage_rate": str(CAMPAIGN_SLIPPAGE_RATE),
        "seed": ECONOMIC_V2_SEED,
        "search_track": track,
        "seed_family_id": str(seed_id) if seed_id is not None else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _family(
    source: SourceInstrument,
    engine: ScreeningEngine,
    duration: int,
    objective: PredictionObjective,
    feature: FeatureVariant,
    *,
    track: str,
    seed_id: UUID | None = None,
    also_seeded_by: tuple[UUID, ...] = (),
) -> EconomicV2Family:
    return EconomicV2Family(
        _fingerprint(
            source=source,
            engine=engine,
            duration=duration,
            objective=objective,
            feature=feature,
            track=track,
            seed_id=seed_id,
        ),
        source,
        track,
        seed_id,
        also_seeded_by,
        engine,
        duration,
        objective,
        feature,
    )


def build_economic_v2_universe(
    sources: tuple[SourceInstrument, ...], seeds: tuple[SeedFamily, ...]
) -> tuple[EconomicV2Family, ...]:
    if len({source.identity for source in sources}) != len(sources):
        raise ValueError("source instrument identities must be unique")
    ordered_sources = tuple(
        sorted(sources, key=lambda item: (*item.identity, item.source_run_id))
    )
    ordered_seeds = tuple(sorted(seeds, key=lambda item: item.id))
    by_source: list[list[EconomicV2Family]] = []
    for source in ordered_sources:
        diversified = [
            _family(
                source,
                engine,
                duration,
                objective,
                FeatureVariant(feature),
                track="diversified",
            )
            for engine in ScreeningEngine
            for duration in ECONOMIC_V2_HORIZONS_SECONDS
            for objective in PredictionObjective
            for feature in ECONOMIC_V2_FEATURE_VARIANTS
        ]
        assert len(diversified) == ECONOMIC_V2_DIVERSIFIED_FAMILIES
        identities = {
            item.scientific_identity: index for index, item in enumerate(diversified)
        }
        local: list[EconomicV2Family] = []
        for seed in ordered_seeds:
            if seed.source_run_id != source.source_run_id:
                continue
            for offset in (-43_200, -21_600, 0, 21_600, 43_200):
                duration = seed.future_duration_seconds + offset
                if not 3_600 <= duration <= 7 * 86_400:
                    continue
                candidate = _family(
                    source,
                    seed.engine,
                    duration,
                    seed.objective,
                    seed.feature_variant,
                    track="seeded_local",
                    seed_id=seed.id,
                    also_seeded_by=(seed.id,),
                )
                duplicate = identities.get(candidate.scientific_identity)
                if offset == 0 and duplicate is not None:
                    existing = diversified[duplicate]
                    diversified[duplicate] = EconomicV2Family(
                        existing.fingerprint,
                        existing.source,
                        existing.search_track,
                        existing.seed_family_id,
                        tuple(sorted((*existing.also_seeded_by, seed.id))),
                        existing.engine,
                        existing.future_duration_seconds,
                        existing.objective,
                        existing.feature_variant,
                    )
                    continue
                local_duplicate = next(
                    (
                        index
                        for index, item in enumerate(local)
                        if item.scientific_identity == candidate.scientific_identity
                    ),
                    None,
                )
                if local_duplicate is not None:
                    existing = local[local_duplicate]
                    local[local_duplicate] = EconomicV2Family(
                        existing.fingerprint,
                        existing.source,
                        existing.search_track,
                        existing.seed_family_id,
                        tuple(sorted({*existing.also_seeded_by, seed.id})),
                        existing.engine,
                        existing.future_duration_seconds,
                        existing.objective,
                        existing.feature_variant,
                    )
                    continue
                local.append(candidate)
        by_source.append((diversified + local)[:ECONOMIC_V2_MAX_FAMILIES_PER_SOURCE])
    maximum = max((len(items) for items in by_source), default=0)
    return tuple(
        items[ordinal]
        for ordinal in range(maximum)
        for items in by_source
        if ordinal < len(items)
    )
