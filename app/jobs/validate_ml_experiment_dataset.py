import argparse
import asyncio
import logging
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.market.catalog import InstrumentCapability, resolve_instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.contract import (
    DEVELOPMENT_START,
    FINAL_TEST_END,
    FINAL_TEST_START,
    FUTURE_DURATIONS,
    ML_EXPERIMENT_CONTRACT_VERSION,
    temporal_fold_definitions,
)
from app.ml.dataset import MLExperimentDataset, build_ml_experiment_dataset
from app.ml.reader import read_ml_experiment_candles
from app.ml.registry import MAXIMUM_HISTORICAL_LOOKBACK, VARIABLE_REGISTRY

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MLValidationSummary:
    dataset: MLExperimentDataset | None
    elapsed_seconds: float
    approximate_peak_bytes: int


def _approximate_peak_resident_bytes(traced_peak: int) -> int:
    """Use OS peak RSS when available, otherwise report traced Python memory."""

    try:
        status = Path("/proc/self/status").read_text(encoding="ascii")
    except OSError:
        return traced_peak
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return traced_peak


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen ML experiment dataset contract"
    )
    parser.add_argument("--asset", required=True, type=Asset)
    parser.add_argument("--contract-only", action="store_true")
    return parser


def _contract_fields() -> dict[str, object]:
    return {
        "contract_version": ML_EXPERIMENT_CONTRACT_VERSION,
        "requested_range": [DEVELOPMENT_START.isoformat(), FINAL_TEST_END.isoformat()],
        "final_test_range": [FINAL_TEST_START.isoformat(), FINAL_TEST_END.isoformat()],
        "feature_count": len(VARIABLE_REGISTRY),
        "features": [item.name for item in VARIABLE_REGISTRY],
        "future_duration_seconds": [
            int(item.total_seconds()) for item in FUTURE_DURATIONS
        ],
        "folds": [
            {
                "name": item.name,
                "training": [
                    item.training_start.isoformat(),
                    item.training_end.isoformat(),
                ],
                "validation": [
                    item.validation_start.isoformat(),
                    item.validation_end.isoformat(),
                ],
            }
            for item in temporal_fold_definitions()
        ],
        "maximum_historical_lookback_seconds": int(
            MAXIMUM_HISTORICAL_LOOKBACK.total_seconds()
        ),
    }


async def run(args: argparse.Namespace) -> MLValidationSummary:
    definition = resolve_instrument(
        asset=args.asset,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        capability=InstrumentCapability.SCORING,
    )
    started = perf_counter()
    tracemalloc.start()
    if args.contract_only:
        configure_logging("INFO")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(
            "ML experiment contract validated",
            extra={"structured_fields": _contract_fields()},
        )
        return MLValidationSummary(
            None, perf_counter() - started, _approximate_peak_resident_bytes(peak)
        )

    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            read_result = await read_ml_experiment_candles(
                session=session,
                instrument=definition.instrument,
                start=DEVELOPMENT_START,
                end=FINAL_TEST_END,
            )
            dataset = build_ml_experiment_dataset(
                instrument=definition.instrument, columns=read_result.columns
            )
            target_summary: dict[str, object] = {}
            for duration in FUTURE_DURATIONS:
                target = dataset.targets.for_duration(duration)
                target_summary[str(int(duration.total_seconds()))] = {
                    "eligible": int(target.eligible.sum()),
                    "exclusions": target.exclusions.as_dict(),
                }
            if session.new or session.dirty or session.deleted:
                raise RuntimeError(
                    "ML dataset validation attempted to mutate the session"
                )
    finally:
        await engine.dispose()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = perf_counter() - started
    approximate_peak_resident_bytes = _approximate_peak_resident_bytes(peak)
    logger.info(
        "ML experiment dataset validated",
        extra={
            "structured_fields": {
                **_contract_fields(),
                "asset": definition.asset.value,
                "provider": definition.provider.value,
                "market_type": definition.market_type.value,
                "interval": definition.interval.value,
                "actual_range": [
                    dataset.metadata.actual_start.isoformat()
                    if dataset.metadata.actual_start
                    else None,
                    dataset.metadata.actual_end.isoformat()
                    if dataset.metadata.actual_end
                    else None,
                ],
                "candle_count": dataset.metadata.candle_count,
                "gap_event_count": dataset.metadata.gaps.gap_event_count,
                "missing_candle_slot_count": (
                    dataset.metadata.gaps.missing_candle_slot_count
                ),
                "first_missing_open_time": (
                    dataset.metadata.gaps.first_missing_open_time.isoformat()
                    if dataset.metadata.gaps.first_missing_open_time
                    else None
                ),
                "last_missing_open_time": (
                    dataset.metadata.gaps.last_missing_open_time.isoformat()
                    if dataset.metadata.gaps.last_missing_open_time
                    else None
                ),
                "raw_boundary_count": len(dataset.boundary_time_ns),
                "feature_eligible_boundary_count": int(dataset.feature_eligible.sum()),
                "feature_exclusions": dataset.feature_exclusions.as_dict(),
                "temporal_folds": [
                    {
                        "name": fold.definition.name,
                        "history_mode": fold.definition.history_mode.value,
                        "training_range": [
                            fold.definition.training_start.isoformat(),
                            fold.definition.training_end.isoformat(),
                        ],
                        "validation_range": [
                            fold.definition.validation_start.isoformat(),
                            fold.definition.validation_end.isoformat(),
                        ],
                        "raw_training_rows": len(fold.training_indices),
                        "raw_validation_rows": len(fold.validation_indices),
                        "feature_eligible_training_rows": int(
                            dataset.feature_eligible[fold.training_indices].sum()
                        ),
                        "feature_eligible_validation_rows": int(
                            dataset.feature_eligible[fold.validation_indices].sum()
                        ),
                    }
                    for fold in dataset.folds
                ],
                "eligible_samples_by_duration": target_summary,
                "final_test_row_count": len(dataset.final_test_indices),
                "dataset_fingerprint": dataset.metadata.fingerprint,
                "elapsed_seconds": elapsed,
                "approximate_peak_resident_bytes": approximate_peak_resident_bytes,
            }
        },
    )
    return MLValidationSummary(dataset, elapsed, approximate_peak_resident_bytes)


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
