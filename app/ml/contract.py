from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

ML_EXPERIMENT_CONTRACT_VERSION = 2
VARIABLE_REGISTRY_VERSION = 1
MISSING_DATA_POLICY_VERSION = 1
DEVELOPMENT_START = datetime(2020, 1, 1, tzinfo=UTC)
DEVELOPMENT_END = datetime(2024, 1, 1, tzinfo=UTC)
FINAL_TEST_START = DEVELOPMENT_END
QUALIFICATION_END = datetime(2025, 1, 1, tzinfo=UTC)
FINAL_TEST_END = datetime(2026, 1, 1, tzinfo=UTC)
DECISION_INTERVAL = timedelta(minutes=30)
CANDLE_INTERVAL = timedelta(minutes=5)
SPLIT_GAP = timedelta(days=7)


@dataclass(frozen=True, slots=True)
class DurationRange:
    start: timedelta
    stop: timedelta
    step: timedelta


FUTURE_DURATION_RANGES = (
    DurationRange(timedelta(minutes=30), timedelta(hours=12), timedelta(minutes=30)),
    DurationRange(timedelta(hours=13), timedelta(hours=48), timedelta(hours=1)),
    DurationRange(timedelta(hours=54), timedelta(days=7), timedelta(hours=6)),
)


def _expand_duration_range(specification: DurationRange) -> tuple[timedelta, ...]:
    count = int((specification.stop - specification.start) / specification.step) + 1
    return tuple(
        specification.start + index * specification.step for index in range(count)
    )


FUTURE_DURATIONS = tuple(
    duration
    for specification in FUTURE_DURATION_RANGES
    for duration in _expand_duration_range(specification)
)
assert len(FUTURE_DURATIONS) == 80
assert tuple(sorted(set(FUTURE_DURATIONS))) == FUTURE_DURATIONS


class TrainingHistoryMode(StrEnum):
    ACCUMULATING = "accumulating"
    ROLLING_TWO_YEARS = "rolling_two_years"
    ROLLING_THREE_YEARS = "rolling_three_years"


@dataclass(frozen=True, slots=True)
class TemporalFoldDefinition:
    name: str
    history_mode: TrainingHistoryMode
    training_start: datetime
    training_end: datetime
    validation_start: datetime
    validation_end: datetime


VALIDATION_BLOCKS = (
    (datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 7, 1, tzinfo=UTC)),
    (datetime(2022, 7, 1, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC)),
    (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 7, 1, tzinfo=UTC)),
    (datetime(2023, 7, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
)


def _subtract_years(value: datetime, years: int) -> datetime:
    return value.replace(year=value.year - years)


def temporal_fold_definitions() -> tuple[TemporalFoldDefinition, ...]:
    folds: list[TemporalFoldDefinition] = []
    for validation_start, validation_end in VALIDATION_BLOCKS:
        training_end = validation_start - SPLIT_GAP
        starts = (
            (TrainingHistoryMode.ACCUMULATING, DEVELOPMENT_START),
            (TrainingHistoryMode.ROLLING_TWO_YEARS, _subtract_years(training_end, 2)),
            (TrainingHistoryMode.ROLLING_THREE_YEARS, _subtract_years(training_end, 3)),
        )
        for mode, training_start in starts:
            minimum_training_start = _subtract_years(training_end, 2)
            if (
                training_start < DEVELOPMENT_START
                or minimum_training_start < DEVELOPMENT_START
            ):
                continue
            folds.append(
                TemporalFoldDefinition(
                    name=f"{validation_start:%Y%m}_{mode.value}",
                    history_mode=mode,
                    training_start=training_start,
                    training_end=training_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                )
            )
    return tuple(folds)

