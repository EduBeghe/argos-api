from dataclasses import dataclass

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument

from .contract import DEVELOPMENT_START, FINAL_TEST_END, FINAL_TEST_START
from .dataset import MLDatasetMetadata, MLExperimentDataset, build_ml_experiment_dataset
from .gaps import datetime_to_ns, ns_to_datetime
from .reader import read_ml_experiment_candles

ML_EXPERIMENT_HISTORY_START = DEVELOPMENT_START
ML_EXPERIMENT_HISTORY_END = FINAL_TEST_END
ML_DEVELOPMENT_EVALUATION_END = FINAL_TEST_START


class DevelopmentDatasetProtectedEvidenceError(RuntimeError):
    def __init__(self, actual_boundary: str) -> None:
        super().__init__(
            "development evaluation dataset reaches protected final-test evidence; "
            f"development_end_exclusive={ML_DEVELOPMENT_EVALUATION_END.isoformat()}; "
            f"actual_boundary={actual_boundary}"
        )


@dataclass(frozen=True, slots=True)
class RefinementDatasetInputs:
    identity_metadata: MLDatasetMetadata
    development_dataset: MLExperimentDataset


async def load_canonical_ml_experiment_dataset(
    *, session: AsyncSession, instrument: Instrument
) -> MLExperimentDataset:
    """Rebuild the immutable screening dataset over its closed exclusive range."""

    candles = await read_ml_experiment_candles(
        session=session,
        instrument=instrument,
        start=ML_EXPERIMENT_HISTORY_START,
        end=ML_EXPERIMENT_HISTORY_END,
    )
    return build_ml_experiment_dataset(
        instrument=instrument,
        columns=candles.columns,
        requested_end=ML_EXPERIMENT_HISTORY_END,
    )


async def load_canonical_ml_experiment_metadata(
    *, session: AsyncSession, instrument: Instrument
) -> MLDatasetMetadata:
    """Rebuild canonical identity and release its full calculation arrays."""

    canonical = await load_canonical_ml_experiment_dataset(
        session=session, instrument=instrument
    )
    metadata = canonical.metadata
    del canonical
    return metadata


async def load_ml_development_dataset(
    *, session: AsyncSession, instrument: Instrument
) -> MLExperimentDataset:
    """Build calculation evidence strictly before the protected final-test range."""

    candles = await read_ml_experiment_candles(
        session=session,
        instrument=instrument,
        start=DEVELOPMENT_START,
        end=ML_DEVELOPMENT_EVALUATION_END,
    )
    dataset = build_ml_experiment_dataset(
        instrument=instrument,
        columns=candles.columns,
        requested_end=ML_DEVELOPMENT_EVALUATION_END,
    )
    protected = dataset.boundary_time_ns >= datetime_to_ns(FINAL_TEST_START)
    if np.any(protected):
        first = int(dataset.boundary_time_ns[np.flatnonzero(protected)[0]])
        raise DevelopmentDatasetProtectedEvidenceError(
            ns_to_datetime(first).isoformat()
        )
    return dataset


async def load_refinement_dataset_inputs(
    *, session: AsyncSession, instrument: Instrument
) -> RefinementDatasetInputs:
    """Load identity then development data without retaining both feature matrices."""

    identity_metadata = await load_canonical_ml_experiment_metadata(
        session=session, instrument=instrument
    )
    development = await load_ml_development_dataset(
        session=session, instrument=instrument
    )
    return RefinementDatasetInputs(identity_metadata, development)
