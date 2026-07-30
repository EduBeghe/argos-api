from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument

from .contract import DEVELOPMENT_START, FINAL_TEST_END
from .dataset import MLExperimentDataset, build_ml_experiment_dataset
from .reader import read_ml_experiment_candles

ML_EXPERIMENT_HISTORY_START = DEVELOPMENT_START
ML_EXPERIMENT_HISTORY_END = FINAL_TEST_END


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
