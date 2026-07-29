from app.market.catalog import MARKET_INSTRUMENTS
from app.ml.contract import DEVELOPMENT_END
from app.ml.dataset import build_ml_experiment_dataset
from app.ml.gaps import datetime_to_ns
from app.ml.reader import candle_columns


def test_refinement_dataset_contract_excludes_protected_final_test() -> None:
    columns = candle_columns(
        open_times=[],
        close_times=[],
        open_prices=[],
        high_prices=[],
        low_prices=[],
        close_prices=[],
        volumes=[],
    )
    dataset = build_ml_experiment_dataset(
        instrument=MARKET_INSTRUMENTS[0].instrument,
        columns=columns,
        requested_end=DEVELOPMENT_END,
    )
    assert dataset.metadata.requested_end == DEVELOPMENT_END
    assert len(dataset.final_test_indices) == 0
    assert dataset.boundary_time_ns[-1] < datetime_to_ns(DEVELOPMENT_END)
