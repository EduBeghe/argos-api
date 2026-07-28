from datetime import timedelta
from typing import cast

from app.jobs.validate_ml_experiment_dataset import _contract_fields, build_parser
from app.market.enums import Asset


def test_contract_only_cli_accepts_one_catalog_asset() -> None:
    args = build_parser().parse_args(["--asset", "BTC/USDT", "--contract-only"])
    assert args.asset is Asset.BTC_USDT
    assert args.contract_only is True
    fields = _contract_fields()
    assert len(cast(list[int], fields["future_duration_seconds"])) == 80
    assert fields["maximum_historical_lookback_seconds"] == int(
        timedelta(days=14, minutes=5).total_seconds()
    )
