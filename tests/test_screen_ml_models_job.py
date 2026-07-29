import pytest

from app.jobs.screen_ml_models import build_parser
from app.ml.contract import FUTURE_DURATIONS


def test_parser_accepts_every_contract_duration() -> None:
    parser = build_parser()
    for duration in FUTURE_DURATIONS:
        seconds = str(int(duration.total_seconds()))
        args = parser.parse_args(
            ["--asset", "BTC/USDT", "--future-duration-seconds", seconds, "--dry-run"]
        )
        assert args.future_duration_seconds == [int(seconds)]


def test_parser_rejects_external_and_conflicting_duration_arguments() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--asset", "BTC/USDT", "--future-duration-seconds", "301"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--asset",
                "BTC/USDT",
                "--future-duration-seconds",
                "1800",
                "--all-future-durations",
            ]
        )
