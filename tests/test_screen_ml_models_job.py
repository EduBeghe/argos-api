from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.jobs import screen_ml_models
from app.jobs.screen_ml_models import build_parser
from app.market.enums import Asset
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


class _SessionContext:
    def __init__(self) -> None:
        self.session = SimpleNamespace(new=set(), dirty=set(), deleted=set())

    async def __aenter__(self) -> SimpleNamespace:
        return self.session

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_dry_run_imports_dependencies_without_writes_or_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated: list[bool] = []
    monkeypatch.setattr(
        screen_ml_models, "validate_engine_dependencies", lambda: validated.append(True)
    )
    monkeypatch.setattr(
        screen_ml_models,
        "get_settings",
        lambda: SimpleNamespace(log_level="INFO", ml_screening_thread_count=2),
    )
    monkeypatch.setattr(screen_ml_models, "configure_logging", lambda level: None)
    monkeypatch.setattr(screen_ml_models, "library_versions", lambda: {"all": "loaded"})
    monkeypatch.setattr(
        screen_ml_models,
        "_resolve_asset",
        lambda asset: SimpleNamespace(instrument=SimpleNamespace(), asset=asset),
    )
    dispose = AsyncMock()
    monkeypatch.setattr(
        screen_ml_models,
        "create_database_engine",
        lambda settings: SimpleNamespace(dispose=dispose),
    )
    context = _SessionContext()
    monkeypatch.setattr(
        screen_ml_models, "create_session_factory", lambda engine: lambda: context
    )
    monkeypatch.setattr(
        screen_ml_models,
        "read_ml_experiment_candles",
        AsyncMock(return_value=SimpleNamespace(columns=object())),
    )
    monkeypatch.setattr(
        screen_ml_models,
        "build_ml_experiment_dataset",
        lambda **kwargs: SimpleNamespace(
            metadata=SimpleNamespace(fingerprint="d" * 64), folds=(object(),) * 7
        ),
    )
    fit = AsyncMock()
    monkeypatch.setattr(screen_ml_models, "screen_models", fit)
    result = await screen_ml_models.run(
        Namespace(
            dry_run=True,
            force_recompute=False,
            asset=Asset.BTC_USDT,
            engine="all",
            all_future_durations=False,
            future_duration_seconds=[1800],
        )
    )
    assert result is None
    assert validated == [True]
    fit.assert_not_awaited()
    dispose.assert_awaited_once()


@pytest.mark.asyncio
async def test_conflicting_dry_run_force_fails_before_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = AsyncMock()
    monkeypatch.setattr(screen_ml_models, "validate_engine_dependencies", validate)
    with pytest.raises(ValueError, match="cannot be combined"):
        await screen_ml_models.run(Namespace(dry_run=True, force_recompute=True))
    validate.assert_not_awaited()
