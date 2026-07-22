from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

import app.optimization.edge_guided_memory as memory_module
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.edge_guided import EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
from app.optimization.edge_guided_memory import (
    EDGE_GUIDED_PERSISTENCE_CONTRACT_VERSION,
    edge_guided_experiment_key,
)
from app.optimization.walk_forward import WalkForwardConfiguration
from app.paper.config import SimulationConfiguration

START = datetime(2025, 1, 1, tzinfo=UTC)
RUN_ID = UUID("00000000-0000-0000-0000-000000000043")
DEFAULT_SIMULATION = SimulationConfiguration()
DEFAULT_WINDOWS = WalkForwardConfiguration()


def key(
    *,
    initial_cash: Decimal = Decimal("10000"),
    simulation: SimulationConfiguration = DEFAULT_SIMULATION,
    fingerprint: str = "a" * 64,
    windows: WalkForwardConfiguration = DEFAULT_WINDOWS,
    hypotheses: tuple[str, ...] = ("b" * 64,),
    maximum_candidates: int = 12,
) -> str:
    return edge_guided_experiment_key(
        predictive_edge_run_id=RUN_ID,
        dataset_fingerprint=fingerprint,
        instrument=MARKET_INSTRUMENTS[0].instrument,
        range_start=START,
        range_end=START + timedelta(days=365),
        windows=windows,
        initial_cash=initial_cash,
        simulation=simulation,
        decision_interval=timedelta(minutes=30),
        maximum_candidates=maximum_candidates,
        hypothesis_fingerprints=hypotheses,
    )


def test_experiment_key_is_deterministic_and_hypothesis_order_independent() -> None:
    assert EDGE_GUIDED_PERSISTENCE_CONTRACT_VERSION == 1
    assert EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION == 3
    assert key(hypotheses=("b" * 64, "c" * 64)) == key(hypotheses=("c" * 64, "b" * 64))


def test_experiment_key_changes_with_cash_or_costs() -> None:
    baseline = key()
    assert key(initial_cash=Decimal("20000")) != baseline
    assert (
        key(simulation=SimulationConfiguration(fee_rate=Decimal("0.002"))) != baseline
    )
    assert (
        key(simulation=SimulationConfiguration(slippage_rate=Decimal("0.002")))
        != baseline
    )


def test_experiment_key_changes_with_dataset_windows_limit_or_hypothesis() -> None:
    baseline = key()
    assert key(fingerprint="d" * 64) != baseline
    assert (
        key(windows=replace(WalkForwardConfiguration(), maximum_windows=5)) != baseline
    )
    assert key(maximum_candidates=3) != baseline
    assert key(hypotheses=("e" * 64,)) != baseline


def test_qualification_contract_version_changes_experiment_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = key()
    monkeypatch.setattr(memory_module, "EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION", 1)
    assert key() != baseline
