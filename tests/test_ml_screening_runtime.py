from pathlib import Path

import pytest

from app.ml.screening import engines
from app.ml.screening.memory import (
    approximate_peak_resident_bytes,
    parse_linux_peak_resident_bytes,
)


def test_dependency_validation_imports_all_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(engines, "import_module", imported.append)
    engines.validate_engine_dependencies()
    assert imported == ["sklearn", "lightgbm", "catboost"]


def test_dependency_validation_propagates_broken_native_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_on_lightgbm(module_name: str) -> None:
        if module_name == "lightgbm":
            raise ImportError("native library unavailable")

    monkeypatch.setattr(engines, "import_module", fail_on_lightgbm)
    with pytest.raises(ImportError, match="native library unavailable"):
        engines.validate_engine_dependencies()


def test_linux_peak_resident_memory_and_fallback(tmp_path: Path) -> None:
    assert parse_linux_peak_resident_bytes("Name:\tpython\nVmHWM:\t123 kB\n") == (
        123 * 1024
    )
    status = tmp_path / "status"
    status.write_text("VmHWM:\t456 kB\n", encoding="ascii")
    assert approximate_peak_resident_bytes(100, status_path=status) == 456 * 1024
    assert approximate_peak_resident_bytes(100, status_path=tmp_path / "missing") == 100
    status.write_text("VmHWM:\tnot-a-number kB\n", encoding="ascii")
    assert approximate_peak_resident_bytes(100, status_path=status) == 100
