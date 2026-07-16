from pathlib import Path


def test_api_start_script_migrates_before_starting_uvicorn() -> None:
    script = Path("scripts/start_api.sh").read_text(encoding="utf-8")

    assert "set -eu" in script
    assert script.index("python -m alembic upgrade head") < script.index(
        "exec uvicorn main:app"
    )
    assert '--port "${PORT:-8000}"' in script
    assert "--host 0.0.0.0" in script
    assert "--workers 1" in script
    assert "--reload" not in script


def test_processes_keep_api_and_ingestion_responsibilities_separate() -> None:
    procfile = Path("Procfile").read_text(encoding="utf-8").splitlines()

    assert procfile == [
        "web: scripts/start_api.sh",
        "ingest: python -m app.jobs.ingest_closed_candles",
    ]


def test_dockerfile_uses_the_production_start_script() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY scripts ./scripts" in dockerfile
    assert 'CMD ["sh", "scripts/start_api.sh"]' in dockerfile
    assert 'CMD ["uvicorn"' not in dockerfile
