from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_application_starts(client: TestClient) -> None:
    application = cast(FastAPI, client.app)

    assert application.title == "argos-api"
    assert application.state.session_factory is not None
