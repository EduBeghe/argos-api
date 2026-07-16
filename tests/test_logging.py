import json
import logging
from typing import cast

from app.core.logging import JsonFormatter


def test_structured_fields_cannot_overwrite_core_log_fields() -> None:
    record = logging.LogRecord(
        name="argos.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="original message",
        args=(),
        exc_info=None,
    )
    record.__dict__["structured_fields"] = {
        "job_name": "test-job",
        "timestamp": "overwritten",
        "level": "overwritten",
        "logger": "overwritten",
        "message": "overwritten",
        "exception": "overwritten",
    }

    payload = cast(dict[str, object], json.loads(JsonFormatter().format(record)))

    assert payload["job_name"] == "test-job"
    assert payload["timestamp"] != "overwritten"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "argos.test"
    assert payload["message"] == "original message"
    assert "exception" not in payload
