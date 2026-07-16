import json
import logging
from datetime import UTC, datetime
from typing import cast


class JsonFormatter(logging.Formatter):
    """Format application logs as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        structured_fields: object = record.__dict__.get("structured_fields")
        if isinstance(structured_fields, dict):
            payload.update(cast(dict[str, object], structured_fields))
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    """Configure production-friendly structured logging."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
