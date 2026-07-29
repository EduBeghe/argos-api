from typing import Literal

from pydantic import BaseModel


class PublicHealthResponse(BaseModel):
    """Minimal public process-health payload."""

    status: Literal["ok"]


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: Literal["ok", "unavailable"]
    service: str
    version: str
