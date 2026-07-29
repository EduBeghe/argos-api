from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class SessionResponse(BaseModel):
    username: str
    role: Literal["admin"]
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    username: str
    role: str
    expires_at: datetime
    csrf_digest: str
