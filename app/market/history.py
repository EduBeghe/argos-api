from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HistoryOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"


class HistoryWindow(BaseModel):
    """Shared bounded, inclusive UTC window for persisted history reads."""

    model_config = ConfigDict(populate_by_name=True)

    from_time: datetime | None = Field(default=None, alias="from")
    to_time: datetime | None = Field(default=None, alias="to")
    order: HistoryOrder = HistoryOrder.DESC
    limit: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        for name in ("from_time", "to_time"):
            value = getattr(self, name)
            if value is not None:
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(f"{name} must be timezone-aware")
                setattr(self, name, value.astimezone(UTC))
        if (
            self.from_time is not None
            and self.to_time is not None
            and self.from_time > self.to_time
        ):
            raise ValueError("from must not be after to")
        return self
