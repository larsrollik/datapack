"""A unit of compression work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Job(BaseModel):
    """One source -> one artifact target, with an optional modality + overrides.

    The source must exist and must differ from the destination — datapack never
    writes to or over the raw original (the core invariant).
    """

    src: Path
    dst: Path
    modality: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("src")
    @classmethod
    def _src_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"source does not exist: {v}")
        return v

    @model_validator(mode="after")
    def _dst_differs_from_src(self) -> Job:
        if self.dst.resolve() == self.src.resolve():
            raise ValueError(
                f"destination must differ from source (got {self.src}); "
                "datapack never overwrites the raw original"
            )
        return self
