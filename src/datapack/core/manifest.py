"""The manifest record and its JSON-lines store.

The manifest is the product datapack hands to the downstream offload/tape stage:
everything needed to trust deleting the raw original lives here. One record per
compressed source; a run appends records to a JSON-lines file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from datapack.core.verify import VerifyResult


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ManifestRecord(BaseModel):
    """A self-describing record of one compress+verify operation."""

    # --- source ---
    source_path: Path
    source_size: int
    source_hash: str | None = None

    # --- classification ---
    modality: str
    engine: str
    engine_version: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    # --- output (primary artifact; multi-file outputs also list output_paths) ---
    output_path: Path
    output_size: int | None = None
    output_hash: str | None = None
    output_paths: list[Path] = Field(default_factory=list)
    ratio: float | None = None  # output_size / source_size (fraction of original)

    # --- integrity ---
    lossless: bool
    verification: VerifyResult | None = None
    ready_to_offload: bool = False

    # --- provenance ---
    tool: str = "datapack"
    tool_version: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    verified_at: datetime | None = None

    # --- engine-specific extras (e.g. mtscomp sha1s, n_channels; frame_count) ---
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def saved_fraction(self) -> float | None:
        """Fraction of bytes saved (``1 - ratio``), or None if ratio unknown."""
        return None if self.ratio is None else 1.0 - self.ratio


class ManifestWriter:
    """Append :class:`ManifestRecord` rows to a JSON-lines file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, record: ManifestRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")


def read_manifest(path: Path | str) -> list[ManifestRecord]:
    """Read all records from a JSON-lines manifest file."""
    records: list[ManifestRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(ManifestRecord.model_validate_json(line))
    return records
