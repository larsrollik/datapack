"""Hashing helpers and the shared verification-result type.

Verification *strategy* is per-modality and lives in each engine (lossless
round-trip for ephys/histology; frame-count + clean-decode for video). This
module only provides the shared primitives: content hashing and the
``VerifyResult`` record that engines return and the manifest stores.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path | str, *, chunk_size: int = _CHUNK) -> str:
    """Return the hex SHA-256 of a file's bytes, read in streaming chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


class VerifyResult(BaseModel):
    """Outcome of an engine's verification gate.

    Parameters
    ----------
    ok
        Whether the artifact passed verification.
    method
        Short identifier of how it was verified, e.g. ``"mtscomp_check"``,
        ``"pixel_array_equal"``, ``"frame_count+decode"``.
    details
        Free-form, JSON-serialisable evidence (counts, hashes, metrics).
    """

    ok: bool
    method: str
    details: dict[str, Any] = Field(default_factory=dict)
