"""Exception hierarchy for datapack."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class DatapackError(Exception):
    """Base class for all datapack errors."""


class ModalityDetectionError(DatapackError):
    """Raised when the modality of a source path cannot be determined."""


class EngineNotAvailable(DatapackError):
    """Raised when no engine is registered for a (known) modality.

    In early phases the core contract exists before the per-modality engines, so
    a known modality can still have no engine. The message lists what *is*
    available to make that explicit rather than cryptic.
    """

    def __init__(self, modality: str, available: Iterable[str] = ()) -> None:
        self.modality = modality
        self.available = sorted(available)
        msg = f"no engine registered for modality {modality!r}"
        msg += (
            f" (available: {', '.join(self.available)})"
            if self.available
            else " (no engines registered yet)"
        )
        super().__init__(msg)


class VerificationError(DatapackError):
    """Raised when a compressed artifact fails its verification gate."""
