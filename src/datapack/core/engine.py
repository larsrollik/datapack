"""The engine contract every modality implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from datapack.core.job import Job
    from datapack.core.manifest import ManifestRecord
    from datapack.core.verify import VerifyResult


class Engine(ABC):
    """Compress one source into a verified artifact + manifest record.

    Subclasses set the class attributes and implement ``compress`` / ``verify``.
    Implementations MUST:

    - never mutate or delete the source;
    - write via :func:`datapack.core.atomic.atomic_write` (temp -> rename);
    - run their verification gate and raise on failure (never return a
      ``ready_to_offload=True`` record for a bad artifact).
    """

    #: Modality this engine handles, e.g. ``"ephys"``.
    modality: ClassVar[str]
    #: Human/log identifier, e.g. ``"mtscomp"``.
    name: ClassVar[str]
    #: Whether the codec is bit-exact reversible.
    lossless: ClassVar[bool]

    @abstractmethod
    def compress(self, job: Job) -> ManifestRecord:
        """Compress ``job.src`` to ``job.dst`` and return a manifest record."""

    @abstractmethod
    def verify(self, job: Job) -> VerifyResult:
        """Verify the artifact at ``job.dst`` against ``job.src``."""
