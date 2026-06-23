"""datapack core: the modality-independent compress/verify/manifest contract."""

from __future__ import annotations

from datapack.core.atomic import atomic_write, atomic_write_many
from datapack.core.detect import (
    available_modalities,
    detect_modality,
    get_engine_class,
    register_engine,
)
from datapack.core.engine import Engine
from datapack.core.errors import (
    DatapackError,
    EngineNotAvailable,
    ModalityDetectionError,
    VerificationError,
)
from datapack.core.job import Job
from datapack.core.manifest import ManifestRecord, ManifestWriter, read_manifest
from datapack.core.verify import VerifyResult, sha256_file

__all__ = [
    "atomic_write",
    "atomic_write_many",
    "available_modalities",
    "detect_modality",
    "get_engine_class",
    "register_engine",
    "Engine",
    "DatapackError",
    "EngineNotAvailable",
    "ModalityDetectionError",
    "VerificationError",
    "Job",
    "ManifestRecord",
    "ManifestWriter",
    "read_manifest",
    "VerifyResult",
    "sha256_file",
]
