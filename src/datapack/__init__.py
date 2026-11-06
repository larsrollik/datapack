"""datapack — verified compression of acquisition data (ephys, video, histology).

Public API (stable; this is what downstream pipelines pin):

- :func:`compress_file` — detect modality, run the engine, return a manifest record.
- :class:`~datapack.core.manifest.ManifestRecord` (aliased ``CompressResult``).
- :class:`~datapack.core.job.Job`, the core types, and :data:`config.DEFAULT_CONFIG`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Import engines for their registration side effect (so get_engine_class works
# after `import datapack`). Engine modules import heavy deps lazily, so this is
# cheap and safe even when e.g. mtscomp is not installed.
from datapack import engines as _engines  # noqa: E402,F401
from datapack.config import DEFAULT_CONFIG, Config
from datapack.core import (
    DatapackError,
    Engine,
    EngineNotAvailable,
    Job,
    ManifestRecord,
    ManifestWriter,
    ModalityDetectionError,
    VerificationError,
    VerifyResult,
    available_modalities,
    detect_modality,
    get_engine_class,
    read_manifest,
    register_engine,
    sha256_file,
)
from datapack.datasets import SerialSectionResult, pack_serial_section  # noqa: E402
from datapack.engines import (  # noqa: E402
    compress_oe_session,
    decompress_oe_session,
)
from datapack.engines.ephys import (  # noqa: E402
    StreamMeta,
    oebin_stream_meta,
    session_stream_meta,
)

try:
    __version__ = version("datapack")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# Name used in the design docs for the result of a compress operation.
CompressResult = ManifestRecord


def compress_file(
    src: Path | str,
    dst: Path | str,
    *,
    modality: str | None = None,
    params: dict[str, Any] | None = None,
) -> ManifestRecord:
    """Compress one file, returning its manifest record.

    Parameters
    ----------
    src
        Source path (never modified).
    dst
        Artifact destination (must differ from ``src``).
    modality
        Force a modality; if None, it is detected from ``src``.
    params
        Engine-specific parameter overrides.

    Raises
    ------
    ModalityDetectionError
        If ``modality`` is None and the source cannot be classified.
    EngineNotAvailable
        If the modality has no registered engine yet (e.g. during early phases).
    """
    src = Path(src)
    modality = modality or detect_modality(src)
    if modality is None:
        raise ModalityDetectionError(
            f"could not detect modality for {src}; pass modality= explicitly"
        )
    engine = get_engine_class(modality)()
    job = Job(src=src, dst=Path(dst), modality=modality, params=params or {})
    return engine.compress(job)


__all__ = [
    "StreamMeta",
    "oebin_stream_meta",
    "session_stream_meta",
    "__version__",
    "compress_file",
    "compress_oe_session",
    "decompress_oe_session",
    "pack_serial_section",
    "SerialSectionResult",
    "CompressResult",
    "Config",
    "DEFAULT_CONFIG",
    "DatapackError",
    "EngineNotAvailable",
    "Engine",
    "Job",
    "ManifestRecord",
    "ManifestWriter",
    "ModalityDetectionError",
    "VerificationError",
    "VerifyResult",
    "available_modalities",
    "detect_modality",
    "get_engine_class",
    "read_manifest",
    "register_engine",
    "sha256_file",
]
