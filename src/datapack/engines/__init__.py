"""Per-modality engines.

Importing this package registers the available engines via
:func:`datapack.core.detect.register_engine`:

- ``ephys`` (mtscomp; lossless) — Phase 1, lifted from an earlier validated pipeline.
- ``histology`` (tifffile/imagecodecs; lossless) — Phase 2.
- ``video`` (ffmpeg hevc_nvenc) — Phase 3 (not yet implemented).

Engine modules import their heavy, optional third-party deps lazily, so importing
this package succeeds even when those deps are not installed.
"""

from __future__ import annotations

from datapack.engines.ephys import (
    EphysEngine,
    SessionResult,
    compress_oe_session,
    decompress_oe_session,
)
from datapack.engines.ephys import verify_artifact as verify_ephys_artifact
from datapack.engines.histology import (
    ArchiveResult,
    HistologyEngine,
    TiffTreeResult,
    UnpackResult,
    compress_tiff_tree,
    pack,
    repack,
    unpack,
)
from datapack.engines.histology import verify_artifact as verify_histology_artifact
from datapack.engines.video import VideoEngine, probe_ffmpeg

__all__ = [
    "EphysEngine",
    "SessionResult",
    "compress_oe_session",
    "decompress_oe_session",
    "verify_ephys_artifact",
    "VideoEngine",
    "probe_ffmpeg",
    "HistologyEngine",
    "TiffTreeResult",
    "compress_tiff_tree",
    "verify_histology_artifact",
    "ArchiveResult",
    "UnpackResult",
    "pack",
    "unpack",
    "repack",
]
