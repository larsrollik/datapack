"""Modality detection and the engine registry.

Detection is engine-independent (by file name/extension), so a source can be
classified even before its engine exists. The registry maps a modality to its
``Engine`` subclass; engines call :func:`register_engine` at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datapack.core.errors import EngineNotAvailable

if TYPE_CHECKING:
    from datapack.core.engine import Engine

# Engine-independent classification rules.
_EPHYS_NAMES = {"continuous.dat"}
_VIDEO_EXTS = {".h264", ".264", ".mp4", ".avi", ".mkv"}
_HISTOLOGY_EXTS = {".tif", ".tiff"}

_REGISTRY: dict[str, type[Engine]] = {}


def detect_modality(path: Path | str) -> str | None:
    """Classify a source path as ephys/video/histology, or None if unknown."""
    p = Path(path)
    if p.name.lower() in _EPHYS_NAMES:
        return "ephys"
    suffix = p.suffix.lower()
    if suffix in _VIDEO_EXTS:
        return "video"
    if suffix in _HISTOLOGY_EXTS:
        return "histology"
    return None


def register_engine(cls: type[Engine]) -> type[Engine]:
    """Register an engine class under its ``modality``. Usable as a decorator."""
    _REGISTRY[cls.modality] = cls
    return cls


def get_engine_class(modality: str) -> type[Engine]:
    """Return the engine class for a modality, or raise ``EngineNotAvailable``."""
    try:
        return _REGISTRY[modality]
    except KeyError:
        raise EngineNotAvailable(modality, _REGISTRY.keys()) from None


def available_modalities() -> list[str]:
    """Modalities that currently have a registered engine."""
    return sorted(_REGISTRY)
