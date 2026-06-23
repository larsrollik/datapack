"""Default compression profiles per modality.

These are *data*: the values the engines will consume in later phases. They
capture the decisions/recommendations from the project ROADMAP so the defaults
are reviewable in one place. Engines are not implemented yet (Phase 1-3).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EphysProfile(BaseModel):
    """IBL mtscomp, truly lossless (ROADMAP D1/D6)."""

    codec: str = "mtscomp"
    mtscomp_pin: str = "==1.0.2"
    # NOTE: dreamteam_etl measured 120 s chunks; mtscomp's API default is 1 s.
    # Resolve this gap explicitly in Phase 1 (affects ratio + random access).
    chunk_duration_s: float = 1.0
    n_threads: int = 8
    # mtscomp's built-in decompress+byte-compare IS the integrity gate (D6).
    check_after_compress: bool = True


class VideoProfile(BaseModel):
    """ffmpeg HEVC, visually lossless (ROADMAP D2/D3/D8)."""

    encoder: str = "hevc_nvenc"
    qp: int = 20  # qp18 = max-safe, qp22 = max-shrink; lock via DLC/SLEAP validation
    # Named presets from the jimbob DLC validation set (preset p7, g100).
    # PROVISIONAL — tracking-accuracy validation not finished (ROADMAP D2).
    qp_presets: dict[str, int] = {"safe": 18, "balanced": 20, "small": 22}
    preset: str = "p7"
    gop: int = 100
    bframes: int = 0
    spatial_aq: bool = True
    pix_fmt: str = "yuv420p"
    container: str = "mp4"
    fallback_encoder: str = "libx265"
    fallback_crf: int = 20


class HistologyProfile(BaseModel):
    """Lossless tiled BigTIFF (ROADMAP D4)."""

    codec: str = "deflate"  # promote to "zstd" after lab-reader compat check
    level: int = 6
    predictor: bool = True
    tile: tuple[int, int] = (256, 256)
    bigtiff: bool = True


class Config(BaseModel):
    """Top-level config bundling the per-modality profiles."""

    ephys: EphysProfile = Field(default_factory=EphysProfile)
    video: VideoProfile = Field(default_factory=VideoProfile)
    histology: HistologyProfile = Field(default_factory=HistologyProfile)


DEFAULT_CONFIG = Config()
