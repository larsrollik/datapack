"""Video engine logic (no ffmpeg required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from datapack.core import available_modalities, get_engine_class
from datapack.engines.video import (
    QP_PRESETS,
    FfmpegCapabilities,
    VideoError,
    _parse_encoders,
    build_encode_cmd,
    resolve_qp,
    select_encoder,
)


def _caps(nvenc: bool, x265: bool) -> FfmpegCapabilities:
    return FfmpegCapabilities("ffmpeg", "ffprobe", nvenc, x265, "ffmpeg 6.x")


def test_parse_encoders() -> None:
    txt = " V..... hevc_nvenc  NVIDIA\n V..... libx265  x265\n"
    assert _parse_encoders(txt) == (True, True)
    assert _parse_encoders("nothing here") == (False, False)


def test_select_prefers_working_nvenc() -> None:
    enc = select_encoder(_caps(True, True), _nvenc_works=lambda _f: True)
    assert enc == "hevc_nvenc"


def test_select_fails_loud_without_gpu() -> None:
    # SLURM default: no silent CPU grind.
    with pytest.raises(VideoError, match="no usable HEVC encoder"):
        select_encoder(_caps(False, True), allow_cpu=False)


def test_select_cpu_fallback_warns() -> None:
    with pytest.warns(UserWarning, match="far slower than realtime"):
        enc = select_encoder(_caps(False, True), allow_cpu=True)
    assert enc == "libx265"


def test_select_nvenc_listed_but_not_functional() -> None:
    # encoder present but GPU session fails -> fail loud unless cpu allowed
    with pytest.raises(VideoError):
        select_encoder(
            _caps(True, True), allow_cpu=False, _nvenc_works=lambda _f: False
        )
    with pytest.warns(UserWarning):
        enc = select_encoder(
            _caps(True, True), allow_cpu=True, _nvenc_works=lambda _f: False
        )
    assert enc == "libx265"


def test_build_cmd_nvenc_raw() -> None:
    cmd = build_encode_cmd(
        "ff",
        Path("in.h264"),
        Path("out.tmp"),
        encoder="hevc_nvenc",
        qp=20,
        crf=20,
        preset="p7",
        gop=100,
        spatial_aq=True,
        fps=100,
        is_raw=True,
    )
    assert "hevc_nvenc" in cmd
    assert "-qp" in cmd and "20" in cmd
    assert "-framerate" in cmd and "100" in cmd  # raw needs forced fps
    assert "cuda" in cmd
    assert cmd[-3:] == ["-f", "mp4", "out.tmp"]  # format forced (temp has no suffix)


def test_build_cmd_libx265_nonraw() -> None:
    cmd = build_encode_cmd(
        "ff",
        Path("in.mp4"),
        Path("out.tmp"),
        encoder="libx265",
        qp=20,
        crf=20,
        preset="p7",
        gop=100,
        spatial_aq=True,
        fps=100,
        is_raw=False,
    )
    assert "libx265" in cmd and "-crf" in cmd
    assert "-framerate" not in cmd  # non-raw carries its own timing
    assert "cuda" not in cmd


def test_qp_presets_match_jimbob() -> None:
    assert QP_PRESETS == {"safe": 18, "balanced": 20, "small": 22}


def test_resolve_qp() -> None:
    assert resolve_qp(None, 19) == 19  # explicit qp wins when no preset
    assert resolve_qp("safe", 19) == 18  # preset overrides
    assert resolve_qp("small", 20) == 22
    with pytest.raises(VideoError, match="unknown quality preset"):
        resolve_qp("ultra", 20)


def test_video_engine_registered_and_lossy() -> None:
    assert "video" in available_modalities()
    eng = get_engine_class("video")
    assert eng.name == "ffmpeg-hevc"
    assert eng.lossless is False  # visually lossless, not bit-exact
