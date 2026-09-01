"""Real HEVC encode + verify — gated on a working NVENC ffmpeg.

Skips unless a working nvenc-capable ffmpeg is reachable (PATH or $DATAPACK_FFMPEG).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from datapack.core import Job
from datapack.engines.video import (
    FfmpegCapabilities,
    VideoEngine,
    nvenc_works,
    probe_ffmpeg,
)

try:
    _caps: FfmpegCapabilities | None = probe_ffmpeg()
except Exception:  # noqa: BLE001
    _caps = None

_have_gpu = bool(_caps and _caps.has_nvenc and nvenc_works(_caps.ffmpeg))
pytestmark = pytest.mark.skipif(
    not _have_gpu, reason="no working NVENC ffmpeg (set $DATAPACK_FFMPEG)"
)


def test_video_encode_and_verify(tmp_path: Path) -> None:
    assert _caps is not None
    inp = tmp_path / "in.mp4"
    # synth source via the same ffmpeg (h264_nvenc is present on an nvenc build)
    subprocess.run(
        [
            _caps.ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=10:duration=1",  # above HEVC NVENC min
            "-c:v",
            "h264_nvenc",
            "-f",
            "mp4",
            str(inp),
        ],
        check=True,
        capture_output=True,
    )
    dst = tmp_path / "out.mp4"
    rec = VideoEngine(ffmpeg=_caps.ffmpeg).compress(
        Job(src=inp, dst=dst, modality="video")
    )
    assert dst.exists()
    assert rec.lossless is False
    assert rec.verification is not None and rec.verification.ok
    assert rec.verification.details["frame_count_match"]
    assert rec.ready_to_offload
