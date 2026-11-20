"""``.avi`` goes through the HEVC re-encode, not the remux.

Uses NVENC when it works, else the libx265 CPU fallback; skips when ffmpeg has
neither HEVC encoder.
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import pytest

from datapack import compress_file
from datapack.engines.remux import mp4_stream_info
from datapack.engines.video import FfmpegCapabilities, nvenc_works, probe_ffmpeg

try:
    _caps: FfmpegCapabilities | None = probe_ffmpeg()
except Exception:  # noqa: BLE001
    _caps = None

_nvenc = bool(_caps and _caps.has_nvenc and nvenc_works(_caps.ffmpeg))
pytestmark = pytest.mark.skipif(
    not (_nvenc or (_caps and _caps.has_libx265)),
    reason="no HEVC encoder (hevc_nvenc or libx265) in ffmpeg",
)


def test_avi_reencodes_to_hevc(tmp_path: Path) -> None:
    assert _caps is not None
    src = tmp_path / "cam0.avi"
    subprocess.run(
        [_caps.ffmpeg, "-v", "error", "-y", "-f", "lavfi"]
        + ["-i", "testsrc2=size=320x240:rate=30:duration=1"]
        + ["-c:v", "mjpeg", "-q:v", "3", str(src)],
        check=True,
    )
    dst = tmp_path / "cam0.mp4"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the CPU fallback warns by design
        rec = compress_file(src, dst, params={"allow_cpu": True})

    assert rec.engine == "ffmpeg-hevc"
    assert rec.params["mode"] == "encode"
    assert rec.lossless is False
    assert rec.verification is not None and rec.verification.ok
    assert rec.verification.details["frame_count_match"]
    assert rec.ready_to_offload
    info = mp4_stream_info(_caps.ffprobe, dst)
    assert info["codec"] == "hevc"
    assert info["nb_frames"] == 30
