"""Real raw H.264 remux + verify on a synthetic stream.

Builds a short H.264 stream with ffmpeg (``testsrc`` plus noise, a few dozen frames)
and fake camera ensemble (YAML) and camera client (JSON) metadata next to it. Skips
when ffmpeg has no H.264 encoder, and the MP4Box tests skip when MP4Box is missing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from datapack import compress_file
from datapack.cli import main
from datapack.core import Job, VerificationError, read_manifest, sha256_file
from datapack.engines.remux import (
    RemuxError,
    h264_slice_hash,
    mp4_slice_hash,
    mp4_stream_info,
    remux_h264,
    resolve_mp4box,
    verify_remux,
)
from datapack.engines.video import FfmpegCapabilities, probe_ffmpeg

N_FRAMES = 48

_caps: FfmpegCapabilities | None
try:
    _caps = probe_ffmpeg()
    _encoders = subprocess.run(
        [_caps.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
except Exception:  # noqa: BLE001
    _caps, _encoders = None, ""

_H264_ENC = next((e for e in ("libx264", "libopenh264") if e in _encoders), None)
_MP4BOX = resolve_mp4box()

pytestmark = pytest.mark.skipif(
    _caps is None or _H264_ENC is None, reason="no ffmpeg with an H.264 encoder"
)
needs_mp4box = pytest.mark.skipif(_MP4BOX is None, reason="MP4Box (GPAC) missing")


def _make_h264(dst: Path, *, seed: int = 0) -> Path:
    """A raw Annex-B stream large enough that MP4 overhead stays under 1 %."""
    assert _caps is not None and _H264_ENC is not None
    rate = ["-qp", "8"] if _H264_ENC == "libx264" else ["-b:v", "20M"]
    subprocess.run(
        [
            _caps.ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=25,noise=alls=40:allf=t:all_seed={seed}",
            "-frames:v",
            str(N_FRAMES),
            "-c:v",
            _H264_ENC,
            *rate,
            "-g",
            "12",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "h264",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    return dst


def _rce(tmp_path: Path, fps: int = 100) -> Path:
    h264 = _make_h264(tmp_path / "s01.rce.pi3.20260301_100000.video.h264")
    h264.with_name("s01.rce.pi3.20260301_100000.debug_acq_meta.yaml").write_text(
        "configs:\n"
        "  camera_config:\n"
        "    max_framerate: 20\n"
        "    framerate: null\n"
        f"  settings:\n    framerate: {fps}\n"
    )
    return h264


def _rcc(tmp_path: Path, fps: int = 60) -> Path:
    h264 = _make_h264(tmp_path / "s01.rcc.video.h264")
    h264.with_name("s01.rcc.metadata.json").write_text(
        json.dumps({"framerate": fps, "resolution": [320, 240]})
    )
    return h264


def _check_artifact(src: Path, dst: Path, fps: float) -> None:
    assert _caps is not None
    info = mp4_stream_info(_caps.ffprobe, dst)
    assert info["codec"] == "h264"  # not re-encoded
    assert info["nb_frames"] == N_FRAMES
    assert info["fps_tag"] == pytest.approx(fps)
    a, b = h264_slice_hash(src), mp4_slice_hash(_caps.ffmpeg, dst)
    assert a == b
    assert a.n_slices >= N_FRAMES


def _check_record_fields(rec_extra: dict, fps: float) -> None:
    assert rec_extra["verdict"] == "match"
    assert rec_extra["source_deletable"] is True
    assert rec_extra["fps"] == fps
    assert rec_extra["slice_sha256_h264"] == rec_extra["slice_sha256_mp4"]
    assert rec_extra["in_frames"] == rec_extra["out_frames"] == N_FRAMES
    assert abs(rec_extra["size_ratio"] - 1) <= 0.01
    assert rec_extra["failed_checks"] == []


@needs_mp4box
def test_remux_rce_via_cli_writes_manifest(tmp_path: Path) -> None:
    src = _rce(tmp_path)
    before = sha256_file(src)
    dst = tmp_path / "out" / "s01.mp4"
    report = tmp_path / "manifest.jsonl"
    assert main(["video", str(src), str(dst), "--report", str(report)]) == 0

    _check_artifact(src, dst, 100)
    assert sha256_file(src) == before  # source never touched, never deleted
    [rec] = read_manifest(report)
    assert rec.engine == "mp4box-remux"
    assert rec.lossless is True
    assert rec.ready_to_offload is True
    assert rec.verification is not None and rec.verification.ok
    assert rec.params["mode"] == "remux"
    assert rec.params["remuxer"] == "mp4box"
    assert rec.params["fps"] == 100.0
    assert rec.params["fps_source"].endswith(".debug_acq_meta.yaml")
    assert "GPAC" in (rec.engine_version or "")
    _check_record_fields(rec.extra, 100.0)
    assert not list(dst.parent.glob(".*.tmp"))  # temp cleaned


@needs_mp4box
def test_remux_rcc_via_compress_file(tmp_path: Path) -> None:
    src = _rcc(tmp_path)
    dst = tmp_path / "s01.rcc.mp4"
    rec = compress_file(src, dst)  # modality and mode from the file type
    _check_artifact(src, dst, 60)
    assert rec.params["fps_source"].endswith(".metadata.json")
    _check_record_fields(rec.extra, 60.0)


def test_remux_ffmpeg_copy_fallback(tmp_path: Path) -> None:
    src = _rce(tmp_path, fps=50)
    dst = tmp_path / "s01.mp4"
    rec = compress_file(src, dst, params={"remuxer": "ffmpeg"})
    assert rec.engine == "ffmpeg-copy-remux"
    _check_artifact(src, dst, 50)
    _check_record_fields(rec.extra, 50.0)


def test_remux_without_metadata_refuses(tmp_path: Path) -> None:
    src = _make_h264(tmp_path / "bare.h264")
    dst = tmp_path / "bare.mp4"
    with pytest.raises(RemuxError, match="never guessed"):
        compress_file(src, dst)
    assert not dst.exists()
    assert main(["video", str(src), str(dst)]) == 1


def test_verify_detects_different_content(tmp_path: Path) -> None:
    assert _caps is not None
    src = _rce(tmp_path)
    good = tmp_path / "good.mp4"
    compress_file(src, good, params={"remuxer": "ffmpeg"})
    other = _make_h264(tmp_path / "other.h264", seed=7)
    bad = tmp_path / "bad.mp4"
    subprocess.run(
        [_caps.ffmpeg, "-v", "error", "-r", "100", "-i", str(other)]
        + ["-c", "copy", str(bad)],
        check=True,
    )
    res = verify_remux(src, bad, fps=100, ffmpeg=_caps.ffmpeg, ffprobe=_caps.ffprobe)
    assert not res.ok
    assert "slice_hash_match" in res.details["failed_checks"]
    assert res.details["verdict"] == "mismatch"
    assert res.details["source_deletable"] is False
    res = verify_remux(src, good, fps=60, ffmpeg=_caps.ffmpeg, ffprobe=_caps.ffprobe)
    assert res.details["failed_checks"] == ["fps_tag_match"]


def test_failed_verification_leaves_nothing(tmp_path: Path) -> None:
    src = _rce(tmp_path)
    dst = tmp_path / "s01.mp4"
    assert _caps is not None
    with pytest.raises(VerificationError, match="size_within_tolerance"):
        remux_h264(
            Job(src=src, dst=dst),
            ffmpeg=_caps.ffmpeg,
            ffprobe=_caps.ffprobe,
            remuxer="ffmpeg",
            size_tolerance=0.0,  # any container overhead now fails
        )
    assert not dst.exists()
    assert not list(tmp_path.glob(".*.tmp"))
