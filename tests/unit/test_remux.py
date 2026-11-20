"""Raw H.264 remux logic: metadata frame rate, slice hash, mode and tool selection.

No ffmpeg or MP4Box required.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from datapack.core import Job
from datapack.engines.remux import (
    RemuxError,
    build_ffmpeg_copy_cmd,
    build_mp4box_cmd,
    find_acq_metadata,
    framerate_for,
    iter_nal_units,
    metadata_framerate,
    select_remuxer,
    slice_hash,
)
from datapack.engines.video import VideoEngine, VideoError, resolve_mode

RCE_YAML = """\
timestamp_saved: '2026-03-01T10:00:00'
camera_controls:
  FrameDurationLimits: [10000, 10000]
configs:
  camera_config:
    max_framerate: 20
    framerate: null
  settings:
    framerate: 100 # Hz
    actual_framerate: 99.7
"""


# --- metadata frame rate -----------------------------------------------------


def _rce(tmp_path: Path, text: str = RCE_YAML) -> Path:
    h264 = tmp_path / "s01.rce.pi3.20260301_100000.video.h264"
    h264.write_bytes(b"")
    h264.with_name("s01.rce.pi3.20260301_100000.debug_acq_meta.yaml").write_text(text)
    return h264


def _rcc(tmp_path: Path, payload: object) -> Path:
    h264 = tmp_path / "s01.rcc.video.h264"
    h264.write_bytes(b"")
    h264.with_name("s01.rcc.metadata.json").write_text(json.dumps(payload))
    return h264


def test_rce_yaml_first_non_null_framerate(tmp_path: Path) -> None:
    h264 = _rce(tmp_path)
    fps, meta = framerate_for(h264)
    assert fps == 100.0  # null skipped; max_/actual_framerate do not match
    assert meta.name == "s01.rce.pi3.20260301_100000.debug_acq_meta.yaml"


def test_rce_yaml_quoted_value(tmp_path: Path) -> None:
    h264 = _rce(tmp_path, "camera:\n  framerate: '59.94'\n")
    assert framerate_for(h264)[0] == pytest.approx(59.94)


def test_rcc_json_framerate(tmp_path: Path) -> None:
    h264 = _rcc(tmp_path, {"framerate": 60, "resolution": [640, 480]})
    fps, meta = framerate_for(h264)
    assert fps == 60.0
    assert meta.name == "s01.rcc.metadata.json"


def test_explicit_meta_overrides_lookup(tmp_path: Path) -> None:
    h264 = tmp_path / "clip.h264"
    h264.write_bytes(b"")
    meta = tmp_path / "elsewhere.json"
    meta.write_text(json.dumps({"framerate": 30}))
    assert framerate_for(h264, meta) == (30.0, meta)


def test_missing_metadata_file_is_an_error(tmp_path: Path) -> None:
    h264 = tmp_path / "s01.rcc.video.h264"
    h264.write_bytes(b"")
    with pytest.raises(RemuxError, match="never guessed"):
        framerate_for(h264)
    with pytest.raises(RemuxError, match="does not exist"):
        framerate_for(h264, tmp_path / "nope.json")


@pytest.mark.parametrize(
    "payload", [{}, {"framerate": None}, {"fps": 60}, [60], {"framerate": "x"}]
)
def test_rcc_without_framerate_is_an_error(tmp_path: Path, payload: object) -> None:
    with pytest.raises(RemuxError):
        framerate_for(_rcc(tmp_path, payload))


@pytest.mark.parametrize(
    "text",
    [
        "camera:\n  framerate: null\n",
        "camera:\n  framerate: ~\n",
        "camera:\n  max_framerate: 20\n",
        "camera:\n  framerate: fast\n",
        "camera:\n  framerate: 0\n",
    ],
)
def test_rce_without_framerate_is_an_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(RemuxError):
        framerate_for(_rce(tmp_path, text))


def test_ambiguous_metadata_is_an_error(tmp_path: Path) -> None:
    h264 = tmp_path / "clip.video.h264"
    h264.write_bytes(b"")
    (tmp_path / "clip.debug_acq_meta.yaml").write_text("framerate: 100\n")
    (tmp_path / "clip.metadata.json").write_text('{"framerate": 60}')
    with pytest.raises(RemuxError, match="more than one"):
        find_acq_metadata(h264)


def test_unknown_metadata_format(tmp_path: Path) -> None:
    meta = tmp_path / "meta.txt"
    meta.write_text("framerate: 100")
    with pytest.raises(RemuxError, match="unknown metadata format"):
        metadata_framerate(meta)


# --- NAL parsing and the slice hash -----------------------------------------


def _nal(nal_type: int, body: bytes) -> bytes:
    return bytes([0x60 | nal_type]) + body


SPS, PPS, AUD = _nal(7, b"\x42\x00\x1f"), _nal(8, b"\xce\x3c\x80"), _nal(9, b"\xf0")
IDR, P1, P2 = _nal(5, b"\x88\x84" * 40), _nal(1, b"\x9a\x01\x02"), _nal(1, b"\x9a\x03")


def _annexb(nals: list[bytes], *, long_codes: bool = False) -> bytes:
    code = b"\x00\x00\x00\x01" if long_codes else b"\x00\x00\x01"
    return b"".join(code + n for n in nals)


def test_iter_nal_units_small_chunks() -> None:
    nals = [SPS, PPS, IDR, P1, P2]
    data = _annexb(nals, long_codes=True) + b"\x00\x00"  # trailing_zero_8bits
    for chunk in (1, 2, 3, 7, 1 << 20):
        got = list(iter_nal_units(io.BytesIO(data), chunk_size=chunk))
        assert got == nals, chunk


def test_slice_hash_ignores_framing_units() -> None:
    a = slice_hash(io.BytesIO(_annexb([SPS, PPS, IDR, P1, P2])))
    b = slice_hash(
        io.BytesIO(_annexb([AUD, SPS, PPS, SPS, IDR, AUD, P1, P2], long_codes=True))
    )
    assert a == b
    assert a.n_slices == 3
    expected = hashlib.sha256()
    for n in (IDR, P1, P2):
        expected.update(len(n).to_bytes(4, "big") + n)
    assert a.sha256 == expected.hexdigest()


def test_slice_hash_detects_changed_or_reordered_slices() -> None:
    ref = slice_hash(io.BytesIO(_annexb([SPS, IDR, P1, P2])))
    changed = _nal(1, b"\x9a\x01\x03")
    assert slice_hash(io.BytesIO(_annexb([SPS, IDR, changed, P2]))) != ref
    assert slice_hash(io.BytesIO(_annexb([SPS, IDR, P2, P1]))) != ref
    assert slice_hash(io.BytesIO(_annexb([SPS, IDR, P1]))) != ref


def test_slice_hash_includes_unit_boundaries() -> None:
    # same concatenated bytes, different split into slices -> different digest
    a = slice_hash(io.BytesIO(_annexb([_nal(1, b"\x01\x02"), _nal(1, b"\x03")])))
    b = slice_hash(io.BytesIO(_annexb([_nal(1, b"\x01"), _nal(1, b"\x02\x03")])))
    assert a.sha256 != b.sha256


# --- commands, tool selection and mode ---------------------------------------


def test_mp4box_cmd() -> None:
    cmd = build_mp4box_cmd("MP4Box", Path("in.h264"), Path("d/out.tmp"), 100.0)
    assert cmd[cmd.index("-add") + 1] == "in.h264:fps=100"
    assert cmd[cmd.index("-new") + 1] == "d/out.tmp"
    assert cmd[cmd.index("-tmp") + 1] == "d"


def test_ffmpeg_copy_cmd_never_reencodes() -> None:
    cmd = build_ffmpeg_copy_cmd("ff", Path("in.h264"), Path("out.tmp"), 59.94)
    assert cmd[cmd.index("-framerate") + 1] == "59.94"
    assert cmd[cmd.index("-r") + 1] == "59.94"  # overrides SPS timing on copy
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert cmd[-3:] == ["-f", "mp4", "out.tmp"]


def test_select_remuxer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATAPACK_MP4BOX", raising=False)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    with pytest.raises(RemuxError, match="MP4Box"):
        select_remuxer("mp4box", None)
    with pytest.warns(UserWarning, match="ffmpeg stream copy"):
        assert select_remuxer("auto", None) == ("ffmpeg", None)
    assert select_remuxer("ffmpeg", None) == ("ffmpeg", None)
    with pytest.raises(RemuxError, match="unknown remuxer"):
        select_remuxer("x", None)
    monkeypatch.setattr("shutil.which", lambda n: f"/usr/bin/{n}")
    assert select_remuxer("auto", None) == ("mp4box", "/usr/bin/MP4Box")


def test_resolve_mode_by_file_type() -> None:
    assert resolve_mode("auto", Path("a.h264")) == "remux"
    assert resolve_mode("auto", Path("a.264")) == "remux"
    assert resolve_mode("auto", Path("a.avi")) == "encode"
    assert resolve_mode("encode", Path("a.h264")) == "encode"
    with pytest.raises(VideoError, match="raw .h264"):
        resolve_mode("remux", Path("a.avi"))
    with pytest.raises(VideoError, match="unknown video mode"):
        resolve_mode("copy", Path("a.h264"))


def test_remux_refuses_explicit_fps(tmp_path: Path) -> None:
    src = _rce(tmp_path)
    job = Job(src=src, dst=tmp_path / "out.mp4", params={"fps": 100})
    with pytest.raises(VideoError, match="acquisition metadata"):
        VideoEngine().compress(job)
