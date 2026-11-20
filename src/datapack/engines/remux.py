"""Raw H.264 to MP4 remux without re-encoding, verified on the coded slices.

Raspberry Pi cameras write raw Annex-B ``.h264`` streams with no timing. The archival
form is an ``.mp4`` holding the same coded pictures: the stream is remuxed, never
re-encoded, at the frame rate recorded by the acquisition itself.

Frame rate
    Read from the acquisition metadata next to the video and never guessed. Two
    layouts are known, both found from the video name with its ``.h264`` (and a
    following ``.video``) suffix removed:

    - ``<base>.debug_acq_meta.yaml`` (camera ensemble, 2026 onwards): the first
      non-null ``framerate:`` key;
    - ``<base>.metadata.json`` (camera client, 2021 to 2023): the top-level
      ``framerate`` key.

    No metadata file, or no frame rate in it, is an error.

Remuxer
    GPAC ``MP4Box -add <h264>:fps=<F> -new <mp4>``. ffmpeg stream copy
    (``-framerate F -r F -i <h264> -c copy``) is the fallback only when MP4Box is
    missing, and emits a warning.

Verification (all must hold before the artifact is accepted)
    1. frame count: counted packets of the ``.h264`` equal the ``.mp4`` container's
       frame count;
    2. size: the ``.mp4`` is within ``size_tolerance`` (1 %) of the ``.h264``;
    3. frame rate tag: the ``.mp4`` stream rate equals the metadata rate;
    4. coded slices: the SHA-256 over every NAL unit of type 1 and 5 in decode order
       is equal for the ``.h264`` and for the ``.mp4`` extracted back to Annex-B
       (``ffmpeg -c copy -bsf:v h264_mp4toannexb``). SPS, PPS, AUD and SEI units
       are framing and may differ.

Source deletion
    datapack never deletes the source (the engine contract). A record with
    ``ready_to_offload=True`` and ``extra["verdict"] == "match"`` is the evidence
    that the ``.h264`` may be deleted by the caller. Frame times live in the TTL and
    timestamp files next to the video and are not touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from datapack.core.atomic import atomic_write
from datapack.core.errors import DatapackError, VerificationError
from datapack.core.manifest import ManifestRecord
from datapack.core.verify import VerifyResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from datapack.core.job import Job

RAW_H264_SUFFIXES = {".h264", ".264"}

#: Metadata file suffixes, in lookup order, appended to the video base name.
META_SUFFIXES = (".debug_acq_meta.yaml", ".metadata.json")

#: NAL unit types holding coded picture data (non-IDR and IDR slices).
SLICE_NAL_TYPES = frozenset({1, 5})

#: Default size tolerance between ``.mp4`` and ``.h264`` (fraction).
SIZE_TOLERANCE = 0.01

#: Allowed difference between the ``.mp4`` frame rate tag and the metadata rate.
FPS_TOLERANCE = 0.01

_YAML_FRAMERATE = re.compile(r"^\s*framerate:\s*(?P<value>.*?)\s*$")
_YAML_NULLS = {"", "null", "~", "none"}
_CHUNK = 1 << 22  # 4 MiB


class RemuxError(DatapackError):
    """Raised when a remux cannot run: no frame rate, no remuxer, tool failure."""


# ---------------------------------------------------------------------------
# Frame rate from acquisition metadata
# ---------------------------------------------------------------------------


def _video_base(h264: Path) -> str:
    name = h264.name
    for suffix in sorted(RAW_H264_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.endswith(".video"):
        name = name[: -len(".video")]
    return name


def find_acq_metadata(h264: Path | str) -> Path:
    """Return the acquisition metadata file that belongs to a raw ``.h264``.

    Raises
    ------
    RemuxError
        If no metadata file exists, or more than one candidate does.
    """
    h264 = Path(h264)
    base = _video_base(h264)
    found = [h264.with_name(base + s) for s in META_SUFFIXES]
    found = [p for p in found if p.exists()]
    if not found:
        names = ", ".join(base + s for s in META_SUFFIXES)
        raise RemuxError(
            f"{h264}: no acquisition metadata next to the video (looked for {names}); "
            "the remux frame rate is never guessed, pass meta=... explicitly"
        )
    if len(found) > 1:
        raise RemuxError(
            f"{h264}: more than one acquisition metadata file "
            f"({', '.join(p.name for p in found)}); pass meta=... explicitly"
        )
    return found[0]


def _positive(value: float, meta: Path) -> float:
    if not value > 0:
        raise RemuxError(f"{meta}: frame rate must be positive, got {value}")
    return value


def _yaml_framerate(meta: Path) -> float | None:
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _YAML_FRAMERATE.match(line)
        if not m:
            continue
        raw = m.group("value").split(" #", 1)[0].strip().strip("'\"")
        if raw.lower() in _YAML_NULLS:
            continue
        try:
            return float(raw)
        except ValueError:
            raise RemuxError(f"{meta}: framerate {raw!r} is not a number") from None
    return None


def _json_framerate(meta: Path) -> float | None:
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RemuxError(f"{meta}: not valid JSON ({exc})") from exc
    value = data.get("framerate") if isinstance(data, dict) else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise RemuxError(f"{meta}: framerate {value!r} is not a number")
    try:
        return float(value)
    except ValueError:
        raise RemuxError(f"{meta}: framerate {value!r} is not a number") from None


def metadata_framerate(meta: Path | str) -> float:
    """Read the acquisition frame rate from a metadata file.

    ``.yaml``/``.yml``: the first non-null ``framerate:`` key (any nesting level;
    ``max_framerate`` and ``actual_framerate`` do not match). ``.json``: the
    top-level ``framerate`` key.

    Raises
    ------
    RemuxError
        If the file has no usable frame rate or an unknown suffix.
    """
    meta = Path(meta)
    suffix = meta.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        value = _yaml_framerate(meta)
    elif suffix == ".json":
        value = _json_framerate(meta)
    else:
        raise RemuxError(f"{meta}: unknown metadata format (expected .yaml or .json)")
    if value is None:
        raise RemuxError(
            f"{meta}: no frame rate in the acquisition metadata; refusing to guess"
        )
    return _positive(value, meta)


def framerate_for(
    h264: Path | str, meta: Path | str | None = None
) -> tuple[float, Path]:
    """Return ``(fps, metadata_path)`` for a raw ``.h264``, from its metadata."""
    path = Path(meta) if meta is not None else find_acq_metadata(h264)
    if not path.exists():
        raise RemuxError(f"metadata file does not exist: {path}")
    return metadata_framerate(path), path


# ---------------------------------------------------------------------------
# Coded-slice hash
# ---------------------------------------------------------------------------


def iter_nal_units(stream: IO[bytes], *, chunk_size: int = _CHUNK) -> Iterator[bytes]:
    """Yield the NAL units of an Annex-B byte stream, in order, without start codes.

    Trailing zero bytes (``trailing_zero_8bits`` and the leading byte of a 4-byte
    start code) are stripped; a NAL unit never ends in a zero byte.
    """
    buf = b""
    start: int | None = None
    pos = 0
    while True:
        chunk = stream.read(chunk_size)
        if chunk:
            buf += chunk
        while True:
            j = buf.find(b"\x00\x00\x01", pos)
            if j < 0:
                break
            if start is not None:
                nal = buf[start:j].rstrip(b"\x00")
                if nal:
                    yield nal
            start = pos = j + 3
        if not chunk:
            break
        if start is None:
            buf = buf[-2:]
            pos = 0
        else:
            buf = buf[start:]
            start = 0
            pos = max(0, len(buf) - 2)
    if start is not None:
        nal = buf[start:].rstrip(b"\x00")
        if nal:
            yield nal


@dataclass(frozen=True)
class SliceHash:
    """SHA-256 over the coded slices of a stream, and how many there were."""

    sha256: str
    n_slices: int


def slice_hash(stream: IO[bytes]) -> SliceHash:
    """Hash every NAL unit of type 1 or 5 of an Annex-B stream, in decode order.

    Each unit enters the hash as its 4-byte big-endian length followed by its bytes,
    so unit boundaries are part of the digest.
    """
    h = hashlib.sha256()
    n = 0
    for nal in iter_nal_units(stream):
        if nal[0] & 0x1F in SLICE_NAL_TYPES:
            h.update(len(nal).to_bytes(4, "big"))
            h.update(nal)
            n += 1
    return SliceHash(h.hexdigest(), n)


def h264_slice_hash(path: Path | str) -> SliceHash:
    """Slice hash of a raw Annex-B ``.h264`` file."""
    with Path(path).open("rb") as fh:
        return slice_hash(fh)


def mp4_slice_hash(ffmpeg: str, path: Path | str) -> SliceHash:
    """Slice hash of the H.264 track of an ``.mp4``, extracted back to Annex-B."""
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-bsf:v",
        "h264_mp4toannexb",
        "-f",
        "h264",
        "-",
    ]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        assert proc.stdout is not None
        try:
            result = slice_hash(proc.stdout)
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            err.seek(0)
            msg = err.read().decode(errors="replace")[-2000:]
            raise RemuxError(f"ffmpeg Annex-B extraction failed for {path}:\n{msg}")
    return result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def resolve_mp4box(mp4box: str | None = None) -> str | None:
    """Resolve MP4Box via the argument, ``$DATAPACK_MP4BOX`` or PATH, else None."""
    cand = mp4box or os.environ.get("DATAPACK_MP4BOX")
    if cand:
        return cand if (Path(cand).exists() or shutil.which(cand)) else None
    return shutil.which("MP4Box")


def _tool_version(cmd: list[str]) -> str | None:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:  # pragma: no cover
        return None
    lines = (res.stdout + res.stderr).strip().splitlines()
    return lines[0] if lines else None


def select_remuxer(remuxer: str, mp4box: str | None) -> tuple[str, str | None]:
    """Pick ``("mp4box", path)`` or the ``("ffmpeg", None)`` stream-copy fallback.

    ``remuxer`` is ``"auto"`` (MP4Box, else ffmpeg with a warning), ``"mp4box"``
    (fail without it) or ``"ffmpeg"``.
    """
    if remuxer not in {"auto", "mp4box", "ffmpeg"}:
        raise RemuxError(f"unknown remuxer {remuxer!r}; use auto, mp4box or ffmpeg")
    if remuxer == "ffmpeg":
        return "ffmpeg", None
    path = resolve_mp4box(mp4box)
    if path is not None:
        return "mp4box", path
    if remuxer == "mp4box":
        raise RemuxError(
            "MP4Box (GPAC) not found; install gpac, set $DATAPACK_MP4BOX or pass "
            "mp4box=..."
        )
    warnings.warn(
        "MP4Box (GPAC) not found; falling back to ffmpeg stream copy for the remux",
        stacklevel=2,
    )
    return "ffmpeg", None


def build_mp4box_cmd(mp4box: str, src: Path, dst: Path, fps: float) -> list[str]:
    """``MP4Box -add <src>:fps=<fps> -new <dst>``, temp files beside ``dst``."""
    return [
        mp4box,
        "-quiet",
        "-tmp",
        str(dst.parent),
        "-add",
        f"{src}:fps={fps:g}",
        "-new",
        str(dst),
    ]


def build_ffmpeg_copy_cmd(ffmpeg: str, src: Path, dst: Path, fps: float) -> list[str]:
    """Fallback: ffmpeg stream copy at a forced input frame rate, no re-encode."""
    return [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-f",
        "h264",
        "-framerate",
        f"{fps:g}",
        # -r on the input also overrides VUI timing in the SPS, which -framerate
        # alone does not under stream copy
        "-r",
        f"{fps:g}",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-an",
        "-f",
        "mp4",
        str(dst),
    ]


def _ffprobe_entries(ffprobe: str, path: Path, args: list[str]) -> dict[str, str]:
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", *args, str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RemuxError(f"ffprobe failed on {path}:\n{res.stderr[-2000:]}")
    return dict(line.split("=", 1) for line in res.stdout.splitlines() if "=" in line)


def h264_packet_count(ffprobe: str, path: Path) -> int:
    """Counted packets of a raw ``.h264`` (one per access unit)."""
    d = _ffprobe_entries(
        ffprobe,
        path,
        [
            "-f",
            "h264",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "default=nw=1",
        ],
    )
    return int(d["nb_read_packets"])


def mp4_stream_info(ffprobe: str, path: Path) -> dict[str, Any]:
    """Container frame count, codec, size and frame rate of the ``.mp4`` video."""
    d = _ffprobe_entries(
        ffprobe,
        path,
        [
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames",
            "-of",
            "default=nw=1",
        ],
    )
    num, _, den = d.get("r_frame_rate", "0/1").partition("/")
    rate = float(num) / float(den or 1) if float(den or 1) else 0.0
    nb = d.get("nb_frames", "")
    return {
        "codec": d.get("codec_name"),
        "width": int(d["width"]) if d.get("width", "").isdigit() else None,
        "height": int(d["height"]) if d.get("height", "").isdigit() else None,
        "r_frame_rate": d.get("r_frame_rate"),
        "fps_tag": rate,
        "nb_frames": int(nb) if nb.isdigit() else None,
    }


# ---------------------------------------------------------------------------
# Verification and the remux itself
# ---------------------------------------------------------------------------


def verify_remux(
    src: Path,
    dst: Path,
    *,
    fps: float,
    ffmpeg: str,
    ffprobe: str,
    size_tolerance: float = SIZE_TOLERANCE,
) -> VerifyResult:
    """Check a remuxed ``.mp4`` against its ``.h264`` (module docstring, 1 to 4)."""
    n_in = h264_packet_count(ffprobe, src)
    info = mp4_stream_info(ffprobe, dst)
    n_out = info["nb_frames"]
    src_size = src.stat().st_size
    size_ratio = dst.stat().st_size / src_size if src_size else float("inf")
    h_in = h264_slice_hash(src)
    h_out = mp4_slice_hash(ffmpeg, dst)

    checks = {
        "codec_h264": info["codec"] == "h264",
        "frame_count_match": n_out is not None and n_in == n_out,
        "size_within_tolerance": abs(size_ratio - 1.0) <= size_tolerance,
        "fps_tag_match": abs(info["fps_tag"] - fps) <= FPS_TOLERANCE,
        "slice_hash_match": h_in.n_slices > 0 and h_in.sha256 == h_out.sha256,
    }
    failed = [k for k, v in checks.items() if not v]
    ok = not failed
    return VerifyResult(
        ok=ok,
        method="framecount+size+fps+slice_sha256",
        details={
            "in_frames": n_in,
            "out_frames": n_out,
            "size_ratio": size_ratio,
            "size_tolerance": size_tolerance,
            "fps": fps,
            "fps_tag": info["fps_tag"],
            "r_frame_rate": info["r_frame_rate"],
            "codec": info["codec"],
            "width": info["width"],
            "height": info["height"],
            "slice_sha256_h264": h_in.sha256,
            "slice_sha256_mp4": h_out.sha256,
            "slices_h264": h_in.n_slices,
            "slices_mp4": h_out.n_slices,
            **checks,
            "failed_checks": failed,
            "verdict": "match" if ok else "mismatch",
            "source_deletable": ok,
        },
    )


def remux_h264(
    job: Job,
    *,
    ffmpeg: str,
    ffprobe: str,
    ffmpeg_version: str | None = None,
    mp4box: str | None = None,
    remuxer: str = "auto",
    meta: Path | str | None = None,
    size_tolerance: float = SIZE_TOLERANCE,
    overwrite: bool = False,
    tool_version: str | None = None,
) -> ManifestRecord:
    """Remux ``job.src`` (raw ``.h264``) to ``job.dst`` (``.mp4``) and verify it.

    The artifact is verified on its temp path and only renamed into place when every
    check passes; on failure nothing is left at ``job.dst`` and
    :class:`~datapack.core.errors.VerificationError` is raised. The source is never
    modified or deleted.
    """
    src, dst = job.src, job.dst
    if src.suffix.lower() not in RAW_H264_SUFFIXES:
        raise RemuxError(f"{src}: remux takes a raw .h264 stream")
    fps, meta_path = framerate_for(src, meta)
    kind, mp4box_path = select_remuxer(remuxer, mp4box)
    found: dict[str, VerifyResult] = {}

    def produce(tmp: Path) -> None:
        if kind == "mp4box":
            assert mp4box_path is not None
            cmd = build_mp4box_cmd(mp4box_path, src, tmp, fps)
        else:
            cmd = build_ffmpeg_copy_cmd(ffmpeg, src, tmp, fps)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RemuxError(f"{kind} remux failed:\n{res.stderr[-2000:]}")
        result = verify_remux(
            src,
            tmp,
            fps=fps,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            size_tolerance=size_tolerance,
        )
        found["v"] = result
        if not result.ok:
            failed = ", ".join(result.details["failed_checks"])
            raise VerificationError(
                f"{dst}: remux verification failed ({failed}); nothing written, "
                f"source kept: {result.details}"
            )

    atomic_write(dst, produce, overwrite=overwrite)
    verification = found["v"]

    remuxer_version = (
        _tool_version([mp4box_path, "-version"]) if mp4box_path else ffmpeg_version
    )
    source_size = src.stat().st_size
    output_size = dst.stat().st_size
    return ManifestRecord(
        source_path=src,
        source_size=source_size,
        modality="video",
        engine="mp4box-remux" if kind == "mp4box" else "ffmpeg-copy-remux",
        engine_version=remuxer_version,
        params={
            "mode": "remux",
            "remuxer": kind,
            "fps": fps,
            "fps_source": str(meta_path),
            "size_tolerance": size_tolerance,
        },
        output_path=dst,
        output_size=output_size,
        ratio=output_size / source_size if source_size else None,
        lossless=True,  # coded slices are bit-identical; only the container changes
        verification=verification,
        ready_to_offload=verification.ok,
        tool_version=tool_version,
        verified_at=datetime.now(UTC),
        extra={
            "remuxer": kind,
            "remuxer_version": remuxer_version,
            "ffmpeg": ffmpeg_version,
            "fps_source": str(meta_path),
            **verification.details,
        },
    )
