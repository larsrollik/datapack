"""Video engine — visually-lossless H.265 re-encoding via ffmpeg (NVENC).

Behavioural video (RPi camera ensemble, ~100 fps raw H.264) is re-encoded to HEVC
for DLC/SLEAP. This is **visually lossless, not bit-exact**: the .mp4
is the working form, consumed directly downstream — there is no decompress step
(unlike the lossless ephys/histology round-trips).

Deployment note (SLURM): we do NOT manage the GPU. The engine probes for a working
nvenc-capable ffmpeg at runtime and, by default, **fails loudly if none is found**
rather than silently grinding on CPU for hours. Pass ``allow_cpu=True`` to opt into
the libx265 fallback (with a prominent slow-encoding warning). Resolve ffmpeg via
the ``ffmpeg=`` arg, ``$DATAPACK_FFMPEG``, or PATH — never assume PATH ffmpeg has
nvenc (it often does not).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datapack.core.atomic import atomic_write
from datapack.core.detect import register_engine
from datapack.core.engine import Engine
from datapack.core.errors import DatapackError, VerificationError
from datapack.core.manifest import ManifestRecord
from datapack.core.verify import VerifyResult

if TYPE_CHECKING:
    from datapack.core.job import Job

RAW_H264_SUFFIXES = {".h264", ".264"}

#: Named QP presets matching the jimbob DLC validation set (preset p7, g100).
#: PROVISIONAL — the DLC/SLEAP tracking-accuracy validation is NOT finished; do not
#: treat "small" (qp22) as confirmed-safe until that lands.
QP_PRESETS = {
    "safe": 18,  # maximally safe for tracking; least compression
    "balanced": 20,  # default — most of the gain, conservative
    "small": 22,  # smallest; only after validating on your own tracking models
}


def resolve_qp(quality: str | None, qp: int) -> int:
    """Map a named quality preset to a QP, or fall back to the explicit ``qp``."""
    if quality is None:
        return qp
    try:
        return QP_PRESETS[quality]
    except KeyError:
        raise VideoError(
            f"unknown quality preset {quality!r}; choose from {sorted(QP_PRESETS)}"
        ) from None


class VideoError(DatapackError):
    """Raised when no usable encoder is available or ffmpeg fails."""


@dataclass(frozen=True)
class FfmpegCapabilities:
    ffmpeg: str
    ffprobe: str
    has_nvenc: bool
    has_libx265: bool
    version: str | None


def _resolve_ffmpeg(ffmpeg: str | None) -> str:
    return (
        ffmpeg
        or os.environ.get("DATAPACK_FFMPEG")
        or shutil.which("ffmpeg")
        or "ffmpeg"
    )


def _sibling_ffprobe(ffmpeg: str) -> str:
    p = Path(ffmpeg)
    cand = p.with_name("ffprobe")
    if cand.exists():
        return str(cand)
    return os.environ.get("DATAPACK_FFPROBE") or shutil.which("ffprobe") or "ffprobe"


def _parse_encoders(encoders_text: str) -> tuple[bool, bool]:
    """Return (has_hevc_nvenc, has_libx265) from ``ffmpeg -encoders`` output."""
    return ("hevc_nvenc" in encoders_text, "libx265" in encoders_text)


def probe_ffmpeg(ffmpeg: str | None = None) -> FfmpegCapabilities:
    """Resolve ffmpeg and report which HEVC encoders it exposes."""
    path = _resolve_ffmpeg(ffmpeg)
    try:
        enc = subprocess.run(
            [path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise VideoError(
            f"ffmpeg not found or not runnable at {path!r}; set $DATAPACK_FFMPEG or "
            "pass ffmpeg=... to an nvenc-enabled build"
        ) from exc
    has_nvenc, has_x265 = _parse_encoders(enc)
    version = None
    try:
        ver = subprocess.run(
            [path, "-version"], capture_output=True, text=True, check=True
        ).stdout
        version = ver.splitlines()[0] if ver else None
    except (FileNotFoundError, subprocess.CalledProcessError):  # pragma: no cover
        pass
    return FfmpegCapabilities(
        ffmpeg=path,
        ffprobe=_sibling_ffprobe(path),
        has_nvenc=has_nvenc,
        has_libx265=has_x265,
        version=version,
    )


def nvenc_works(ffmpeg: str) -> bool:
    """Functional probe: can this ffmpeg actually open an HEVC NVENC session?"""
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=256x256:rate=5:duration=1",  # above HEVC NVENC min dimensions
        "-c:v",
        "hevc_nvenc",
        "-rc",
        "constqp",
        "-qp",
        "30",
        "-f",
        "null",
        "-",
    ]
    try:
        return subprocess.run(cmd, capture_output=True).returncode == 0
    except FileNotFoundError:  # pragma: no cover
        return False


def select_encoder(
    caps: FfmpegCapabilities,
    *,
    allow_cpu: bool = False,
    functional_check: bool = True,
    _nvenc_works: Any = nvenc_works,
) -> str:
    """Choose ``hevc_nvenc`` if a working GPU session exists, else fail (or warn+CPU).

    On SLURM the GPU is managed externally, so the default is to fail loudly when
    NVENC is unavailable — never silently fall back to hours-long CPU encoding.
    """
    if caps.has_nvenc and (not functional_check or _nvenc_works(caps.ffmpeg)):
        return "hevc_nvenc"
    if allow_cpu and caps.has_libx265:
        warnings.warn(
            "No working NVENC GPU encoder; falling back to CPU libx265. CPU HEVC "
            "encoding is far slower than realtime — expect very long processing "
            "times on full sessions.",
            stacklevel=2,
        )
        return "libx265"
    raise VideoError(
        "no usable HEVC encoder: need an nvenc-enabled ffmpeg with a working GPU "
        "(set $DATAPACK_FFMPEG or pass ffmpeg=...), or pass allow_cpu=True to accept "
        "the slow libx265 CPU fallback"
    )


def build_encode_cmd(
    ffmpeg: str,
    src: Path,
    dst: Path,
    *,
    encoder: str,
    qp: int,
    crf: int,
    preset: str,
    gop: int,
    spatial_aq: bool,
    fps: float | None,
    is_raw: bool,
) -> list[str]:
    """Construct the ffmpeg command. ``-f mp4`` is forced (temp path has no suffix)."""
    cmd = [ffmpeg, "-y"]
    if encoder == "hevc_nvenc":
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if is_raw and fps:
        cmd += ["-framerate", str(fps)]
    cmd += ["-i", str(src)]
    if encoder == "hevc_nvenc":
        cmd += [
            "-c:v",
            "hevc_nvenc",
            "-preset",
            preset,
            "-rc",
            "constqp",
            "-qp",
            str(qp),
            "-g",
            str(gop),
            "-bf",
            "0",
        ]
        if spatial_aq:
            cmd += ["-spatial-aq", "1"]
        # NOTE: no -pix_fmt here — frames stay in GPU (cuda) memory and nvenc
        # encodes NV12 (8-bit 4:2:0) directly. Forcing -pix_fmt yuv420p inserts a
        # software filter that can't run on cuda frames and breaks the pipeline.
    else:  # libx265 (CRF != QP scale) — software frames, set 8-bit 4:2:0
        cmd += [
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-x265-params",
            f"keyint={gop}:bframes=0:aq-mode=2",
            "-pix_fmt",
            "yuv420p",
        ]
    cmd += [
        "-tag:v",
        "hvc1",
        "-an",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(dst),
    ]
    return cmd


def _count_frames(ffprobe: str, path: Path, *, raw: bool) -> int:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
    ]
    if raw:
        cmd += ["-f", "h264"]
    cmd += [str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def _probe_stream(ffprobe: str, path: Path) -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,r_frame_rate",
        "-of",
        "csv=p=0",
        str(path),
    ]
    parts = (
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        .stdout.strip()
        .split(",")
    )
    keys = ["width", "height", "pix_fmt", "r_frame_rate"]
    return dict(zip(keys, parts, strict=False))


@register_engine
class VideoEngine(Engine):
    """Visually-lossless HEVC re-encode of behavioural video.

    NVENC, with a libx265 fallback.
    """

    modality = "video"
    name = "ffmpeg-hevc"
    lossless = False

    def __init__(
        self,
        *,
        qp: int = 20,
        crf: int = 20,
        preset: str = "p7",
        gop: int = 100,
        spatial_aq: bool = True,
        fps: float = 100.0,
        ffmpeg: str | None = None,
        allow_cpu: bool = False,
    ) -> None:
        self.qp = qp
        self.crf = crf
        self.preset = preset
        self.gop = gop
        self.spatial_aq = spatial_aq
        self.fps = fps
        self.ffmpeg = ffmpeg
        self.allow_cpu = allow_cpu

    def compress(self, job: Job) -> ManifestRecord:
        p = job.params
        qp = resolve_qp(p.get("quality"), int(p.get("qp", self.qp)))
        fps = float(p.get("fps", self.fps))
        allow_cpu = bool(p.get("allow_cpu", self.allow_cpu))
        overwrite = bool(p.get("overwrite", False))

        caps = probe_ffmpeg(p.get("ffmpeg", self.ffmpeg))
        encoder = select_encoder(caps, allow_cpu=allow_cpu)
        is_raw = job.src.suffix.lower() in RAW_H264_SUFFIXES

        def produce(tmp: Path) -> None:
            cmd = build_encode_cmd(
                caps.ffmpeg,
                job.src,
                tmp,
                encoder=encoder,
                qp=qp,
                crf=int(p.get("crf", self.crf)),
                preset=self.preset,
                gop=self.gop,
                spatial_aq=self.spatial_aq,
                fps=fps,
                is_raw=is_raw,
            )
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise VideoError(f"ffmpeg encode failed:\n{res.stderr[-2000:]}")

        atomic_write(job.dst, produce, overwrite=overwrite)

        verification = self._verify(caps, job.src, job.dst, is_raw=is_raw)
        if not verification.ok:
            job.dst.unlink(missing_ok=True)
            raise VerificationError(f"{job.dst}: {verification.details}")

        source_size = job.src.stat().st_size
        output_size = job.dst.stat().st_size
        stream = _probe_stream(caps.ffprobe, job.dst)
        return ManifestRecord(
            source_path=job.src,
            source_size=source_size,
            modality=self.modality,
            engine=self.name,
            engine_version=caps.version,
            params={
                "encoder": encoder,
                "qp": qp,
                "crf": int(p.get("crf", self.crf)),
                "preset": self.preset,
                "gop": self.gop,
                "spatial_aq": self.spatial_aq,
                "fps": fps,
                "pix_fmt": "yuv420p",
            },
            output_path=job.dst,
            output_size=output_size,
            ratio=output_size / source_size if source_size else None,
            lossless=False,  # visually lossless, NOT bit-exact
            verification=verification,
            ready_to_offload=verification.ok,
            tool_version=_datapack_version(),
            verified_at=datetime.now(UTC),
            extra={
                "encoder": encoder,
                "ffmpeg": caps.version,
                **stream,
                **verification.details,
            },
        )

    def verify(self, job: Job) -> VerifyResult:
        caps = probe_ffmpeg(job.params.get("ffmpeg", self.ffmpeg))
        is_raw = job.src.suffix.lower() in RAW_H264_SUFFIXES
        return self._verify(caps, job.src, job.dst, is_raw=is_raw)

    def _verify(
        self, caps: FfmpegCapabilities, src: Path, dst: Path, *, is_raw: bool
    ) -> VerifyResult:
        # 1) frame count in == out (the key equivalence check for a lossy re-encode)
        n_in = _count_frames(caps.ffprobe, src, raw=is_raw)
        n_out = _count_frames(caps.ffprobe, dst, raw=False)
        # 2) output decodes cleanly end-to-end
        dec = subprocess.run(
            [caps.ffmpeg, "-v", "error", "-i", str(dst), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        decode_clean = dec.returncode == 0 and not dec.stderr.strip()
        ok = n_in == n_out and decode_clean
        return VerifyResult(
            ok=ok,
            method="framecount+decode",
            details={
                "in_frames": n_in,
                "out_frames": n_out,
                "frame_count_match": n_in == n_out,
                "decode_clean": decode_clean,
            },
        )


def _datapack_version() -> str | None:
    try:
        from datapack import __version__

        return __version__
    except Exception:  # pragma: no cover
        return None
