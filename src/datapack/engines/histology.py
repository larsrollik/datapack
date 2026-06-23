"""Histology engine — lossless TIFF recompression (tifffile + imagecodecs).

Targets the lab's brainsaw/StitchIt serial-section datasets: ``serial_section/
<sample>/`` trees of uint16 grayscale TIFFs — big full-res sections under
``stitchedImages_*/<channel>/`` and multi-page downsampled stacks under
``downsampled_stacks/<µm>/`` — plus recipe/log/mat/ini metadata. Measured on real
data: a 152 MB section → 44 MB (71% saved), bit-exact.

Each TIFF is re-encoded in place to a tiled BigTIFF with Deflate (zlib) + the
horizontal-differencing predictor (universally readable; promote to ``zstd`` once
the lab's downstream readers are confirmed — ROADMAP D4). Losslessness is proven
by decoding both files and comparing the pixel arrays (``np.array_equal``) — never
by byte hashing, since identical pixels still differ at the TIFF byte level. The
decoded-pixel SHA-256 is recorded in the manifest so integrity is re-provable
later without the original.

``tifffile``/``imagecodecs``/``numpy`` are optional: install ``datapack[histology]``.
They are imported lazily so the rest of datapack works without them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datapack.core.atomic import atomic_write
from datapack.core.detect import register_engine
from datapack.core.engine import Engine
from datapack.core.errors import DatapackError, VerificationError
from datapack.core.job import Job
from datapack.core.manifest import ManifestRecord, ManifestWriter
from datapack.core.verify import VerifyResult, sha256_file

TIFF_SUFFIXES = {".tif", ".tiff"}
#: Name of the integrity manifest embedded inside every datapack TIFF archive.
ARCHIVE_MANIFEST = ".datapack-archive.json"


class HistologyError(DatapackError):
    """Raised when a TIFF cannot be read or has an unsupported structure."""


def _lazy_tifffile() -> Any:
    try:
        import tifffile
    except ModuleNotFoundError as exc:
        raise DatapackError(
            "the histology engine needs tifffile/imagecodecs; install it with: "
            "pip install 'datapack[histology]'"
        ) from exc
    return tifffile


def _rational_to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return v[0] / v[1] if v[1] else None
    return float(v)


@dataclass
class _TiffData:
    array: Any  # numpy ndarray
    photometric: Any
    xresolution: float | None
    yresolution: float | None
    resolutionunit: Any
    description: str | None
    n_pages: int


def _read_primary(tifffile: Any, path: Path) -> _TiffData:
    """Read the primary series array + the tags we must preserve.

    Multi-series TIFFs are rejected (rather than silently dropping data) — the
    serial-section files are single-series (one image or one page-stack).
    """
    with tifffile.TiffFile(path) as tf:
        if len(tf.series) != 1:
            raise HistologyError(
                f"{path}: {len(tf.series)} series found; multi-series TIFF is not "
                "supported in v1 (would risk dropping data)"
            )
        series = tf.series[0]
        page0 = tf.pages[0]

        def tag(name: str) -> Any:
            t = page0.tags.get(name)
            return t.value if t is not None else None

        return _TiffData(
            array=series.asarray(),
            photometric=series.keyframe.photometric,
            xresolution=_rational_to_float(tag("XResolution")),
            yresolution=_rational_to_float(tag("YResolution")),
            resolutionunit=tag("ResolutionUnit"),
            description=tag("ImageDescription"),
            n_pages=len(tf.pages),
        )


def _pixel_sha256(array: Any) -> str:
    """SHA-256 of the decoded pixel data (+ dtype/shape guard), not the file bytes."""
    import numpy as np

    h = hashlib.sha256()
    h.update(str((array.dtype.str, array.shape)).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def decode_pixel_sha256(path: Path | str) -> str:
    """Decode a TIFF and hash its pixels (for re-proof against the manifest)."""
    tifffile = _lazy_tifffile()
    return _pixel_sha256(_read_primary(tifffile, Path(path)).array)


def verify_artifact(path: Path | str, expected_pixel_sha256: str) -> VerifyResult:
    """Re-prove a recompressed TIFF still decodes to the expected pixels.

    Needs no original — compares the artifact's decoded-pixel hash to the value
    recorded in the manifest at compression time.
    """
    actual = decode_pixel_sha256(path)
    return VerifyResult(
        ok=actual == expected_pixel_sha256,
        method="pixel_sha256",
        details={"expected": expected_pixel_sha256, "actual": actual},
    )


@register_engine
class HistologyEngine(Engine):
    """Lossless re-encoding of TIFF images (tiled BigTIFF, Deflate/ZSTD + predictor)."""

    modality = "histology"
    name = "tifffile"
    lossless = True

    def __init__(
        self,
        *,
        codec: str = "deflate",  # promote to "zstd" after lab-reader compat check (D4)
        level: int = 6,
        tile: tuple[int, int] | None = (256, 256),
    ) -> None:
        self.codec = codec
        self.level = level
        self.tile = tile

    def compress(self, job: Job) -> ManifestRecord:
        import numpy as np

        tifffile = _lazy_tifffile()
        codec = str(job.params.get("codec", self.codec))
        level = int(job.params.get("level", self.level))
        tile = job.params.get("tile", self.tile)
        overwrite = bool(job.params.get("overwrite", False))

        data = _read_primary(tifffile, job.src)
        src_arr = data.array
        # predictor 3 for floating point, 2 (horizontal differencing) otherwise.
        predictor = 3 if src_arr.dtype.kind == "f" else 2
        pixel_hash = _pixel_sha256(src_arr)

        # Map user-facing "deflate" to ADOBE_DEFLATE (TIFF tag 8), the universally
        # recognised deflate codec — tifffile's "deflate" emits tag 32946 (DEFLATE),
        # which some legacy readers don't accept. Both are zlib (same ratio).
        tiff_codec = {"deflate": "adobe_deflate"}.get(codec, codec)
        write_kwargs: dict[str, Any] = {
            "compression": tiff_codec,
            "compressionargs": {"level": level},
            "predictor": predictor,
            "bigtiff": True,
            "photometric": data.photometric,
        }
        if tile is not None:
            write_kwargs["tile"] = tuple(tile)
        if data.xresolution and data.yresolution:
            write_kwargs["resolution"] = (data.xresolution, data.yresolution)
            if data.resolutionunit is not None:
                write_kwargs["resolutionunit"] = data.resolutionunit
        if data.description:
            write_kwargs["description"] = data.description

        def produce(tmp: Path) -> None:
            tifffile.imwrite(str(tmp), src_arr, **write_kwargs)
            # Verification gate, BEFORE the artifact is put in place: decode the
            # temp and compare pixel arrays. Lossy/corrupt output never lands.
            back = _read_primary(tifffile, tmp).array
            same = (
                src_arr.dtype == back.dtype
                and src_arr.shape == back.shape
                and np.array_equal(src_arr, back)
            )
            if not same:
                raise VerificationError(
                    f"{job.src} -> {job.dst}: decoded pixels differ (NOT lossless)"
                )

        atomic_write(job.dst, produce, overwrite=overwrite)

        source_size = job.src.stat().st_size
        output_size = job.dst.stat().st_size
        return ManifestRecord(
            source_path=job.src,
            source_size=source_size,
            modality=self.modality,
            engine=self.name,
            engine_version=getattr(tifffile, "__version__", None),
            params={
                "codec": codec,
                "level": level,
                "predictor": predictor,
                "tile": tile,
            },
            output_path=job.dst,
            output_size=output_size,
            ratio=output_size / source_size if source_size else None,
            lossless=True,
            verification=VerifyResult(
                ok=True,
                method="pixel_array_equal",
                details={"pixel_sha256": pixel_hash},
            ),
            ready_to_offload=True,
            tool_version=_datapack_version(),
            verified_at=datetime.now(UTC),
            extra={
                "pixel_sha256": pixel_hash,
                "dtype": src_arr.dtype.str,
                "shape": list(src_arr.shape),
                "n_pages": data.n_pages,
                "photometric": int(data.photometric),
                "codec": codec,
                "predictor": predictor,
            },
        )

    def verify(self, job: Job) -> VerifyResult:
        """Confirm ``job.dst`` decodes to the same pixels as ``job.src``."""
        src_hash = decode_pixel_sha256(job.src)
        return verify_artifact(job.dst, src_hash)


@dataclass(frozen=True)
class TiffTreeResult:
    """Summary of compressing every TIFF in a directory tree."""

    source: Path
    dest: Path
    records: list[ManifestRecord]
    sidecars_copied: int
    n_tiffs: int
    raw_bytes: int
    output_bytes: int
    dry_run: bool = False

    @property
    def ratio(self) -> float | None:
        if not self.raw_bytes or not self.output_bytes:
            return None
        return self.output_bytes / self.raw_bytes

    @property
    def saved_fraction(self) -> float | None:
        r = self.ratio
        return None if r is None else 1.0 - r


def compress_tiff_tree(
    source: Path | str,
    dest: Path | str,
    *,
    codec: str = "deflate",
    level: int = 6,
    tile: tuple[int, int] | None = (256, 256),
    copy_sidecars: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    manifest_path: Path | str | None = None,
) -> TiffTreeResult:
    """Compress every TIFF under a tree, preserving structure (e.g. a serial_section).

    Mirrors ``source`` to ``dest``, re-encoding each ``*.tif``/``*.tiff`` losslessly
    and copying every other file (recipe ``*.yml``, ``FINISHED``, ``*.txt`` logs,
    ``*.mat``, ``*.ini`` …) verbatim, so the result is still a valid StitchIt
    dataset. The source is only read, never modified.
    """
    source = Path(source).resolve()
    dest = Path(dest)
    if not source.is_dir():
        raise HistologyError(f"not a directory: {source}")

    engine = HistologyEngine(codec=codec, level=level, tile=tile)
    tiffs = sorted(
        p
        for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES
    )

    records: list[ManifestRecord] = []
    raw_bytes = 0
    output_bytes = 0
    for tif in tiffs:
        rel = tif.relative_to(source)
        raw_bytes += tif.stat().st_size
        if dry_run:
            continue
        rec = engine.compress(
            Job(
                src=tif,
                dst=dest / rel,
                modality="histology",
                params={"overwrite": overwrite},
            )
        )
        records.append(rec)
        output_bytes += rec.output_size or 0

    sidecars_copied = 0
    if copy_sidecars and not dry_run:
        for f in source.rglob("*"):
            if f.is_file() and f.suffix.lower() not in TIFF_SUFFIXES:
                target = dest / f.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not overwrite:
                    continue
                shutil.copy2(f, target)
                sidecars_copied += 1

    if manifest_path and not dry_run and records:
        writer = ManifestWriter(manifest_path)
        for rec in records:
            writer.append(rec)

    return TiffTreeResult(
        source=source,
        dest=dest,
        records=records,
        sidecars_copied=sidecars_copied,
        n_tiffs=len(tiffs),
        raw_bytes=raw_bytes,
        output_bytes=output_bytes,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Archive (Tier 1): bundle a raw-acquisition TIFF dir into one zip, with an
# internal integrity manifest, and unpack/repack for the segmentation workflow.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArchiveResult:
    """Summary of packing/repacking a TIFF directory into one archive."""

    source: Path
    archive: Path
    n_tiffs: int
    n_other: int
    raw_bytes: int
    archive_bytes: int
    dry_run: bool = False

    @property
    def ratio(self) -> float | None:
        if not self.raw_bytes or not self.archive_bytes:
            return None
        return self.archive_bytes / self.raw_bytes

    @property
    def saved_fraction(self) -> float | None:
        r = self.ratio
        return None if r is None else 1.0 - r


@dataclass(frozen=True)
class UnpackResult:
    """Summary of unpacking an archive to a working directory."""

    archive: Path
    dest: Path
    n_files: int
    verified: bool


def _iter_data_files(root: Path) -> list[Path]:
    """All regular files under ``root`` except a prior embedded archive manifest."""
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.name != ARCHIVE_MANIFEST
    )


def pack(
    source: Path | str,
    archive: Path | str,
    *,
    codec: str = "deflate",
    level: int = 6,
    tile: tuple[int, int] | None = (256, 256),
    overwrite: bool = False,
    dry_run: bool = False,
) -> ArchiveResult:
    """Bundle a TIFF directory into one zip: TIFFs recompressed then STORED.

    Each ``*.tif``/``*.tiff`` is recompressed losslessly (Deflate/ZSTD + predictor,
    pixel-array verified) and added to the zip in STORE mode (already compressed);
    every other file is added verbatim (deflated). An internal
    ``.datapack-archive.json`` records per-file pixel-SHA-256 (TIFF) / SHA-256
    (other) so unpack/repack can re-verify. ZIP64 enabled for >4 GB. Source is
    only read.
    """
    source = Path(source).resolve()
    archive = Path(archive)
    if not source.is_dir():
        raise HistologyError(f"not a directory: {source}")

    files = _iter_data_files(source)
    n_tiffs = sum(1 for f in files if f.suffix.lower() in TIFF_SUFFIXES)
    raw_bytes = sum(f.stat().st_size for f in files)

    if dry_run:
        return ArchiveResult(
            source=source,
            archive=archive,
            n_tiffs=n_tiffs,
            n_other=len(files) - n_tiffs,
            raw_bytes=raw_bytes,
            archive_bytes=0,
            dry_run=True,
        )

    engine = HistologyEngine(codec=codec, level=level, tile=tile)

    def produce(tmp_archive: Path) -> None:
        entries: list[dict[str, Any]] = []
        with (
            tempfile.TemporaryDirectory(dir=archive.parent) as td,
            zipfile.ZipFile(
                tmp_archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
            ) as zf,
        ):
            tdir = Path(td)
            for f in files:
                rel = f.relative_to(source)
                if f.suffix.lower() in TIFF_SUFFIXES:
                    scratch = tdir / f"{uuid.uuid4().hex}.tif"
                    rec = engine.compress(Job(src=f, dst=scratch, modality="histology"))
                    zf.write(scratch, str(rel), compress_type=zipfile.ZIP_STORED)
                    entries.append(
                        {
                            "path": str(rel),
                            "type": "tiff",
                            "raw_size": f.stat().st_size,
                            "pixel_sha256": rec.extra["pixel_sha256"],
                            "dtype": rec.extra["dtype"],
                            "shape": rec.extra["shape"],
                        }
                    )
                    scratch.unlink(missing_ok=True)
                else:
                    zf.write(f, str(rel))  # deflate small sidecars
                    entries.append(
                        {
                            "path": str(rel),
                            "type": "verbatim",
                            "raw_size": f.stat().st_size,
                            "sha256": sha256_file(f),
                        }
                    )
            manifest = {
                "tool": "datapack",
                "tool_version": _datapack_version(),
                "created_at": datetime.now(UTC).isoformat(),
                "source_name": source.name,
                "codec": codec,
                "files": entries,
            }
            zf.writestr(ARCHIVE_MANIFEST, json.dumps(manifest, indent=2))

    atomic_write(archive, produce, overwrite=overwrite)

    return ArchiveResult(
        source=source,
        archive=archive,
        n_tiffs=n_tiffs,
        n_other=len(files) - n_tiffs,
        raw_bytes=raw_bytes,
        archive_bytes=archive.stat().st_size,
    )


def _safe_extract_member(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts)


def unpack(
    archive: Path | str,
    dest: Path | str,
    *,
    verify: bool = True,
) -> UnpackResult:
    """Extract an archive to ``dest`` (tree preserved) and re-verify integrity.

    TIFFs come out as valid (compressed) TIFFs ready to work on. With
    ``verify=True`` each file's hash is checked against the embedded manifest
    (decoded-pixel SHA-256 for TIFFs, byte SHA-256 otherwise).
    """
    archive = Path(archive)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        try:
            manifest = json.loads(zf.read(ARCHIVE_MANIFEST))
        except KeyError as exc:
            raise HistologyError(
                f"{archive}: not a datapack archive (no {ARCHIVE_MANIFEST})"
            ) from exc
        for name in zf.namelist():
            if not _safe_extract_member(name):
                raise HistologyError(f"{archive}: unsafe path in archive: {name}")
        zf.extractall(dest)

    n_files = len(manifest["files"])
    if verify:
        for entry in manifest["files"]:
            p = dest / entry["path"]
            if entry["type"] == "tiff":
                actual = decode_pixel_sha256(p)
                expected = entry["pixel_sha256"]
            else:
                actual = sha256_file(p)
                expected = entry["sha256"]
            if actual != expected:
                raise VerificationError(
                    f"{archive}: {entry['path']} failed verification after unpack"
                )

    return UnpackResult(archive=archive, dest=dest, n_files=n_files, verified=verify)


def repack(
    workdir: Path | str,
    archive: Path | str,
    *,
    codec: str = "deflate",
    level: int = 6,
    tile: tuple[int, int] | None = (256, 256),
    overwrite: bool = True,
) -> ArchiveResult:
    """Re-bundle a working directory (after editing) into an archive.

    Thin wrapper over :func:`pack` (defaults to ``overwrite=True``); picks up any
    files added during work (e.g. new segmentation outputs) and skips a leftover
    embedded manifest from a prior unpack.
    """
    return pack(
        workdir,
        archive,
        codec=codec,
        level=level,
        tile=tile,
        overwrite=overwrite,
    )


def _datapack_version() -> str | None:
    try:
        from datapack import __version__

        return __version__
    except Exception:  # pragma: no cover
        return None
