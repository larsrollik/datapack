"""Ephys engine — lossless Neuropixel/OpenEphys compression via IBL mtscomp.

The OpenEphys metadata helpers (``_oebin_stream_meta`` / ``_stream_meta_for_dat``)
and the OE-tree-preserving compression approach are lifted from the validated
``dreamteam_etl`` script and now owned here. ``dreamteam_etl`` becomes a consumer
of this engine.

Integrity gate (ROADMAP D6): mtscomp's ``check_after_compress=True`` decompresses
the freshly written ``.cbin`` and byte-compares it to the source, raising on any
mismatch. That *is* the bit-exact roundtrip — no separate sha256 pass is run. We
additionally record mtscomp's own ``sha1_uncompressed`` / ``sha1_compressed``
(stored in the ``.ch`` sidecar) in the manifest, so the downstream offload/tape
stage gets an integrity contract for free, re-provable without the original.

``mtscomp`` (+ ``numpy``, ``tqdm``) is an optional dependency: install
``datapack[ephys]``. It is imported lazily so the rest of datapack works without it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datapack.core.atomic import atomic_write, atomic_write_many
from datapack.core.detect import register_engine
from datapack.core.engine import Engine
from datapack.core.errors import DatapackError
from datapack.core.job import Job
from datapack.core.manifest import ManifestRecord, ManifestWriter
from datapack.core.verify import VerifyResult

# OpenEphys `continuous.dat` is always signed 16-bit little-endian.
OE_DTYPE = "int16"
_SHA1_CHUNK = 1 << 20


class EphysMetaError(DatapackError):
    """Raised when sample_rate/n_channels cannot be resolved for a source."""


@dataclass(frozen=True)
class StreamMeta:
    """What a raw `.dat`/`.bin` carries no header for.

    The first three are what mtscomp needs to compress. ``bit_volts``/``units`` ride
    along because this is the only place OE stream metadata is parsed, and consumers of
    a *compressed* session have no other route to them: SpikeInterface and neo both
    memmap ``continuous.dat`` during ``parse_header()`` and raise ``FileNotFoundError``
    once it has become ``continuous.cbin``. Scaling ADC counts to µV therefore depends
    on this being read here, at compression time.
    """

    sample_rate: float
    n_channels: int
    dtype: str = OE_DTYPE
    bit_volts: float | list[float] | None = None
    units: str | None = None


def _oebin_stream_meta(recording_dir: Path) -> dict[str, StreamMeta]:
    """Map ``folder_name`` -> StreamMeta from a recording's ``structure.oebin``."""
    oebin = recording_dir / "structure.oebin"
    meta: dict[str, StreamMeta] = {}
    if not oebin.exists():
        return meta
    data = json.loads(oebin.read_text())
    for stream in data.get("continuous", []):
        folder = str(stream.get("folder_name", "")).rstrip("/")
        if not folder:
            continue
        chans = stream.get("channels") or []
        gains = [float(c["bit_volts"]) for c in chans if c.get("bit_volts") is not None]
        # Scalar when every channel shares a gain (the normal case), the full list
        # otherwise, so a heterogeneous probe is never collapsed to a single number.
        bit_volts = (gains[0] if len(set(gains)) == 1 else gains) if gains else None
        meta[folder] = StreamMeta(
            sample_rate=float(stream["sample_rate"]),
            n_channels=int(stream["num_channels"]),
            bit_volts=bit_volts,
            units=(chans[0].get("units") if chans else None),
        )
    return meta


def oebin_stream_meta(recording_dir: Path | str) -> dict[str, StreamMeta]:
    """Public: ``folder_name`` -> :class:`StreamMeta` from a recording's
    ``structure.oebin``.

    Metadata only; no sample data is opened, so this works on a compressed session where
    ``continuous.dat`` has become ``continuous.cbin``.
    """
    return _oebin_stream_meta(Path(recording_dir))


def session_stream_meta(session_dir: Path | str) -> dict[str, StreamMeta]:
    """Same, but searches a whole session dir for its ``structure.oebin``."""
    session_dir = Path(session_dir)
    for oebin in sorted(session_dir.rglob("structure.oebin")):
        meta = _oebin_stream_meta(oebin.parent)
        if meta:
            return meta
    return {}


def _stream_meta_for_dat(dat: Path) -> StreamMeta | None:
    """Resolve StreamMeta for a ``continuous.dat`` from its enclosing oebin.

    Layout: ``.../recording<M>/continuous/<folder>/continuous.dat`` with the
    ``structure.oebin`` at the recording-dir level.
    """
    folder = dat.parent.name
    for up in dat.parents:
        if (up / "structure.oebin").exists():
            return _oebin_stream_meta(up).get(folder)
    return None


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_SHA1_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _lazy_mtscomp() -> Any:
    try:
        import mtscomp
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via message
        raise DatapackError(
            "the ephys engine needs mtscomp; install it with: "
            "pip install 'datapack[ephys]'"
        ) from exc
    # mtscomp is chatty (INFO logs); keep library use quiet.
    logging.getLogger("mtscomp").setLevel(logging.WARNING)
    return mtscomp


def verify_artifact(cbin: Path | str) -> VerifyResult:
    """Re-prove a `.cbin`/`.ch` artifact is intact, without needing the original.

    Decompresses the artifact and compares the SHA-1 of the decompressed stream
    to the ``sha1_uncompressed`` recorded by mtscomp in the ``.ch`` sidecar.
    """
    _lazy_mtscomp()  # import guard + quiet logging
    from mtscomp import Reader

    cbin = Path(cbin)
    ch = cbin.with_suffix(".ch")
    cmeta = json.loads(ch.read_text())
    stored = cmeta["sha1_uncompressed"]
    with tempfile.TemporaryDirectory(dir=cbin.parent) as td:
        out = Path(td) / "roundtrip.dat"
        reader = Reader()
        reader.open(str(cbin), str(ch))
        try:
            reader.tofile(str(out))
        finally:
            reader.close()
        actual = _sha1_file(out)
    return VerifyResult(
        ok=actual == stored,
        method="mtscomp_roundtrip_sha1",
        details={
            "sha1_uncompressed_stored": stored,
            "sha1_uncompressed_actual": actual,
        },
    )


@register_engine
class EphysEngine(Engine):
    """Lossless compression of flat int16 ephys binaries with IBL mtscomp."""

    modality = "ephys"
    name = "mtscomp"
    lossless = True

    def __init__(
        self,
        *,
        chunk_duration_s: float = 1.0,
        n_threads: int = 8,
    ) -> None:
        # NOTE: dreamteam_etl reported measurements at 120 s chunks while the
        # mtscomp API default is 1 s. We default to the API value explicitly;
        # callers tune via params. Larger chunks -> better ratio, coarser seek.
        self.chunk_duration_s = chunk_duration_s
        self.n_threads = n_threads

    def _resolve_meta(self, job: Job) -> StreamMeta:
        p = job.params
        if "sample_rate" in p and "n_channels" in p:
            return StreamMeta(
                sample_rate=float(p["sample_rate"]),
                n_channels=int(p["n_channels"]),
                dtype=str(p.get("dtype", OE_DTYPE)),
            )
        meta = _stream_meta_for_dat(job.src)
        if meta is None:
            raise EphysMetaError(
                f"cannot resolve sample_rate/n_channels for {job.src}: no "
                "structure.oebin found above it. Pass them in params, e.g. "
                "params={'sample_rate': 30000, 'n_channels': 384}."
            )
        return meta

    def compress(self, job: Job) -> ManifestRecord:
        mtscomp = _lazy_mtscomp()
        meta = self._resolve_meta(job)
        chunk = float(job.params.get("chunk_duration_s", self.chunk_duration_s))
        threads = int(job.params.get("n_threads", self.n_threads))

        cbin = job.dst
        ch = cbin.with_suffix(".ch")

        def produce(tmps: list[Path]) -> None:
            tmp_cbin, tmp_ch = tmps
            # check_after_compress=True is the integrity gate: mtscomp
            # decompresses + byte-compares to the source and raises on mismatch.
            mtscomp.compress(
                str(job.src),
                str(tmp_cbin),
                str(tmp_ch),
                sample_rate=meta.sample_rate,
                n_channels=meta.n_channels,
                dtype=meta.dtype,
                chunk_duration=chunk,
                n_threads=threads,
                check_after_compress=True,
            )

        atomic_write_many([cbin, ch], produce)

        cmeta = json.loads(ch.read_text())
        source_size = job.src.stat().st_size
        output_size = cbin.stat().st_size + ch.stat().st_size
        sha1_unc = cmeta.get("sha1_uncompressed")
        sha1_cmp = cmeta.get("sha1_compressed")

        extra: dict[str, Any] = {
            "sha1_uncompressed": sha1_unc,
            "sha1_compressed": sha1_cmp,
            "n_channels": meta.n_channels,
            "sample_rate": meta.sample_rate,
            "dtype": meta.dtype,
            "shape": cmeta.get("shape"),
            "mtscomp": {
                "algorithm": cmeta.get("algorithm"),
                "version": cmeta.get("version"),
                "comp_level": cmeta.get("comp_level"),
                "n_chunks": len(cmeta.get("chunk_bounds", [])) - 1,
            },
            "cbin_size": cbin.stat().st_size,
            "ch_size": ch.stat().st_size,
        }

        return ManifestRecord(
            source_path=job.src,
            source_size=source_size,
            modality=self.modality,
            engine=self.name,
            engine_version=getattr(mtscomp, "__version__", None),
            params={
                "chunk_duration_s": chunk,
                "n_threads": threads,
                "check_after_compress": True,
                "sample_rate": meta.sample_rate,
                "n_channels": meta.n_channels,
                "dtype": meta.dtype,
            },
            output_path=cbin,
            output_paths=[cbin, ch],
            output_size=output_size,
            ratio=output_size / source_size if source_size else None,
            lossless=True,
            verification=VerifyResult(
                ok=True,
                method="mtscomp_check_after_compress",
                details={"sha1_uncompressed": sha1_unc, "sha1_compressed": sha1_cmp},
            ),
            ready_to_offload=True,
            tool_version=_datapack_version(),
            verified_at=datetime.now(UTC),
            extra=extra,
        )

    def verify(self, job: Job) -> VerifyResult:
        """Re-prove the artifact at ``job.dst`` (a `.cbin`) roundtrips correctly."""
        return verify_artifact(job.dst)


@dataclass(frozen=True)
class SessionResult:
    """Summary of compressing one OpenEphys session tree."""

    source: Path
    dest: Path
    records: list[ManifestRecord]
    sidecars_copied: int
    skipped: list[Path]  # continuous.dat streams with no resolvable oebin meta
    raw_bytes: int
    output_bytes: int
    n_streams: int = 0  # total continuous.dat found (compressed + skipped)
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


def compress_oe_session(
    source: Path | str,
    dest: Path | str,
    *,
    n_threads: int = 8,
    chunk_duration_s: float = 1.0,
    copy_sidecars: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    manifest_path: Path | str | None = None,
) -> SessionResult:
    """Compress every ``continuous.dat`` in an OpenEphys session, preserving the tree.

    Mirrors the OE folder structure from ``source`` to ``dest``, replacing each
    ``continuous.dat`` with mtscomp ``.cbin``+``.ch`` and copying every other file
    (``structure.oebin``, ``*.npy``, ``events/``, ``sync_messages.txt``,
    ``settings.xml`` …) verbatim, so the result is still a self-describing OE
    session. ``sample_rate``/``n_channels`` come from each stream's
    ``structure.oebin``; a stream whose meta cannot be resolved is skipped (not
    failed) and listed in ``SessionResult.skipped``.

    The source is only read, never modified. Migration-specific concerns
    (scratch staging, remote stores, moves) are intentionally NOT handled here —
    that belongs to the caller's orchestration.
    """
    source = Path(source).resolve()
    dest = Path(dest)
    if not source.is_dir():
        raise EphysMetaError(f"not an OpenEphys session directory: {source}")

    engine = EphysEngine(chunk_duration_s=chunk_duration_s, n_threads=n_threads)
    dats = sorted(source.rglob("continuous.dat"))

    records: list[ManifestRecord] = []
    skipped: list[Path] = []
    raw_bytes = 0
    output_bytes = 0

    for dat in dats:
        rel = dat.relative_to(source)
        raw_bytes += dat.stat().st_size
        if _stream_meta_for_dat(dat) is None:
            skipped.append(rel)
            continue
        if dry_run:
            continue
        cbin = dest / rel.parent / "continuous.cbin"
        rec = engine.compress(Job(src=dat, dst=cbin, modality="ephys"))
        records.append(rec)
        output_bytes += rec.output_size or 0

    sidecars_copied = 0
    if copy_sidecars and not dry_run:
        for f in source.rglob("*"):
            if f.is_file() and f.name != "continuous.dat":
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

    return SessionResult(
        source=source,
        dest=dest,
        records=records,
        sidecars_copied=sidecars_copied,
        skipped=skipped,
        raw_bytes=raw_bytes,
        output_bytes=output_bytes,
        n_streams=len(dats),
        dry_run=dry_run,
    )


def decompress_cbin(
    cbin: Path | str,
    dst_dat: Path | str,
    *,
    overwrite: bool = False,
    verify: bool = False,
    n_threads: int | None = None,
) -> Path:
    """Restore a `.cbin`/`.ch` pair to a bit-exact flat `continuous.dat` (mtsdecomp).

    ``verify`` controls mtscomp's ``check_after_decompress``, which defaults to True in
    mtscomp and is OFF here. That pass re-reads the whole decompressed file and
    re-decompresses every chunk to compare, one chunk at a time with no thread pool -- so it
    costs roughly twice the decompress it is checking. Measured on a 384 ch / 4.6 h probe
    stream: 1.5 h to decompress, then 2.9 h to verify what had just been written.

    Leaving it off is not a loss of integrity cover, because the part that can actually rot
    is still checked. zlib validates every chunk as it inflates, so a damaged ``.cbin``
    raises during the decompress itself; ``check_after_compress`` already verified the
    archive when it was written; and these sessions were confirmed byte-identical when they
    were placed. What the pass uniquely catches is corruption of the freshly written output
    between write and read, which is the filesystem's job. Pass ``verify=True`` where that
    is worth two extra hours -- validating a new codec, or a suspect disk.

    ``n_threads`` sizes mtscomp's decompress batches (one chunk per thread). mtscomp defaults
    to ``mp.cpu_count()``, which on a cluster node is the machine's core count and NOT the
    job's cgroup allocation: a worker holding 8 CPUs was running 32-chunk batches, which is
    how a steady decompress came to wander between 8 and 13 s per batch. Callers under a
    scheduler should pass their real allocation.
    """
    _lazy_mtscomp()  # import guard + quiet logging
    from mtscomp import Reader

    cbin = Path(cbin)
    ch = cbin.with_suffix(".ch")
    dst_dat = Path(dst_dat)

    opts: dict[str, object] = {"check_after_decompress": bool(verify)}
    if n_threads:
        opts["n_threads"] = int(n_threads)

    def produce(tmp: Path) -> None:
        reader = Reader(**opts)
        reader.open(str(cbin), str(ch))
        try:
            reader.tofile(str(tmp))
        finally:
            reader.close()

    return atomic_write(dst_dat, produce, overwrite=overwrite)


def _stream_of(rel: Path) -> str | None:
    """The continuous stream folder a session-relative path belongs to, or None.

    OE nests streams as ``<Record Node>/experiment<i>/recording<j>/continuous/<stream>/...``.
    Anything outside a ``continuous/`` directory -- ``structure.oebin``, ``events/``,
    ``sync_messages.txt``, ``settings.xml`` -- belongs to no stream and is session-level.
    """
    parts = rel.parts
    for i, part in enumerate(parts[:-1]):
        if part == "continuous" and i + 1 < len(parts) - 1:
            return parts[i + 1]
    return None


def _filter_oebin(src: Path, dst: Path, keep: set[str]) -> None:
    """Copy ``structure.oebin``, narrowing its ``continuous`` list to ``keep``.

    Required, not cosmetic. neo/SpikeInterface build a memmap per entry in that list when
    the session is opened, so an entry whose ``continuous.dat`` was never expanded makes the
    open fail rather than simply hiding the stream.

    ``events`` is deliberately left whole: TTL directories are small, they are not what the
    decompress is avoiding, and the barcode/sync routes read every line of them.
    """
    meta = json.loads(src.read_text())
    entries = meta.get("continuous")
    if isinstance(entries, list):
        meta["continuous"] = [
            e for e in entries if str(e.get("folder_name", "")).rstrip("/") in keep
        ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(meta, indent=4))


def decompress_oe_session(
    source: Path | str,
    dest: Path | str,
    *,
    overwrite: bool = False,
    streams: Collection[str] | None = None,
    verify: bool = False,
    n_threads: int | None = None,
) -> SessionResult:
    """Restore a compressed OE session to a working tree (the round-trip of compress).

    Mirrors ``source`` to ``dest``, expanding each ``continuous.cbin``/``.ch`` back to
    a bit-exact ``continuous.dat`` and copying every other file verbatim — so the
    result is a normal OE session ready for SpikeInterface/KS4 in a scratch/tmp
    location. Inverse of :func:`compress_oe_session`.

    ``streams`` names the continuous stream folders to expand, e.g.
    ``{"Neuropix-PXI-100.ProbeA"}``; ``None`` expands all of them, which is the original
    behaviour and what LFP/EMG and a full restore want. Sorting wants one: a dual-probe
    session expands ~384 GB per probe plus the DAQ stream, so sorting one probe was paying
    for three streams and the two it never read dominated the job. Unselected streams are
    reported in ``SessionResult.skipped``.

    ``verify`` and ``n_threads`` are passed to :func:`decompress_cbin`; see there for why
    verification is off by default and why a scheduled job should state its CPU allocation.

    A stream name that matches nothing raises, rather than quietly producing a tree with no
    traces in it -- the sorter would then refuse for a reason several steps from the cause.
    """
    source = Path(source).resolve()
    dest = Path(dest)
    if not source.is_dir():
        raise EphysMetaError(f"not a compressed OE session directory: {source}")

    all_cbins = sorted(source.rglob("continuous.cbin"))
    available = {c.parent.name for c in all_cbins}
    if streams is None:
        keep = set(available)
    else:
        keep = {str(s).rstrip("/") for s in streams}
        if unknown := keep - available:
            raise EphysMetaError(
                f"no such stream(s) in {source}: {sorted(unknown)}; "
                f"available: {sorted(available)}"
            )

    cbins = [c for c in all_cbins if c.parent.name in keep]
    raw_bytes = 0
    output_bytes = 0
    records: list[ManifestRecord] = []
    for cbin in cbins:
        rel = cbin.relative_to(source)
        raw_bytes += cbin.stat().st_size
        dat = dest / rel.parent / "continuous.dat"
        out = decompress_cbin(
            cbin, dat, overwrite=overwrite, verify=verify, n_threads=n_threads
        )
        output_bytes += out.stat().st_size

    sidecars = 0
    narrowed = streams is not None and keep != available
    for f in source.rglob("*"):
        if not f.is_file() or f.suffix in {".cbin", ".ch"}:
            continue
        rel = f.relative_to(source)
        # A skipped stream's own sidecars go with it. sample_numbers.npy and timestamps.npy
        # are 4 GB each per stream, so copying them for streams nobody expanded is 16 GB of
        # pure cost on a dual-probe session.
        stream = _stream_of(rel)
        if stream is not None and stream not in keep:
            continue
        target = dest / rel
        if target.exists() and not overwrite:
            continue
        if narrowed and f.name == "structure.oebin":
            _filter_oebin(f, target, keep)
            sidecars += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        sidecars += 1

    return SessionResult(
        source=source,
        dest=dest,
        records=records,
        sidecars_copied=sidecars,
        skipped=sorted(
            c.parent.relative_to(source) for c in all_cbins if c.parent.name not in keep
        ),
        raw_bytes=raw_bytes,
        output_bytes=output_bytes,
        n_streams=len(cbins),
    )


def _datapack_version() -> str | None:
    try:
        from datapack import __version__

        return __version__
    except Exception:  # pragma: no cover
        return None
