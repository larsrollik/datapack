"""Command-line interface: ``datapack compress <path>``.

Phase 0 wires the pipeline end-to-end — detect modality, build the job, hand off
to the engine, and append the manifest record. ``--dry-run`` reports the plan
without touching anything; with no engines registered yet, a real run fails
loudly with ``EngineNotAvailable`` (it never silently skips compression).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datapack import (
    __version__,
    compress_file,
    compress_oe_session,
    decompress_oe_session,
    pack_serial_section,
)
from datapack.core import (
    DatapackError,
    ManifestWriter,
    available_modalities,
    detect_modality,
)
from datapack.engines import compress_tiff_tree, pack, repack, unpack


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datapack",
        description="Verified compression of acquisition data.",
    )
    parser.add_argument(
        "--version", action="version", version=f"datapack {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compress", help="compress one source file")
    c.add_argument("source", type=Path, help="source file to compress")
    c.add_argument("dest", type=Path, help="artifact destination path")
    c.add_argument(
        "--modality",
        choices=["ephys", "video", "histology"],
        help="force modality (default: auto-detect from the source)",
    )
    c.add_argument(
        "--dry-run",
        action="store_true",
        help="report the plan (detected modality, target) without compressing",
    )
    c.add_argument(
        "--report",
        type=Path,
        help="append the manifest record to this JSON-lines file",
    )
    c.set_defaults(func=_cmd_compress)

    s = sub.add_parser(
        "oe-session",
        help="compress every continuous.dat in an OpenEphys session tree",
    )
    s.add_argument("source", type=Path, help="OpenEphys session directory")
    s.add_argument("dest", type=Path, help="destination directory (tree mirrored)")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--chunk-duration", type=float, default=1.0, dest="chunk_duration")
    s.add_argument(
        "--no-sidecars",
        action="store_true",
        help="do not copy non-continuous.dat files into the destination",
    )
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="walk + resolve meta only")
    s.add_argument(
        "--report",
        type=Path,
        help="write per-stream manifest records here (JSON-lines)",
    )
    s.set_defaults(func=_cmd_oe_session)

    r = sub.add_parser(
        "oe-restore",
        help="decompress a compressed OE session back to a working tree (round-trip)",
    )
    r.add_argument("source", type=Path, help="compressed OE session directory")
    r.add_argument("dest", type=Path, help="working/scratch dir to restore into")
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(func=_cmd_oe_restore)

    v = sub.add_parser("video", help="visually-lossless HEVC re-encode (NVENC)")
    v.add_argument("source", type=Path, help="video file (e.g. .h264)")
    v.add_argument("dest", type=Path, help="output .mp4")
    v.add_argument(
        "--qp", type=int, default=20, help="constqp value (18 safe..22 small)"
    )
    v.add_argument(
        "--quality",
        choices=["safe", "balanced", "small"],
        help="named QP preset (safe=18, balanced=20, small=22); overrides --qp. "
        "PROVISIONAL pending DLC validation",
    )
    v.add_argument("--fps", type=float, default=100.0, help="forced fps for raw .h264")
    v.add_argument(
        "--ffmpeg", help="path to an nvenc-enabled ffmpeg (or $DATAPACK_FFMPEG)"
    )
    v.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit the slow libx265 CPU fallback if no GPU (emits a warning)",
    )
    v.add_argument("--report", type=Path, help="append manifest record (JSON-lines)")
    v.set_defaults(func=_cmd_video)

    ss = sub.add_parser(
        "serial-section",
        help="pack a brainsaw/StitchIt serial_section sample (dataset-aware)",
    )
    ss.add_argument("source", type=Path, help="serial_section/<sample>/ directory")
    ss.add_argument("dest", type=Path, help="destination directory")
    ss.add_argument(
        "--archive-raw",
        action="store_true",
        help="bundle stitchedImages/downsampled into one zip each (default: recompress "
        "in place, browsable). Zipping adds no ratio — only one-object-per-dir "
        "bundling.",
    )
    ss.add_argument("--codec", choices=["deflate", "zstd"], default="deflate")
    ss.add_argument("--overwrite", action="store_true")
    ss.add_argument("--dry-run", action="store_true")
    ss.set_defaults(func=_cmd_serial_section)

    h = sub.add_parser("histology", help="TIFF archive workflow (pack/unpack/repack)")
    hsub = h.add_subparsers(dest="hcommand", required=True)

    hp = hsub.add_parser("pack", help="bundle a TIFF dir into one archive")
    hp.add_argument(
        "source", type=Path, help="TIFF directory (e.g. stitchedImages_100)"
    )
    hp.add_argument("archive", type=Path, help="output .zip archive")
    hp.add_argument("--codec", choices=["deflate", "zstd"], default="deflate")
    hp.add_argument("--level", type=int, default=6)
    hp.add_argument("--overwrite", action="store_true")
    hp.add_argument("--dry-run", action="store_true")
    hp.set_defaults(func=_cmd_hist_pack)

    hu = hsub.add_parser("unpack", help="extract an archive to a working dir + verify")
    hu.add_argument("archive", type=Path)
    hu.add_argument("dest", type=Path, help="working directory")
    hu.add_argument("--no-verify", action="store_true")
    hu.set_defaults(func=_cmd_hist_unpack)

    hr = hsub.add_parser("repack", help="re-bundle a working dir after editing")
    hr.add_argument("workdir", type=Path)
    hr.add_argument("archive", type=Path)
    hr.add_argument("--codec", choices=["deflate", "zstd"], default="deflate")
    hr.add_argument("--level", type=int, default=6)
    hr.set_defaults(func=_cmd_hist_repack)

    ht = hsub.add_parser(
        "tree",
        help="recompress every TIFF in a tree in place (Tier 2; stays browsable)",
    )
    ht.add_argument(
        "source", type=Path, help="dir tree (e.g. a brainreg registration/)"
    )
    ht.add_argument("dest", type=Path, help="destination tree (structure mirrored)")
    ht.add_argument("--codec", choices=["deflate", "zstd"], default="deflate")
    ht.add_argument("--level", type=int, default=6)
    ht.add_argument(
        "--no-tile",
        action="store_true",
        help="write stripped (not tiled) TIFFs for maximal legacy-reader compatibility",
    )
    ht.add_argument("--overwrite", action="store_true")
    ht.add_argument("--dry-run", action="store_true")
    ht.add_argument("--report", type=Path, help="per-file manifest (JSON-lines)")
    ht.set_defaults(func=_cmd_hist_tree)
    return parser


def _cmd_compress(args: argparse.Namespace) -> int:
    modality = args.modality or detect_modality(args.source)
    if modality is None:
        print(
            f"error: could not detect modality for {args.source}; pass --modality",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        engines = available_modalities()
        engine_state = "engine ready" if modality in engines else "no engine yet"
        print(f"[dry-run] {args.source} -> {args.dest}")
        print(f"          modality={modality} ({engine_state})")
        return 0

    record = compress_file(args.source, args.dest, modality=modality)
    if args.report:
        ManifestWriter(args.report).append(record)
    saved = record.saved_fraction
    saved_str = f"{saved:.1%} saved" if saved is not None else "ratio n/a"
    print(f"ok: {args.source} -> {record.output_path} ({saved_str})")
    return 0


def _cmd_oe_session(args: argparse.Namespace) -> int:
    result = compress_oe_session(
        args.source,
        args.dest,
        n_threads=args.threads,
        chunk_duration_s=args.chunk_duration,
        copy_sidecars=not args.no_sidecars,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        manifest_path=args.report,
    )
    if args.dry_run:
        n_ok = result.n_streams - len(result.skipped)
        print(
            f"[dry-run] {args.source}: {result.n_streams} continuous.dat stream(s) "
            f"({n_ok} compressible, {len(result.skipped)} without meta)"
        )
    else:
        saved = result.saved_fraction
        saved_str = f"{saved:.1%} saved" if saved is not None else "ratio n/a"
        print(
            f"ok: {len(result.records)} stream(s) compressed ({saved_str}), "
            f"{result.sidecars_copied} sidecar(s) copied -> {result.dest}"
        )
    if result.skipped:
        print(f"warning: skipped {len(result.skipped)} stream(s) with no oebin meta:")
        for rel in result.skipped:
            print(f"  - {rel}")
    return 0


def _cmd_oe_restore(args: argparse.Namespace) -> int:
    result = decompress_oe_session(args.source, args.dest, overwrite=args.overwrite)
    print(
        f"ok: restored {result.n_streams} stream(s) + "
        f"{result.sidecars_copied} sidecar(s) -> {result.dest}"
    )
    return 0


def _cmd_video(args: argparse.Namespace) -> int:
    record = compress_file(
        args.source,
        args.dest,
        modality="video",
        params={
            "qp": args.qp,
            "quality": args.quality,
            "fps": args.fps,
            "ffmpeg": args.ffmpeg,
            "allow_cpu": args.allow_cpu,
        },
    )
    if args.report:
        ManifestWriter(args.report).append(record)
    saved = record.saved_fraction
    saved_str = f"{saved:.1%} saved" if saved is not None else "ratio n/a"
    v = record.verification
    fc = v.details.get("frame_count_match") if v else None
    print(
        f"ok: {args.source} -> {record.output_path} ({saved_str}; "
        f"encoder={record.extra.get('encoder')}, frames_match={fc})"
    )
    return 0


def _cmd_hist_pack(args: argparse.Namespace) -> int:
    result = pack(
        args.source,
        args.archive,
        codec=args.codec,
        level=args.level,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(
            f"[dry-run] {args.source}: {result.n_tiffs} TIFF(s) + "
            f"{result.n_other} other file(s) -> {args.archive}"
        )
        return 0
    saved = result.saved_fraction
    saved_str = f"{saved:.1%} saved" if saved is not None else "ratio n/a"
    print(
        f"ok: packed {result.n_tiffs} TIFF(s) + {result.n_other} other -> "
        f"{result.archive} ({saved_str})"
    )
    return 0


def _cmd_hist_unpack(args: argparse.Namespace) -> int:
    result = unpack(args.archive, args.dest, verify=not args.no_verify)
    verified = "verified" if result.verified else "NOT verified"
    print(f"ok: unpacked {result.n_files} file(s) -> {result.dest} ({verified})")
    return 0


def _cmd_hist_repack(args: argparse.Namespace) -> int:
    result = repack(args.workdir, args.archive, codec=args.codec, level=args.level)
    saved = result.saved_fraction
    saved_str = f"{saved:.1%} saved" if saved is not None else "ratio n/a"
    print(
        f"ok: repacked {result.n_tiffs} TIFF(s) + {result.n_other} other -> "
        f"{result.archive} ({saved_str})"
    )
    return 0


def _cmd_serial_section(args: argparse.Namespace) -> int:
    result = pack_serial_section(
        args.source,
        args.dest,
        archive_raw=args.archive_raw,
        codec=args.codec,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(
            f"[dry-run] {args.source}: "
            f"{len(result.archived_dirs)} dir(s) to archive, "
            f"{len(result.recompressed_dirs)} dir(s) to recompress, "
            f"{result.files_copied} top-level file(s) to copy"
        )
        return 0
    saved = result.saved_fraction
    saved_str = f"{saved:.1%} saved" if saved is not None else "ratio n/a"
    print(
        f"ok: {len(result.archived_dirs)} archive(s), "
        f"{len(result.recompressed_dirs)} recompressed dir(s), "
        f"{result.files_copied} file(s) copied -> {result.dest} ({saved_str})"
    )
    return 0


def _cmd_hist_tree(args: argparse.Namespace) -> int:
    result = compress_tiff_tree(
        args.source,
        args.dest,
        codec=args.codec,
        level=args.level,
        tile=None if args.no_tile else (256, 256),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        manifest_path=args.report,
    )
    if args.dry_run:
        print(
            f"[dry-run] {args.source}: {result.n_tiffs} TIFF(s) would be recompressed"
        )
        return 0
    saved = result.saved_fraction
    saved_str = f"{saved:.1%} saved" if saved is not None else "ratio n/a"
    print(
        f"ok: recompressed {len(result.records)} TIFF(s) + "
        f"{result.sidecars_copied} sidecar(s) -> {result.dest} ({saved_str})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DatapackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
