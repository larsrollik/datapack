"""Dataset-aware pack drivers.

The per-modality engines compress individual files/trees; these drivers know a
*specific dataset layout* and dispatch the right tool per sub-directory, leaving
everything else verbatim. The principle: compress only the large data, never the
small metadata/track files.

- Open Ephys binary tree → :func:`datapack.compress_oe_session` (compress every
  ``continuous.dat``, copy all sidecars verbatim). Already covered; not duplicated here.
- brainsaw/StitchIt ``serial_section/<sample>/`` → :func:`pack_serial_section` (below).

All compression is lossless: raw TIFFs are recompressed (ADOBE_DEFLATE + predictor,
pixel-verified) before archiving; derived TIFFs are recompressed in place; ``.npy``
tracks, recipes, logs, JSON/CSV are copied byte-for-byte.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from datapack.engines.histology import compress_tiff_tree, pack

#: Top-level sub-dirs of a serial_section sample that hold the big raw TIFFs and
#: are bundled into one archive each (Tier 1). Matched case-insensitively by prefix.
RAW_TIFF_DIR_PREFIXES = ("stitchedimages", "downsampled")


@dataclass(frozen=True)
class SerialSectionResult:
    """Summary of packing one serial_section sample."""

    source: Path
    dest: Path
    archived_dirs: list[Path] = field(default_factory=list)  # -> one .zip each (Tier 1)
    recompressed_dirs: list[Path] = field(
        default_factory=list
    )  # in-place TIFF (Tier 2)
    files_copied: int = 0  # top-level metadata copied verbatim
    raw_bytes: int = 0
    output_bytes: int = 0
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


def _is_raw_tiff_dir(name: str) -> bool:
    low = name.lower()
    return any(low.startswith(p) for p in RAW_TIFF_DIR_PREFIXES)


def _tree_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def pack_serial_section(
    source: Path | str,
    dest: Path | str,
    *,
    archive_raw: bool = False,
    codec: str = "deflate",
    overwrite: bool = False,
    dry_run: bool = False,
) -> SerialSectionResult:
    """Pack one brainsaw/StitchIt ``serial_section/<sample>/`` directory.

    Default = **deflate-in-place everywhere**: every TIFF (raw and derived) is
    recompressed losslessly in place so it stays a natively-readable file, with no
    unpack step needed to work on it. Zipping adds *no* compression over this (all
    the gain is the internal TIFF deflate+predictor) — it only bundles — so it is
    opt-in via ``archive_raw=True`` for when a single object per raw dir is wanted
    for transfer/tape.

    Dispatch per top-level entry:

    - directories → **recompress TIFFs in place** to ``dest/<name>`` (browsable),
      copying non-TIFF — including brainreg ``.npy`` tracks — **byte-for-byte verbatim**.
      With ``archive_raw=True``, ``stitchedImages*``/``downsampled*`` dirs are instead
      **bundled** to ``dest/<name>.zip``.
    - top-level files (``recipe_*.yml``, ``FINISHED``, ``*_Log.txt``, ``*.mat``,
      ``*.ini`` …) → copied **verbatim**.

    The source is only read, never modified.
    """
    source = Path(source).resolve()
    dest = Path(dest)
    if not source.is_dir():
        raise NotADirectoryError(f"not a serial_section directory: {source}")

    archived: list[Path] = []
    recompressed: list[Path] = []
    files_copied = 0
    raw_bytes = 0
    output_bytes = 0

    for entry in sorted(source.iterdir()):
        if entry.is_dir():
            raw_bytes += _tree_bytes(entry)
            if archive_raw and _is_raw_tiff_dir(entry.name):
                archive = dest / f"{entry.name}.zip"
                if not dry_run:
                    arc = pack(entry, archive, codec=codec, overwrite=overwrite)
                    output_bytes += arc.archive_bytes
                archived.append(archive)
            else:
                out_dir = dest / entry.name
                if not dry_run:
                    tree = compress_tiff_tree(
                        entry, out_dir, codec=codec, overwrite=overwrite
                    )
                    output_bytes += tree.output_bytes
                recompressed.append(out_dir)
        elif entry.is_file():
            raw_bytes += entry.stat().st_size
            if not dry_run:
                target = dest / entry.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not (target.exists() and not overwrite):
                    shutil.copy2(entry, target)
                    output_bytes += target.stat().st_size
            files_copied += 1

    return SerialSectionResult(
        source=source,
        dest=dest,
        archived_dirs=archived,
        recompressed_dirs=recompressed,
        files_copied=files_copied,
        raw_bytes=raw_bytes,
        output_bytes=output_bytes,
        dry_run=dry_run,
    )
