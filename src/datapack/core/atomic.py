"""Atomic, source-preserving artifact writes.

Every engine produces its output via :func:`atomic_write` (single file) or
:func:`atomic_write_many` (a set that must land together, e.g. mtscomp's
``.cbin`` + ``.ch``). The ``producer`` writes to temp paths in the destination
directory, and only on success are they renamed into place. On any error the
temp files are removed and the destinations are left untouched. The source is
never passed to the producer, so it cannot be mutated here.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _temp_for(dst: Path) -> Path:
    # A temp name that does not yet exist: some external encoders refuse to
    # overwrite an existing (even empty) file.
    return dst.parent / f".{dst.name}.{uuid.uuid4().hex}.tmp"


def atomic_write_many(
    dsts: Sequence[Path | str],
    producer: Callable[[list[Path]], object],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write several files via temps, then rename them all into place.

    Parameters
    ----------
    dsts
        Final destination paths (order is preserved into ``producer``).
    producer
        Callable given the list of temp paths to write, in the same order.
    overwrite
        If False (default), raise ``FileExistsError`` if any destination exists.

    Returns
    -------
    list[Path]
        The destination paths on success.

    Notes
    -----
    The temps are written and existence-checked before any rename, so a failing
    producer leaves nothing in place. The renames themselves are per-file (POSIX
    has no multi-file atomic rename); the window between them is a single local
    ``rename`` each and is treated as acceptable.
    """
    targets = [Path(d) for d in dsts]
    for d in targets:
        if d.exists() and not overwrite:
            raise FileExistsError(f"destination exists: {d}")
    for d in targets:
        d.parent.mkdir(parents=True, exist_ok=True)

    tmps = [_temp_for(d) for d in targets]
    try:
        producer(tmps)
        for t in tmps:
            if not t.exists():
                raise RuntimeError(
                    f"producer did not create the expected output at {t}"
                )
        for t, d in zip(tmps, targets, strict=True):
            t.replace(d)
        return targets
    except BaseException:
        for t in tmps:
            t.unlink(missing_ok=True)
        raise


def atomic_write(
    dst: Path | str,
    producer: Callable[[Path], object],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a single file via a temp then atomically rename to ``dst``."""
    [out] = atomic_write_many(
        [dst], lambda tmps: producer(tmps[0]), overwrite=overwrite
    )
    return out
