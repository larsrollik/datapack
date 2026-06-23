"""Ephys engine tests: oebin metadata parsing (no heavy deps)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datapack.core import Job
from datapack.engines.ephys import (
    EphysEngine,
    EphysMetaError,
    StreamMeta,
    _oebin_stream_meta,
    _stream_meta_for_dat,
)


def _make_oe_tree(root: Path, *, folder: str = "Neuropix-PXI-100.0") -> Path:
    """Create a minimal OE recording tree and return the continuous.dat path."""
    rec = root / "recording1"
    dat_dir = rec / "continuous" / folder
    dat_dir.mkdir(parents=True)
    (rec / "structure.oebin").write_text(
        json.dumps(
            {
                "continuous": [
                    {
                        "folder_name": f"{folder}/",
                        "sample_rate": 30000,
                        "num_channels": 384,
                    }
                ]
            }
        )
    )
    dat = dat_dir / "continuous.dat"
    dat.write_bytes(b"\x00\x00" * 384 * 4)  # 4 samples x 384 ch int16
    return dat


def test_oebin_stream_meta(tmp_path: Path) -> None:
    dat = _make_oe_tree(tmp_path)
    rec = dat.parents[2]
    meta = _oebin_stream_meta(rec)
    assert meta["Neuropix-PXI-100.0"] == StreamMeta(
        sample_rate=30000.0, n_channels=384, dtype="int16"
    )


def test_stream_meta_for_dat_walks_up(tmp_path: Path) -> None:
    dat = _make_oe_tree(tmp_path)
    meta = _stream_meta_for_dat(dat)
    assert meta is not None
    assert meta.n_channels == 384
    assert meta.sample_rate == 30000.0


def test_resolve_meta_from_params(tmp_path: Path) -> None:
    src = tmp_path / "continuous.dat"
    src.write_bytes(b"\x00\x00" * 10)
    job = Job(
        src=src,
        dst=tmp_path / "out.cbin",
        params={"sample_rate": 25000, "n_channels": 32},
    )
    meta = EphysEngine()._resolve_meta(job)
    assert meta.n_channels == 32
    assert meta.sample_rate == 25000.0


def test_resolve_meta_missing_raises(tmp_path: Path) -> None:
    src = tmp_path / "continuous.dat"  # no oebin above it, no params
    src.write_bytes(b"\x00\x00" * 10)
    job = Job(src=src, dst=tmp_path / "out.cbin")
    with pytest.raises(EphysMetaError, match="structure.oebin"):
        EphysEngine()._resolve_meta(job)
