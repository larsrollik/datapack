"""Session-level OE compression: tree-walk + sidecars + manifest.

Skipped if the optional ephys deps (mtscomp/numpy) are not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datapack.cli import main
from datapack.core import read_manifest
from datapack.engines.ephys import (
    compress_oe_session,
    decompress_oe_session,
)

pytest.importorskip("mtscomp")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402


def _write_dat(path: Path, *, n_samples: int, n_channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (
        np.random.default_rng(0).standard_normal((n_samples, n_channels)) * 50
    ).astype("int16")
    path.write_bytes(arr.tobytes())


def _make_session(root: Path) -> Path:
    """A recording with two continuous streams (AP+LFP) + sidecars."""
    rec = root / "recording1"
    cont = rec / "continuous"
    rec.mkdir(parents=True, exist_ok=True)
    ap, lfp = "Neuropix-PXI-100.0", "Neuropix-PXI-100.1"
    (rec / "structure.oebin").write_text(
        json.dumps(
            {
                "continuous": [
                    {"folder_name": ap + "/", "sample_rate": 30000, "num_channels": 8},
                    {"folder_name": lfp + "/", "sample_rate": 2500, "num_channels": 8},
                ]
            }
        )
    )
    _write_dat(cont / ap / "continuous.dat", n_samples=30000, n_channels=8)
    _write_dat(cont / lfp / "continuous.dat", n_samples=2500, n_channels=8)
    # sidecars that must be copied verbatim
    (cont / ap / "timestamps.npy").write_bytes(b"\x93NUMPY-fake")
    (rec / "sync_messages.txt").write_text("start")
    return root


def test_compress_oe_session(tmp_path: Path) -> None:
    src = _make_session(tmp_path / "session")
    dest = tmp_path / "compressed"
    report = tmp_path / "manifest.jsonl"

    result = compress_oe_session(src, dest, n_threads=2, manifest_path=report)

    assert len(result.records) == 2
    assert result.skipped == []
    assert (
        result.sidecars_copied == 3
    )  # structure.oebin, timestamps.npy, sync_messages.txt
    assert result.saved_fraction is not None and 0 < result.saved_fraction < 1

    # tree mirrored: each stream -> .cbin + .ch, sidecars copied verbatim
    ap = dest / "recording1" / "continuous" / "Neuropix-PXI-100.0"
    assert (ap / "continuous.cbin").exists() and (ap / "continuous.ch").exists()
    assert not (ap / "continuous.dat").exists()  # replaced, not copied
    assert (ap / "timestamps.npy").read_bytes() == b"\x93NUMPY-fake"
    assert (dest / "recording1" / "structure.oebin").exists()
    assert (dest / "recording1" / "sync_messages.txt").read_text() == "start"

    # source untouched
    assert (
        src / "recording1" / "continuous" / "Neuropix-PXI-100.0" / "continuous.dat"
    ).exists()

    # manifest persisted, one record per stream, with the free sha1 contract
    records = read_manifest(report)
    assert len(records) == 2
    assert all(r.engine == "mtscomp" and r.extra["sha1_uncompressed"] for r in records)


def test_compress_oe_session_skips_stream_without_meta(tmp_path: Path) -> None:
    src = _make_session(tmp_path / "session")
    # add a stray continuous.dat with no matching oebin entry
    orphan = src / "recording1" / "continuous" / "Unlisted-200.0" / "continuous.dat"
    _write_dat(orphan, n_samples=100, n_channels=4)
    dest = tmp_path / "compressed"

    result = compress_oe_session(src, dest, n_threads=2)
    assert len(result.records) == 2  # the two listed streams
    assert len(result.skipped) == 1
    assert result.skipped[0].parts[-2] == "Unlisted-200.0"


def test_compress_oe_session_dry_run(tmp_path: Path) -> None:
    src = _make_session(tmp_path / "session")
    dest = tmp_path / "compressed"
    result = compress_oe_session(src, dest, dry_run=True)
    assert result.dry_run is True
    assert result.records == []
    assert result.n_streams == 2  # both streams counted even in dry-run
    assert not dest.exists()  # nothing written


def test_cli_oe_session(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _make_session(tmp_path / "session")
    dest = tmp_path / "out"
    report = tmp_path / "m.jsonl"
    rc = main(
        ["oe-session", str(src), str(dest), "--threads", "2", "--report", str(report)]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "stream(s) compressed" in out
    cbin = dest / "recording1" / "continuous" / "Neuropix-PXI-100.0" / "continuous.cbin"
    assert cbin.exists()
    assert len(read_manifest(report)) == 2


def test_cli_oe_session_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _make_session(tmp_path / "session")
    rc = main(["oe-session", str(src), str(tmp_path / "out"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in out


def test_oe_session_roundtrip_is_bit_exact(tmp_path: Path) -> None:
    """compress_oe_session -> decompress_oe_session restores bit-identical .dat."""
    src = _make_session(tmp_path / "session")
    compressed = tmp_path / "compressed"
    restored = tmp_path / "restored"

    compress_oe_session(src, compressed, n_threads=2)
    result = decompress_oe_session(compressed, restored)

    assert result.n_streams == 2
    for ch in ("Neuropix-PXI-100.0", "Neuropix-PXI-100.1"):
        orig = src / "recording1" / "continuous" / ch / "continuous.dat"
        back = restored / "recording1" / "continuous" / ch / "continuous.dat"
        assert back.read_bytes() == orig.read_bytes()  # bit-exact round-trip
    # sidecars restored too (oebin, timestamps, sync_messages)
    assert (restored / "recording1" / "structure.oebin").exists()
    assert (restored / "recording1" / "sync_messages.txt").read_text() == "start"
