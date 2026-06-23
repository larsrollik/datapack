"""Ephys engine integration test: a real mtscomp roundtrip on synthetic int16.

Skipped if the optional ephys deps (mtscomp/numpy) are not installed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import datapack
from datapack.core import Job, read_manifest
from datapack.engines.ephys import EphysEngine, verify_artifact

pytest.importorskip("mtscomp")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402


def _make_dat(path: Path, *, n_samples: int, n_channels: int) -> bytes:
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal((n_samples, n_channels)) * 50).astype("int16")
    data = arr.tobytes()
    path.write_bytes(data)
    return data


def test_ephys_compress_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "continuous.dat"
    raw = _make_dat(src, n_samples=30000, n_channels=8)
    src_sha = hashlib.sha256(raw).hexdigest()
    cbin = tmp_path / "out" / "continuous.cbin"

    rec = EphysEngine(chunk_duration_s=1.0, n_threads=2).compress(
        Job(src=src, dst=cbin, params={"sample_rate": 30000, "n_channels": 8})
    )

    # artifacts exist as a pair; source untouched
    ch = cbin.with_suffix(".ch")
    assert cbin.exists() and ch.exists()
    assert hashlib.sha256(src.read_bytes()).hexdigest() == src_sha  # not mutated

    # manifest record is well-formed and marks ready-to-offload
    assert rec.modality == "ephys" and rec.engine == "mtscomp"
    assert rec.lossless is True and rec.ready_to_offload is True
    assert rec.verification is not None and rec.verification.ok
    assert rec.output_size and rec.output_size < rec.source_size  # actually smaller
    assert 0 < rec.ratio < 1
    # mtscomp's stored sha1s captured for the offload-stage contract
    assert rec.extra["sha1_uncompressed"] == hashlib.sha1(raw).hexdigest()
    assert rec.extra["sha1_compressed"] == hashlib.sha1(cbin.read_bytes()).hexdigest()
    assert rec.extra["n_channels"] == 8

    # standalone re-proof (no original needed) round-trips
    vr = verify_artifact(cbin)
    assert vr.ok and vr.method == "mtscomp_roundtrip_sha1"


def test_ephys_failure_leaves_no_partial_artifact(tmp_path: Path) -> None:
    # Wrong n_channels so the byte count is inconsistent -> mtscomp errors;
    # the atomic pair-write must leave nothing behind.
    src = tmp_path / "continuous.dat"
    _make_dat(src, n_samples=1000, n_channels=8)
    cbin = tmp_path / "continuous.cbin"
    with pytest.raises(Exception):  # noqa: B017 - any mtscomp failure
        EphysEngine().compress(
            Job(src=src, dst=cbin, params={"sample_rate": 30000, "n_channels": 7})
        )
    assert not cbin.exists()
    assert not cbin.with_suffix(".ch").exists()
    assert list(tmp_path.glob(".*tmp")) == []


def test_compress_file_dispatches_to_ephys(tmp_path: Path) -> None:
    src = tmp_path / "continuous.dat"
    _make_dat(src, n_samples=3000, n_channels=4)
    cbin = tmp_path / "continuous.cbin"
    report = tmp_path / "manifest.jsonl"

    rec = datapack.compress_file(
        src, cbin, params={"sample_rate": 30000, "n_channels": 4}
    )
    assert rec.engine == "mtscomp"

    # and the manifest round-trips through the writer/reader
    from datapack.core import ManifestWriter

    ManifestWriter(report).append(rec)
    [back] = read_manifest(report)
    assert back.extra["sha1_uncompressed"] == rec.extra["sha1_uncompressed"]
