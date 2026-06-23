"""Phase 0 core contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import datapack
from datapack.config import DEFAULT_CONFIG, Config
from datapack.core import (
    Engine,
    EngineNotAvailable,
    Job,
    ManifestRecord,
    ManifestWriter,
    VerifyResult,
    atomic_write,
    available_modalities,
    detect_modality,
    get_engine_class,
    read_manifest,
    register_engine,
    sha256_file,
)
from datapack.core.detect import _REGISTRY


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #
def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    data = b"datapack" * 1000
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# atomic_write
# --------------------------------------------------------------------------- #
def test_atomic_write_success(tmp_path: Path) -> None:
    dst = tmp_path / "out" / "a.txt"
    out = atomic_write(dst, lambda tmp: tmp.write_text("hello"))
    assert out == dst
    assert dst.read_text() == "hello"
    # no temp leftovers
    assert list(dst.parent.glob(".*tmp")) == []


def test_atomic_write_failure_leaves_no_artifact(tmp_path: Path) -> None:
    dst = tmp_path / "a.txt"

    def boom(tmp: Path) -> None:
        tmp.write_text("partial")
        raise RuntimeError("encoder died")

    with pytest.raises(RuntimeError, match="encoder died"):
        atomic_write(dst, boom)
    assert not dst.exists()
    assert list(tmp_path.glob(".*tmp")) == []


def test_atomic_write_refuses_existing_without_overwrite(tmp_path: Path) -> None:
    dst = tmp_path / "a.txt"
    dst.write_text("old")
    with pytest.raises(FileExistsError):
        atomic_write(dst, lambda tmp: tmp.write_text("new"))
    assert dst.read_text() == "old"


def test_atomic_write_overwrite(tmp_path: Path) -> None:
    dst = tmp_path / "a.txt"
    dst.write_text("old")
    atomic_write(dst, lambda tmp: tmp.write_text("new"), overwrite=True)
    assert dst.read_text() == "new"


def test_atomic_write_raises_if_producer_writes_nothing(tmp_path: Path) -> None:
    dst = tmp_path / "a.txt"
    with pytest.raises(RuntimeError, match="did not create"):
        atomic_write(dst, lambda tmp: None)
    assert not dst.exists()


# --------------------------------------------------------------------------- #
# Job validation (core invariant)
# --------------------------------------------------------------------------- #
def test_job_requires_existing_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source does not exist"):
        Job(src=tmp_path / "nope.dat", dst=tmp_path / "out.cbin")


def test_job_dst_must_differ_from_src(tmp_path: Path) -> None:
    src = tmp_path / "continuous.dat"
    src.write_bytes(b"0")
    with pytest.raises(ValueError, match="must differ from source"):
        Job(src=src, dst=src)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("continuous.dat", "ephys"),
        ("rec.h264", "video"),
        ("clip.mp4", "video"),
        ("section.tif", "histology"),
        ("section.TIFF", "histology"),
        ("notes.txt", None),
    ],
)
def test_detect_modality(name: str, expected: str | None) -> None:
    assert detect_modality(Path("/data") / name) == expected


# --------------------------------------------------------------------------- #
# engine registry
# --------------------------------------------------------------------------- #
def test_get_engine_unregistered_raises() -> None:
    with pytest.raises(EngineNotAvailable) as ei:
        get_engine_class("bogus")
    assert ei.value.modality == "bogus"


def test_ephys_engine_is_registered() -> None:
    assert "ephys" in available_modalities()
    assert get_engine_class("ephys").name == "mtscomp"


def test_register_and_get_engine() -> None:
    class _Dummy(Engine):
        modality = "_dummy"
        name = "dummy"
        lossless = True

        def compress(self, job: Job) -> ManifestRecord:  # pragma: no cover
            raise NotImplementedError

        def verify(self, job: Job) -> VerifyResult:  # pragma: no cover
            raise NotImplementedError

    try:
        register_engine(_Dummy)
        assert get_engine_class("_dummy") is _Dummy
        assert "_dummy" in available_modalities()
    finally:
        _REGISTRY.pop("_dummy", None)


def test_histology_engine_is_registered() -> None:
    assert "histology" in available_modalities()
    assert get_engine_class("histology").name == "tifffile"


def test_all_modality_engines_registered() -> None:
    assert set(available_modalities()) >= {"ephys", "histology", "video"}


# --------------------------------------------------------------------------- #
# manifest round-trip
# --------------------------------------------------------------------------- #
def test_manifest_jsonl_roundtrip(tmp_path: Path) -> None:
    rec = ManifestRecord(
        source_path=Path("/data/continuous.dat"),
        source_size=1000,
        source_hash="abc",
        modality="ephys",
        engine="mtscomp",
        output_path=Path("/store/continuous.cbin"),
        output_size=400,
        ratio=0.4,
        lossless=True,
        verification=VerifyResult(ok=True, method="mtscomp_check"),
        ready_to_offload=True,
    )
    assert rec.saved_fraction == pytest.approx(0.6)

    mpath = tmp_path / "run" / "manifest.jsonl"
    writer = ManifestWriter(mpath)
    writer.append(rec)
    writer.append(rec)

    back = read_manifest(mpath)
    assert len(back) == 2
    assert back[0].engine == "mtscomp"
    assert back[0].verification is not None and back[0].verification.ok
    assert back[0].output_path == Path("/store/continuous.cbin")


# --------------------------------------------------------------------------- #
# compress_file wiring
# --------------------------------------------------------------------------- #
def test_compress_file_unknown_modality(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("x")
    with pytest.raises(datapack.ModalityDetectionError):
        datapack.compress_file(src, tmp_path / "out.bin")


def test_compress_file_unregistered_modality(tmp_path: Path) -> None:
    # forcing a modality with no engine fails loud, never silently skips.
    src = tmp_path / "x.bin"
    src.write_bytes(b"0" * 10)
    with pytest.raises(EngineNotAvailable):
        datapack.compress_file(src, tmp_path / "out", modality="bogus")


# --------------------------------------------------------------------------- #
# config defaults
# --------------------------------------------------------------------------- #
def test_config_defaults() -> None:
    assert isinstance(DEFAULT_CONFIG, Config)
    assert DEFAULT_CONFIG.ephys.codec == "mtscomp"
    assert DEFAULT_CONFIG.video.qp == 20
    assert DEFAULT_CONFIG.histology.codec == "deflate"
