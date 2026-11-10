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


class _RefusingHandle:
    """A sink that takes part of a write then refuses (EAGAIN), a set number of times.

    Mirrors Python's BufferedWriter on a filesystem answering EAGAIN: it raises
    BlockingIOError carrying how many bytes it did take.
    """

    def __init__(self, refusals: int, take: int) -> None:
        self.data = bytearray()
        self.refusals = refusals
        self.take = take
        self.flush_refusals = 1
        self.closed = False

    def write(self, buffer: bytes | memoryview) -> int:
        chunk = bytes(buffer)
        if self.refusals:
            self.refusals -= 1
            accepted = chunk[: self.take]
            self.data += accepted
            raise BlockingIOError(
                11, "write could not complete without blocking", len(accepted)
            )
        self.data += chunk
        return len(chunk)

    def flush(self) -> None:
        if self.flush_refusals:
            self.flush_refusals -= 1
            raise BlockingIOError(11, "flush would block", 0)

    def close(self) -> None:
        self.closed = True


def test_retrying_writer_resends_only_what_was_refused(monkeypatch) -> None:
    from datapack.engines import ephys

    monkeypatch.setattr(ephys, "WRITE_RETRY_FIRST_S", 0.0)
    sink = _RefusingHandle(refusals=3, take=5)
    payload = bytes(range(256)) * 4
    with ephys._RetryingWriter(sink) as w:
        assert w.write(payload) == len(payload)
        w.write(b"tail")
        assert w.retries == 3
    # Every byte once, in order: resending accepted bytes would duplicate them.
    assert bytes(sink.data) == payload + b"tail"
    assert sink.closed


def test_retrying_writer_gives_up_after_its_budget(monkeypatch) -> None:
    import pytest

    from datapack.engines import ephys

    monkeypatch.setattr(ephys, "WRITE_RETRY_FIRST_S", 0.01)
    sink = _RefusingHandle(refusals=10_000, take=0)
    with pytest.raises(BlockingIOError):
        ephys._RetryingWriter(sink, budget_s=0.05).write(b"x" * 10)


def test_retrying_open_wraps_only_binary_writes(tmp_path) -> None:
    from datapack.engines import ephys

    path = tmp_path / "f.bin"
    with ephys._retrying_open(path, "wb") as fh:
        assert isinstance(fh, ephys._RetryingWriter)
        fh.write(b"abc")
    with ephys._retrying_open(path, "rb") as fh:
        assert not isinstance(fh, ephys._RetryingWriter)
        assert fh.read() == b"abc"
