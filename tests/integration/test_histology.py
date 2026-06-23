"""Histology engine + TIFF archive workflow.

Skipped if the optional histology deps (tifffile/imagecodecs/numpy) are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datapack.cli import main
from datapack.core import Job
from datapack.engines.histology import (
    HistologyEngine,
    HistologyError,
    compress_tiff_tree,
    pack,
    repack,
    unpack,
    verify_artifact,
)

pytest.importorskip("tifffile")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402
import tifffile  # noqa: E402


def _u16(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 4000, size=shape)).astype("uint16")


def _make_stitched_dir(root: Path) -> Path:
    """A stitchedImages-like dir: 2 channels of uint16 sections + a sidecar."""
    for ch in ("2", "3"):
        d = root / ch
        d.mkdir(parents=True)
        tifffile.imwrite(d / "section_001_01.tif", _u16((512, 640), int(ch)))
        tifffile.imwrite(d / "section_002_01.tif", _u16((512, 640), int(ch) + 10))
        (d / "details").mkdir()
        (d / "details" / "tile_log.txt").write_text("ok")
    return root


# --------------------------------------------------------------------------- #
# per-file engine
# --------------------------------------------------------------------------- #
def test_histology_engine_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "section.tif"
    arr = _u16((1024, 1024))
    tifffile.imwrite(src, arr)
    dst = tmp_path / "out" / "section.tif"

    rec = HistologyEngine().compress(Job(src=src, dst=dst, modality="histology"))

    assert dst.exists()
    assert rec.lossless and rec.ready_to_offload and rec.verification.ok
    assert 0 < rec.ratio < 1  # actually smaller
    # output is still a valid TIFF decoding to identical pixels
    assert np.array_equal(tifffile.imread(dst), arr)
    # source untouched
    assert np.array_equal(tifffile.imread(src), arr)
    # re-proof without the original
    assert verify_artifact(dst, rec.extra["pixel_sha256"]).ok


def test_histology_engine_multipage(tmp_path: Path) -> None:
    src = tmp_path / "stack.tif"
    arr = _u16((20, 256, 256))  # multi-page (downsampled-stack shape)
    tifffile.imwrite(src, arr)
    dst = tmp_path / "stack_out.tif"
    rec = HistologyEngine().compress(Job(src=src, dst=dst, modality="histology"))
    assert rec.extra["n_pages"] == 20
    assert np.array_equal(tifffile.imread(dst), arr)


# --------------------------------------------------------------------------- #
# archive: pack / unpack / repack
# --------------------------------------------------------------------------- #
def test_pack_unpack_roundtrip(tmp_path: Path) -> None:
    src = _make_stitched_dir(tmp_path / "stitchedImages_100")
    archive = tmp_path / "stitched.zip"

    result = pack(src, archive)
    assert archive.exists()
    assert result.n_tiffs == 4 and result.n_other == 2  # 2 details/*.txt
    assert result.saved_fraction is not None and result.saved_fraction > 0

    work = tmp_path / "work"
    up = unpack(archive, work, verify=True)
    assert up.verified and up.n_files == 6
    # every unpacked TIFF decodes to the original pixels; tree preserved
    for ch in ("2", "3"):
        for name in ("section_001_01.tif", "section_002_01.tif"):
            assert np.array_equal(
                tifffile.imread(work / ch / name),
                tifffile.imread(src / ch / name),
            )
    assert (work / "2" / "details" / "tile_log.txt").read_text() == "ok"


def test_unpack_rejects_non_datapack_zip(tmp_path: Path) -> None:
    import zipfile

    bad = tmp_path / "plain.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("x.txt", "hi")
    with pytest.raises(HistologyError, match="not a datapack archive"):
        unpack(bad, tmp_path / "out")


def test_repack_picks_up_added_segmentation(tmp_path: Path) -> None:
    src = _make_stitched_dir(tmp_path / "stitchedImages_100")
    archive = tmp_path / "a.zip"
    pack(src, archive)
    work = tmp_path / "work"
    unpack(archive, work)

    # simulate manual-segmentation work: add a new file in the working dir
    (work / "2" / "segmentation.tif").parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(work / "2" / "segmentation.tif", _u16((128, 128), 99))

    archive2 = tmp_path / "a2.zip"
    res = repack(work, archive2)
    assert res.n_tiffs == 5  # 4 original + 1 new segmentation
    # and it round-trips
    work2 = tmp_path / "work2"
    up = unpack(archive2, work2, verify=True)
    assert up.verified and (work2 / "2" / "segmentation.tif").exists()


def test_pack_dry_run(tmp_path: Path) -> None:
    src = _make_stitched_dir(tmp_path / "stitchedImages_100")
    archive = tmp_path / "x.zip"
    res = pack(src, archive, dry_run=True)
    assert res.dry_run and res.n_tiffs == 4
    assert not archive.exists()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _make_registration_dir(root: Path) -> Path:
    """A brainreg-like registration/ with mixed dtypes + metadata sidecars."""
    root.mkdir(parents=True)
    tifffile.imwrite(
        root / "deformation_field_0.tiff",
        (np.random.default_rng(1).standard_normal((8, 64, 64))).astype("float32"),
    )
    tifffile.imwrite(
        root / "registered_atlas.tiff",
        np.random.default_rng(2).integers(0, 1300, size=(8, 64, 64)).astype("uint32"),
    )
    tifffile.imwrite(
        root / "registered_hemispheres.tiff",
        np.random.default_rng(3).integers(0, 3, size=(8, 64, 64)).astype("uint8"),
    )
    tifffile.imwrite(root / "downsampled.tiff", _u16((8, 64, 64), 4))
    (root / "volumes.csv").write_text("region,volume\nCP,1.0\n")
    (root / "brainreg.json").write_text('{"atlas":"allen_mouse_25um"}')
    return root


def test_tier2_tree_recompress_keeps_native_readability(tmp_path: Path) -> None:
    src = _make_registration_dir(tmp_path / "registration")
    dst = tmp_path / "registration_compressed"

    result = compress_tiff_tree(src, dst, manifest_path=tmp_path / "m.jsonl")
    assert result.n_tiffs == 4
    assert result.saved_fraction is not None and result.saved_fraction > 0

    # every recompressed TIFF: bit-identical pixels AND a standard Deflate codec
    # (natively readable by any TIFF reader, not an outer wrapper)
    for name in (
        "deformation_field_0.tiff",
        "registered_atlas.tiff",
        "registered_hemispheres.tiff",
        "downsampled.tiff",
    ):
        assert np.array_equal(tifffile.imread(dst / name), tifffile.imread(src / name))
        with tifffile.TiffFile(dst / name) as tf:
            # ADOBE_DEFLATE (tag 8) — the universally-readable deflate codec
            assert int(tf.pages[0].compression) == 8
    # metadata copied verbatim; source untouched
    assert (dst / "brainreg.json").read_text() == '{"atlas":"allen_mouse_25um"}'
    assert (src / "registered_atlas.tiff").exists()


def test_cli_histology_tree(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _make_registration_dir(tmp_path / "registration")
    dst = tmp_path / "out"
    assert main(["histology", "tree", str(src), str(dst)]) == 0
    assert "recompressed 4 TIFF" in capsys.readouterr().out
    assert (dst / "deformation_field_0.tiff").exists()


def test_cli_histology_pack_unpack(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = _make_stitched_dir(tmp_path / "stitchedImages_100")
    archive = tmp_path / "s.zip"
    assert main(["histology", "pack", str(src), str(archive)]) == 0
    assert "packed 4 TIFF" in capsys.readouterr().out
    assert archive.exists()

    assert main(["histology", "unpack", str(archive), str(tmp_path / "w")]) == 0
    assert "verified" in capsys.readouterr().out
