"""Serial-section dataset-aware pack driver.

Skipped if histology deps (tifffile/imagecodecs/numpy) are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datapack.cli import main
from datapack.datasets import pack_serial_section

pytest.importorskip("tifffile")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402
import tifffile  # noqa: E402


def _lowent_u16(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    # low-entropy so it actually compresses (ratio < 1)
    return (np.zeros(shape, dtype="uint16") + seed).astype("uint16")


def _make_sample(root: Path) -> Path:
    """A serial_section sample: raw TIFF dirs + registration (w/ npy) + metadata."""
    # raw acquisition (Tier 1 -> archives)
    (root / "stitchedImages_100" / "2").mkdir(parents=True)
    tifffile.imwrite(
        root / "stitchedImages_100" / "2" / "section_001_01.tif",
        _lowent_u16((256, 256), 1),
    )
    (root / "downsampled_stacks" / "025_micron").mkdir(parents=True)
    tifffile.imwrite(
        root / "downsampled_stacks" / "025_micron" / "ds_ch02.tif",
        _lowent_u16((8, 128, 128), 2),
    )
    (root / "downsampled_stacks" / "025_micron" / "ds_ch02.txt").write_text("meta")

    # registration (Tier 2 -> in-place recompress, npy verbatim)
    reg = root / "registration"
    (reg / "manual_segmentation" / "atlas_space" / "tracks").mkdir(parents=True)
    tifffile.imwrite(
        reg / "registered_atlas.tiff",
        np.zeros((8, 64, 64), dtype="uint32"),
    )
    np.save(
        reg / "manual_segmentation" / "atlas_space" / "tracks" / "track_0.npy",
        np.arange(30, dtype="float64").reshape(10, 3),
    )
    (reg / "brainreg.json").write_text('{"atlas":"allen_mouse_25um"}')

    # top-level metadata (verbatim)
    (root / "recipe_x.yml").write_text("sample:\n  ID: ap5\n")
    (root / "FINISHED").write_text("")
    return root


def test_pack_serial_section_default_is_deflate_in_place(tmp_path: Path) -> None:
    src = _make_sample(tmp_path / "sample")
    dest = tmp_path / "packed"

    result = pack_serial_section(src, dest)  # default: no archiving

    # nothing zipped; every TIFF recompressed in place, browsable
    assert result.archived_dirs == []
    assert not list(dest.glob("*.zip"))
    for rel in (
        "stitchedImages_100/2/section_001_01.tif",
        "downsampled_stacks/025_micron/ds_ch02.tif",
        "registration/registered_atlas.tiff",
    ):
        assert np.array_equal(tifffile.imread(dest / rel), tifffile.imread(src / rel))
        with tifffile.TiffFile(dest / rel) as tf:  # natively readable deflate
            assert int(tf.pages[0].compression) == 8

    # brainreg track .npy left BYTE-FOR-BYTE verbatim (not compressed)
    npy_rel = Path("registration/manual_segmentation/atlas_space/tracks/track_0.npy")
    assert (dest / npy_rel).read_bytes() == (src / npy_rel).read_bytes()
    # metadata verbatim; source untouched
    assert (dest / "recipe_x.yml").exists() and (dest / "FINISHED").exists()
    assert (src / "stitchedImages_100" / "2" / "section_001_01.tif").exists()


def test_pack_serial_section_archive_raw_opt_in(tmp_path: Path) -> None:
    src = _make_sample(tmp_path / "sample")
    dest = tmp_path / "packed"
    result = pack_serial_section(src, dest, archive_raw=True)
    # raw dirs bundled into one zip each; registration still in place
    assert sorted(p.name for p in result.archived_dirs) == [
        "downsampled_stacks.zip",
        "stitchedImages_100.zip",
    ]
    assert (dest / "stitchedImages_100.zip").exists()
    assert (dest / "registration" / "registered_atlas.tiff").exists()


def test_pack_serial_section_dry_run(tmp_path: Path) -> None:
    src = _make_sample(tmp_path / "sample")
    dest = tmp_path / "packed"
    result = pack_serial_section(src, dest, dry_run=True)
    assert result.dry_run
    assert result.archived_dirs == []  # default in-place
    assert len(result.recompressed_dirs) == 3  # stitched, downsampled, registration
    assert not dest.exists()


def test_cli_serial_section(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src = _make_sample(tmp_path / "sample")
    dest = tmp_path / "packed"
    assert main(["serial-section", str(src), str(dest)]) == 0
    out = capsys.readouterr().out
    assert "recompressed dir(s)" in out
    assert not list(dest.glob("*.zip"))  # in-place by default
    assert (dest / "stitchedImages_100" / "2" / "section_001_01.tif").exists()
