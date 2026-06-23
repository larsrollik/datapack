"""CLI tests (Phase 0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from datapack.cli import main


def test_cli_dry_run_detects_modality(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = tmp_path / "continuous.dat"
    src.write_bytes(b"0")
    rc = main(["compress", str(src), str(tmp_path / "out.cbin"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "modality=ephys" in out
    assert "engine ready" in out  # ephys engine landed in Phase 1


def test_cli_dry_run_video_engine_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = tmp_path / "clip.h264"  # video engine landed in Phase 3
    src.write_bytes(b"0")
    rc = main(["compress", str(src), str(tmp_path / "out.mp4"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "modality=video" in out
    assert "engine ready" in out


def test_cli_dry_run_unknown_modality(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("x")
    rc = main(["compress", str(src), str(tmp_path / "out.bin"), "--dry-run"])
    assert rc == 2
    assert "could not detect modality" in capsys.readouterr().err


def test_cli_real_run_unknown_modality_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src = tmp_path / "notes.txt"  # undetectable -> fail loud, never silent
    src.write_text("x")
    rc = main(["compress", str(src), str(tmp_path / "out.bin")])
    assert rc == 2
    assert "could not detect modality" in capsys.readouterr().err
