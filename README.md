# datapack

Lossless and visually-lossless compression of acquisition data (ephys, video,
histology) with verification and manifests.

`datapack` moves raw acquisition data into a compressed state, **verifies** every
artifact, and writes a **manifest** so a downstream offload/tape process can move it
and delete the original with confidence. It only ever **reads** the source — it never
mutates or deletes the raw data.

## What it handles

| Modality | Lossless? | Engine | Typical saving | Notes |
|---|---|---|---|---|
| **Ephys** (Neuropixels / Open Ephys) | **lossless** (bit-exact) | IBL `mtscomp` (`.cbin`+`.ch`) | ~⅓ (NP1 ~59%) | round-trips back to identical `.dat` |
| **Video** (behavioural, H.264) | **visually** lossless | `ffmpeg` HEVC (NVENC) | ~35–65% | re-encode for DLC/SLEAP; the `.mp4` is the working form |
| **Histology** (TIFF: brainsaw/StitchIt, brainreg) | **lossless** (bit-exact) | `tifffile` ADOBE_DEFLATE + predictor | ~⅔ (labels 40–75×) | files stay natively readable; tracks/metadata copied verbatim |

**Design principles**

- **Compress only the large data.** Metadata, `.npy` tracks, recipes, logs, JSON/CSV
  are copied **byte-for-byte verbatim**.
- **Verify before trusting.** Lossless engines prove a bit-exact round-trip; video
  checks frame-count + clean decode. Verification gates the artifact before it lands.
- **Round-trip in place.** Lossless outputs stay readable / restorable, so you can work
  on them in scratch (spike-sorting, image processing) and recompress — no lock-in.
- **Never touch the source.** Write-to-temp → verify → atomic rename; originals untouched.
- **Manifests** record sizes, ratios, hashes, params and tool versions for the offload step.

## Installation

```sh
pip install datapack                 # core (CLI + histology-independent pieces)
pip install "datapack[ephys]"        # + mtscomp, numpy, tqdm (ephys engine)
pip install "datapack[histology]"    # + tifffile, imagecodecs, numpy (histology engine)
```

Video needs an **NVENC-enabled `ffmpeg`** on the system (not a Python dep). datapack
resolves it from `--ffmpeg`, then `$DATAPACK_FFMPEG`, then `PATH`. On a cluster, assume
the GPU is managed externally — datapack **fails loudly** if no working GPU encoder is
found rather than silently grinding on CPU (opt into the slow CPU path with `--allow-cpu`).

## CLI

Run `datapack <command> --help` for full flags. All commands take a source and a
destination and never modify the source.

### `compress` — one file, modality auto-detected
```sh
datapack compress <src> <dst> [--modality ephys|video|histology] [--dry-run] [--report manifest.jsonl]
```
Detects the modality from the path (`continuous.dat`→ephys, `.h264/.mp4`→video,
`.tif/.tiff`→histology) and runs the right engine. For tuned video/ephys runs prefer the
dedicated commands below.

### Ephys
```sh
# Compress a whole Open Ephys session: every continuous.dat -> .cbin/.ch,
# all sidecars (structure.oebin, *.npy, events/, settings.xml ...) copied verbatim.
datapack oe-session <session_dir> <dest_dir> [--threads N] [--chunk-duration S] \
                    [--no-sidecars] [--overwrite] [--dry-run] [--report manifest.jsonl]

# Restore a compressed session to a working/scratch tree (bit-exact .dat) for
# SpikeInterface / Kilosort. Inverse of oe-session.
datapack oe-restore <compressed_session_dir> <work_dir> [--overwrite]
```

### Video
```sh
# Visually-lossless HEVC re-encode (NVENC; libx265 CPU fallback only with --allow-cpu).
datapack video <src.h264> <dst.mp4> [--quality safe|balanced|small] [--qp N] [--fps F] \
               [--ffmpeg /path/to/ffmpeg] [--allow-cpu] [--report manifest.jsonl]
```
`--quality` presets (preset p7, GOP 100): `safe`=qp18, `balanced`=qp20 (default),
`small`=qp22. **Provisional** pending the lab's DLC/SLEAP tracking validation; `--qp`
overrides. Raw `.h264` has no timing, so `--fps` (default 100) is forced on input.

### Histology
```sh
# In-place lossless recompression of every TIFF in a tree (e.g. a brainreg
# registration/), copying non-TIFF (incl. brainreg .npy tracks) verbatim. Output
# files stay normal, natively-readable TIFFs (ADOBE_DEFLATE) — no unpack needed.
datapack histology tree <src_dir> <dst_dir> [--codec deflate|zstd] [--level N] \
                        [--no-tile] [--overwrite] [--dry-run] [--report manifest.jsonl]

# Optional bundling: recompress a TIFF dir then store it as ONE archive (adds no
# compression over `tree` — purely bundling for transfer/tape).
datapack histology pack   <tiff_dir> <archive.zip> [--codec ...] [--level N] [--overwrite] [--dry-run]
datapack histology unpack <archive.zip> <work_dir> [--no-verify]   # extract + re-verify
datapack histology repack <work_dir> <archive.zip> [--codec ...] [--level N]  # after editing
```

### Datasets (dataset-aware "pack and stash")
```sh
# Pack a brainsaw/StitchIt serial_section/<sample>/ sample. Default: recompress every
# TIFF in place (raw + registration) -> stays browsable; copy .npy tracks + recipe/
# logs/mat/ini verbatim. Use --archive-raw to bundle stitchedImages/downsampled
# into one zip each (no ratio benefit, only one-object-per-dir for transfer/tape).
datapack serial-section <sample_dir> <dest_dir> [--archive-raw] [--codec deflate|zstd] \
                        [--overwrite] [--dry-run]
```

## Python API

```python
import datapack

# one file (modality auto-detected or forced)
rec = datapack.compress_file("continuous.dat", "out.cbin",
                             params={"sample_rate": 30000, "n_channels": 384})
print(rec.ratio, rec.lossless, rec.verification.ok, rec.ready_to_offload)

# ephys session round-trip
datapack.compress_oe_session("session/", "compressed/")     # -> .cbin/.ch + sidecars
datapack.decompress_oe_session("compressed/", "scratch/")   # -> bit-exact .dat tree

# dataset-aware
datapack.pack_serial_section("serial_section/ap5l/", "packed/")

from datapack.engines import (
    EphysEngine, HistologyEngine, VideoEngine,   # Engine.compress(Job) -> ManifestRecord
    compress_tiff_tree, pack, unpack, repack,    # histology tree + archive
    probe_ffmpeg,                                 # video capability probe
)
from datapack.core import Job, ManifestRecord, ManifestWriter, read_manifest
```

Every engine returns a `ManifestRecord` (pydantic): source/output paths + sizes, ratio,
`lossless`, the `VerifyResult`, engine + tool versions, and engine-specific `extra` (e.g.
mtscomp `sha1_uncompressed`/`sha1_compressed`, or a TIFF decoded-pixel SHA-256). Write a
run's records to JSON-lines with `ManifestWriter` and read them back with `read_manifest`.

## Development setup

```sh
git clone https://github.com/MurineShiftWork/datapack.git
cd datapack
uv sync --extra dev
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Running tests

```sh
uv run pytest
```

Tests needing optional deps or a GPU skip automatically when unavailable. To exercise the
real NVENC video path, point `$DATAPACK_FFMPEG` at an nvenc-enabled `ffmpeg`.

## Release workflow

1. Work on a `feature/` or `fix/` branch, committing with `cz commit`
2. Open a PR — CI (lint + tests + secrets scan) must pass before merge
3. Merge to main → version bump and tag are created automatically
4. Tag triggers release: GitHub release + PyPI publish

## License

See [LICENSE](LICENSE).
