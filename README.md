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
| **Video** (behavioural, e.g. `.avi`) | **visually** lossless | `ffmpeg` HEVC (NVENC) | ~35–65% | re-encode for DLC/SLEAP; the `.mp4` is the working form |
| **Video** (raw Pi `.h264`) | **lossless** (identical coded slices) | `MP4Box` remux to `.mp4` | ~0% | archival form; frame rate from the acquisition metadata |
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
pip install datapack-neuro                # core (CLI + histology-independent pieces)
pip install "datapack-neuro[ephys]"       # + mtscomp, numpy, tqdm (ephys engine)
pip install "datapack-neuro[histology]"   # + tifffile, imagecodecs, numpy (histology engine)
```

The distribution is `datapack-neuro`; the import name is `datapack`. An unrelated
`datapack` project already holds that name on PyPI.

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
# Raw .h264: remux to .mp4 without re-encoding (MP4Box), at the acquisition frame rate.
# Anything else (e.g. .avi): visually-lossless HEVC re-encode (NVENC; libx265 CPU
# fallback only with --allow-cpu).
datapack video <src> <dst.mp4> [--mode auto|remux|encode] [--meta acq_meta.yaml|.json] \
               [--remuxer auto|mp4box|ffmpeg] [--mp4box /path/to/MP4Box] \
               [--quality safe|balanced|small] [--qp N] [--fps F] \
               [--ffmpeg /path/to/ffmpeg] [--allow-cpu] [--report manifest.jsonl]
```
`--mode auto` (default) picks by file type: raw `.h264`/`.264` is **remuxed**, every
other video is **re-encoded**. `--mode encode` forces the HEVC re-encode of a raw `.h264`.

**Remux.** The frame rate is read from the acquisition metadata next to the video, found
from the video name without its `.h264` (and a following `.video`):
`<base>.debug_acq_meta.yaml` (first non-null `framerate:` key) or `<base>.metadata.json`
(top-level `framerate`); `--meta` names the file explicitly. No frame rate is an error:
it is never guessed, and `--fps` is refused. The remux is `MP4Box -add <h264>:fps=<F>
-new <mp4>`; ffmpeg stream copy is used only when MP4Box is missing (with a warning) or
with `--remuxer ffmpeg`. The `.mp4` is accepted only when all of these hold, checked on
the temp file before it is renamed into place:

1. frame count: counted `.h264` packets equal the `.mp4` container frame count;
2. size: the `.mp4` is within 1 % of the `.h264`;
3. frame rate: the `.mp4` rate tag equals the metadata rate;
4. coded slices: SHA-256 over all NAL units of type 1 and 5, in decode order, is equal
   for the `.h264` and for the `.mp4` extracted to Annex-B (`-bsf:v h264_mp4toannexb`).
   SPS, PPS, AUD and SEI may differ (framing only).

The manifest record carries the frame rate and its metadata file, both slice hashes,
every check and `verdict`. datapack never deletes the source: `ready_to_offload=True`
with `verdict == "match"` is the evidence that the caller may delete the `.h264`. Frame
times stay in the TTL/timestamp files next to the video. Resolve MP4Box via `--mp4box`,
`$DATAPACK_MP4BOX` or PATH.

**Re-encode.** `--quality` presets (preset p7, GOP 100): `safe`=qp18, `balanced`=qp20
(default), `small`=qp22. **Provisional** pending the lab's DLC/SLEAP tracking
validation; `--qp` overrides. A raw `.h264` forced through `--mode encode` has no timing,
so `--fps` (default 100) is forced on input.

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
    remux_h264, verify_remux, framerate_for,      # raw .h264 remux
)
from datapack.core import Job, ManifestRecord, ManifestWriter, read_manifest
```

Every engine returns a `ManifestRecord` (pydantic): source/output paths + sizes, ratio,
`lossless`, the `VerifyResult`, engine + tool versions, and engine-specific `extra` (e.g.
mtscomp `sha1_uncompressed`/`sha1_compressed`, or a TIFF decoded-pixel SHA-256). Write a
run's records to JSON-lines with `ManifestWriter` and read them back with `read_manifest`.

## Development setup

```sh
git clone https://github.com/larsrollik/datapack.git
cd datapack
uv sync --extra dev
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Running tests

```sh
uv run pytest
```

Tests needing optional deps or a GPU skip automatically when unavailable. To exercise the
real NVENC video path, point `$DATAPACK_FFMPEG` at an nvenc-enabled `ffmpeg`. The remux
tests need an ffmpeg with an H.264 encoder (libx264 or libopenh264) and MP4Box (GPAC);
the `.avi` test needs hevc_nvenc or libx265.

## Release workflow

1. Work on a `feature/` or `fix/` branch, committing with `cz commit`
2. Open a PR — CI (lint + tests + secrets scan) must pass before merge
3. Merge to main → version bump and tag are created automatically
4. Tag triggers release: GitHub release + PyPI publish

## License

See [LICENSE](LICENSE).
