# datapack

Lossless and visually-lossless compression of acquisition data (ephys, video, histology) with verification and manifests.

## Installation

```sh
pip install datapack-neuro
```

## Usage

_Add usage examples here._

## Development

```sh
git clone https://github.com/larsrollik/datapack.git
cd datapack
uv sync --extra dev
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pytest
```

## Docs

```sh
uv sync --extra docs
uv run mkdocs serve
```
