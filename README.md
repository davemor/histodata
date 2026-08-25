# Histodata

Histodata downloads, standardizes, and validates externally hosted
histopathology datasets. It stores each dataset as a versioned JSON manifest,
a typed Parquet index, and the source data in its publisher-provided layout.

The first built-in adapter installs the color-normalized
[Kather100K dataset](https://zenodo.org/records/1214456):
NCT-CRC-HE-100K for training and CRC-VAL-HE-7K for validation.

## Install

Install the CLI directly from the repository with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/davemor/histodata
```

For development:

```bash
uv sync
uv run histodata --help
```

## Configure the datasets root

Every download is placed in a canonical subdirectory of a datasets root. Pass
the root explicitly:

```bash
histodata download kather100k --output /data/histopathology
```

For regular use, set `HISTODATA_DATASETS_ROOT` in the environment:

```bash
export HISTODATA_DATASETS_ROOT=/data/histopathology
histodata download kather100k
```

Alternatively, create `./.env` in the directory from which the command is run:

```dotenv
HISTODATA_DATASETS_ROOT=/data/histopathology
```

An explicit `--output` takes precedence over the process environment, which
takes precedence over `./.env`. Histodata does not search parent directories
for `.env` files. Relative roots are resolved from the current directory and
`~` is expanded.

## Commands

Discover available datasets and inspect their pinned sources without network
access:

```bash
histodata list
histodata info kather100k
```

Download Kather100K. The verified ZIP files are deleted after extraction by
default to save approximately 12.5 GB:

```bash
histodata download kather100k --output /data/histopathology
histodata download kather100k --output /data/histopathology --keep-archives
```

Downloads use resumable `.part` files and verify the publisher-provided archive
sizes and MD5 checksums before extraction. Re-running a download for a valid,
complete package is safe and performs no network request.

Validate the manifest, Parquet schema, metadata, and every indexed file path:

```bash
histodata validate /data/histopathology/kather100k
```

Deep validation additionally opens all 107,180 TIFF images and checks their
integrity and dimensions:

```bash
histodata validate /data/histopathology/kather100k --deep
```

## Dataset package format

```text
kather100k/
├── manifest.json
├── index.parquet
├── data/
│   ├── NCT-CRC-HE-100K/<class>/*.tif
│   └── CRC-VAL-HE-7K/<class>/*.tif
└── archives/                         # only with --keep-archives
```

`manifest.json` identifies the `histodata-dataset` format version and records
the dataset citation, licence, labels, typed index schema, source URLs,
checksums, and installation provenance. `index.parquet` contains one row per
patch. Its required core columns are:

- `id`: stable ID independent of the installation path
- `path`: POSIX path relative to the package directory
- `split`: publisher-defined split
- `media_type`: file media type

Adapters may declare additional typed columns. Kather100K records its cohort,
label, label index, dimensions, resolution, stain, color-normalization status,
and file size. Future WSI adapters will use the same core contract; derived
embedding artifacts can align with source data through `id`.

## Development and quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest --cov
uv build
```

The automated suite uses miniature local archives and never downloads the full
dataset. A full installation is intentionally a manual integration test because
the two official archives total approximately 12.5 GB.

## Dataset attribution and licence

Kather100K is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite:

> Kather, J. N., Halama, N., & Marx, A. (2018). *100,000 histological images
> of human colorectal cancer and healthy tissue* (v0.1). Zenodo.
> https://doi.org/10.5281/zenodo.1214456

## Project licence

Histodata is available under the MIT licence. See `LICENSE`.
