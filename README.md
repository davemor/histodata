# Histodata

Histodata downloads, standardizes, and validates externally hosted
histopathology datasets. It stores each dataset as a versioned JSON manifest,
a typed Parquet index, and the source data in its publisher-provided layout.

The built-in adapters install the normalized and non-normalized
[Kather100K datasets](https://zenodo.org/records/1214456),
[PatchCamelyon (PCam)](https://github.com/basveeling/pcam), and the four
organ-specific [SPIDER datasets](https://github.com/histai/spider) published
by HistAI.

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
histodata download kather100k-nonorm --output /data/histopathology
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

SPIDER is gated on Hugging Face. First accept access on the page for the organ
dataset, create a read token, and expose it as `HF_TOKEN`:

```bash
export HF_TOKEN=hf_your_read_token
histodata download spider-colorectal
```

The token can instead be stored alongside the datasets root in `./.env`:

```dotenv
HISTODATA_DATASETS_ROOT=/data/histopathology
HF_TOKEN=hf_your_read_token
```

An existing shell token overrides `./.env`. Do not commit `.env`; it is ignored
by Git.

## Commands

Discover available datasets and inspect their pinned sources without network
access:

```bash
histodata list
histodata info kather100k
histodata info pcam
histodata info spider-skin
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

Install PCam from the pinned Zenodo mirror:

```bash
histodata download pcam
```

PCam remains in its official HDF5 representation. Histodata decompresses the
three image and label containers, preserves the three slide-coordinate CSVs,
and uses `item_index` to address individual patches without creating 327,680
extra files.

Install any SPIDER organ after accepting its Hugging Face access conditions:

```bash
histodata download spider-skin
histodata download spider-colorectal
histodata download spider-thorax
histodata download spider-breast
```

SPIDER downloads are pinned to exact repository revisions. The Hub client
resumes multipart downloads, Histodata safely extracts the combined tar stream,
and the index stores one row per labelled central patch plus its 24 context
paths. Source archives range from about 56 GB for thorax to about 242 GB for
skin, so installation needs room for both archives and extracted data.

Validate the manifest, Parquet schema, metadata, and every indexed file path:

```bash
histodata validate /data/histopathology/kather100k
```

Deep validation additionally opens all 107,180 TIFF images and checks their
integrity and dimensions:

```bash
histodata validate /data/histopathology/kather100k --deep
```

For PCam, deep validation reads every HDF5 image and verifies container shapes
and labels. For SPIDER, it verifies every central and distinct context image.

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

The non-normalized variant uses the same package contract under
`kather100k-nonorm/`, contains only the publisher's 100,000-image training
cohort, and records `color_normalized=false` in its manifest and index.

PCam uses a container-aware variant of the same contract:

```text
pcam/
├── manifest.json
├── index.parquet
└── data/pcam/
    ├── camelyonpatch_level_2_split_{train,valid,test}_x.h5
    ├── camelyonpatch_level_2_split_{train,valid,test}_y.h5
    └── camelyonpatch_level_2_split_{train,valid,test}_meta.csv
```

Each SPIDER organ is independently installable:

```text
spider-<organ>/
├── manifest.json
├── index.parquet
└── data/spider-<organ>/
    ├── metadata.json
    └── images/...
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
label, dimensions, resolution, stain, color-normalization status, and file
size. PCam adds its HDF5 row offset, source-slide coordinates, and auxiliary
label/metadata paths. SPIDER adds organ, slide, and context paths. Future WSI
adapters will use the same core contract; derived embedding artifacts can align
with source data through `id`.

## Development and quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest --cov
uv build
```

The automated suite uses miniature local ZIP, gzip, HDF5, and multipart-tar
fixtures and never downloads a full dataset. Full installations are manual
integration tests because the official releases range from roughly 8 GB to
hundreds of gigabytes.

## Dataset attribution and licence

Kather100K is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite:

> Kather, J. N., Halama, N., & Marx, A. (2018). *100,000 histological images
> of human colorectal cancer and healthy tissue* (v0.1). Zenodo.
> https://doi.org/10.5281/zenodo.1214456

PCam data is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Cite Veeling et
al., *Rotation Equivariant CNNs for Digital Pathology*, MICCAI 2018,
[doi:10.1007/978-3-030-00934-2_24](https://doi.org/10.1007/978-3-030-00934-2_24).

SPIDER is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for research
use. Cite Nechaev, Pchelnikov, and Ivanova, *SPIDER: A Comprehensive Multi-Organ
Supervised Pathology Dataset and Baseline Models*,
[arXiv:2503.02876](https://arxiv.org/abs/2503.02876).

## Project licence

Histodata is available under the MIT licence. See `LICENSE`.
