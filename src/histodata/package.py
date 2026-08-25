"""Write standardized Histodata package metadata and indexes."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from histodata import __version__
from histodata.models import DatasetAdapter

FORMAT_NAME = "histodata-dataset"
FORMAT_VERSION = "1.0.0"
MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "index.parquet"

COLUMN_DESCRIPTIONS = {
    "id": "Stable, location-independent identifier for this dataset item.",
    "path": "POSIX path to the item, relative to the dataset package directory.",
    "split": "Publisher-defined dataset split.",
    "media_type": "IANA media type of the indexed file.",
    "cohort": "Publisher-defined source cohort.",
    "label": "Publisher-provided tissue class label.",
    "label_index": "Stable integer corresponding to label.",
    "width": "Image width in pixels.",
    "height": "Image height in pixels.",
    "mpp_x": "Horizontal microns per pixel.",
    "mpp_y": "Vertical microns per pixel.",
    "stain": "Histological stain.",
    "color_normalized": "Whether color normalization was applied by the publisher.",
    "byte_size": "Extracted file size in bytes.",
    "item_index": "Zero-based row offset within a container file.",
    "label_path": "Relative path to the publisher's label container.",
    "metadata_path": "Relative path to the publisher's item metadata.",
    "slide_id": "Publisher-provided source whole-slide image identifier.",
    "coord_x": "Patch x-coordinate in the source whole-slide image.",
    "coord_y": "Patch y-coordinate in the source whole-slide image.",
    "tumor_patch": "Whether any tumor occurs within the full source patch.",
    "central_path": "Relative path to the labelled central patch.",
    "context_paths": "Relative paths to the surrounding context patches.",
    "organ": "Organ represented by the patch.",
}


def write_package(adapter: DatasetAdapter, package_dir: Path, *, keep_archives: bool) -> None:
    """Build the index and atomically publish package metadata."""
    table = adapter.build_index(package_dir)
    adapter_errors = adapter.validate_table(table)
    if adapter_errors:
        raise ValueError("Cannot publish invalid dataset index: " + "; ".join(adapter_errors))

    index_path = package_dir / INDEX_FILENAME
    index_temporary = package_dir / f".{INDEX_FILENAME}.tmp"
    manifest_path = package_dir / MANIFEST_FILENAME
    manifest_temporary = package_dir / f".{MANIFEST_FILENAME}.tmp"

    index_temporary.unlink(missing_ok=True)
    manifest_temporary.unlink(missing_ok=True)
    pq.write_table(table, index_temporary, compression="zstd")
    os.replace(index_temporary, index_path)

    manifest = build_manifest(adapter, table, keep_archives=keep_archives)
    with manifest_temporary.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(manifest_temporary, manifest_path)


def build_manifest(
    adapter: DatasetAdapter, table: pa.Table, *, keep_archives: bool
) -> dict[str, object]:
    metadata = adapter.metadata
    dataset: dict[str, object] = {
        "id": metadata.identifier,
        "title": metadata.title,
        "version": metadata.version,
        "kind": metadata.kind,
        "description": metadata.description,
        "homepage": metadata.homepage,
        "doi": metadata.doi,
        "license": {"id": metadata.license_id, "url": metadata.license_url},
        "creators": list(metadata.creators),
        "citation": metadata.citation,
    }
    dataset.update(adapter.manifest_dataset_metadata())

    columns = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
            "description": COLUMN_DESCRIPTIONS.get(
                field.name, "Dataset adapter-defined metadata column."
            ),
        }
        for field in table.schema
    ]
    sources = []
    for source in adapter.sources:
        source_data = asdict(source)
        source_data["local_archive"] = f"archives/{source.filename}" if keep_archives else None
        sources.append(source_data)

    return {
        "format": {"name": FORMAT_NAME, "version": FORMAT_VERSION},
        "dataset": dataset,
        "index": {
            "path": INDEX_FILENAME,
            "row_count": table.num_rows,
            "primary_key": "id",
            "path_column": "path",
            "columns": columns,
        },
        "storage": {"data_root": "data"},
        "sources": sources,
        "installation": {
            "created_at": datetime.now(UTC).isoformat(),
            "histodata_version": __version__,
            "archives_retained": keep_archives,
        },
    }
