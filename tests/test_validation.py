from __future__ import annotations

import json
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import MiniDataset
from PIL import Image

from histodata.download import install_dataset
from histodata.validation import validate_package


def _installed_package(
    tmp_path: Path, mini_dataset: MiniDataset, transport: httpx.MockTransport
) -> Path:
    return install_dataset(mini_dataset.adapter, tmp_path, transport=transport, retry_delay=0)


def test_fast_and_deep_validation(
    tmp_path: Path, mini_dataset: MiniDataset, archive_transport: httpx.MockTransport
) -> None:
    package = _installed_package(tmp_path, mini_dataset, archive_transport)
    fast = validate_package(package, adapter=mini_dataset.adapter)
    deep = validate_package(package, deep=True, adapter=mini_dataset.adapter)
    assert fast.valid and fast.checked_files == 18 and not fast.deep
    assert deep.valid and deep.checked_files == 18 and deep.deep


def test_reports_missing_file(
    tmp_path: Path, mini_dataset: MiniDataset, archive_transport: httpx.MockTransport
) -> None:
    package = _installed_package(tmp_path, mini_dataset, archive_transport)
    next((package / "data").rglob("*.tif")).unlink()
    report = validate_package(package, adapter=mini_dataset.adapter)
    assert not report.valid
    assert any("does not exist" in error for error in report.errors)


def test_reports_corrupt_or_wrong_sized_image(
    tmp_path: Path, mini_dataset: MiniDataset, archive_transport: httpx.MockTransport
) -> None:
    package = _installed_package(tmp_path, mini_dataset, archive_transport)
    image = next((package / "data").rglob("*.tif"))
    Image.new("RGB", (10, 10)).save(image, format="TIFF")
    report = validate_package(package, deep=True, adapter=mini_dataset.adapter)
    assert not report.valid
    assert any("File size differs" in error for error in report.errors)
    assert any("Expected 224x224" in error for error in report.errors)


def test_reports_duplicate_ids(
    tmp_path: Path, mini_dataset: MiniDataset, archive_transport: httpx.MockTransport
) -> None:
    package = _installed_package(tmp_path, mini_dataset, archive_transport)
    index_path = package / "index.parquet"
    table = pq.read_table(index_path)
    ids = table["id"].to_pylist()
    ids[1] = ids[0]
    replacement = pa.chunked_array([pa.array(ids, type=pa.string())])
    table = table.set_column(
        table.schema.get_field_index("id"), table.schema.field("id"), replacement
    )
    pq.write_table(table, index_path)
    report = validate_package(package, adapter=mini_dataset.adapter)
    assert any("duplicate item ids" in error for error in report.errors)


def test_reports_manifest_version_mismatch(
    tmp_path: Path, mini_dataset: MiniDataset, archive_transport: httpx.MockTransport
) -> None:
    package = _installed_package(tmp_path, mini_dataset, archive_transport)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format"]["version"] = "99.0"
    manifest_path.write_text(json.dumps(manifest))
    report = validate_package(package, adapter=mini_dataset.adapter)
    assert any("Unsupported format version" in error for error in report.errors)
