"""Validation for standardized Histodata dataset packages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from histodata.models import DatasetAdapter
from histodata.package import FORMAT_NAME, FORMAT_VERSION
from histodata.registry import UnknownDatasetError, get_adapter

MAX_REPORTED_ERRORS = 100
CORE_COLUMNS = {
    "id": "string",
    "path": "string",
    "split": "string",
    "media_type": "string",
}


@dataclass(frozen=True)
class ValidationReport:
    package: Path
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checked_files: int
    deep: bool

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_package(
    package_dir: Path, *, deep: bool = False, adapter: DatasetAdapter | None = None
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    checked_files = 0
    package_dir = package_dir.resolve()

    manifest = _read_manifest(package_dir / "manifest.json", errors)
    if manifest is None:
        return ValidationReport(package_dir, tuple(errors), tuple(warnings), 0, deep)

    _validate_manifest_header(manifest, errors)
    dataset = manifest.get("dataset")
    dataset_id = dataset.get("id") if isinstance(dataset, dict) else None
    if adapter is None and isinstance(dataset_id, str):
        try:
            adapter = get_adapter(dataset_id)
        except UnknownDatasetError as exc:
            _add_error(errors, str(exc).strip("'"))
    elif adapter is not None and dataset_id != adapter.metadata.identifier:
        _add_error(errors, f"Manifest dataset id does not match adapter: {dataset_id}")

    index_section = manifest.get("index")
    if not isinstance(index_section, dict):
        _add_error(errors, "Manifest index section is missing or invalid")
        return ValidationReport(package_dir, tuple(errors), tuple(warnings), 0, deep)

    index_name = index_section.get("path")
    if not isinstance(index_name, str) or not _safe_relative_path(index_name):
        _add_error(errors, "Manifest index path must be a safe relative path")
        return ValidationReport(package_dir, tuple(errors), tuple(warnings), 0, deep)

    index_path = package_dir / index_name
    try:
        table = pq.read_table(index_path)
    except (OSError, pa.ArrowException) as exc:
        _add_error(errors, f"Cannot read Parquet index {index_path}: {exc}")
        return ValidationReport(package_dir, tuple(errors), tuple(warnings), 0, deep)

    _validate_table_schema(table, index_section, errors)
    declared_rows = index_section.get("row_count")
    if declared_rows != table.num_rows:
        _add_error(
            errors,
            f"Manifest row_count is {declared_rows}, Parquet index has {table.num_rows} rows",
        )

    if adapter is not None:
        for error in adapter.validate_table(table):
            _add_error(errors, error)

    checked_files = _validate_rows(package_dir, table, errors, deep=deep)
    return ValidationReport(
        package=package_dir,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checked_files=checked_files,
        deep=deep,
    )


def _read_manifest(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError:
        _add_error(errors, f"Manifest not found: {path}")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        _add_error(errors, f"Cannot read manifest {path}: {exc}")
        return None
    if not isinstance(value, dict):
        _add_error(errors, "Manifest root must be a JSON object")
        return None
    return value


def _validate_manifest_header(manifest: dict[str, Any], errors: list[str]) -> None:
    format_section = manifest.get("format")
    if not isinstance(format_section, dict):
        _add_error(errors, "Manifest format section is missing or invalid")
        return
    if format_section.get("name") != FORMAT_NAME:
        _add_error(errors, f"Unsupported format name: {format_section.get('name')}")
    if format_section.get("version") != FORMAT_VERSION:
        _add_error(errors, f"Unsupported format version: {format_section.get('version')}")


def _validate_table_schema(
    table: pa.Table, index_section: dict[str, Any], errors: list[str]
) -> None:
    fields = {field.name: field for field in table.schema}
    for name, expected_type in CORE_COLUMNS.items():
        field = fields.get(name)
        if field is None:
            _add_error(errors, f"Required index column is missing: {name}")
        elif str(field.type) != expected_type:
            _add_error(errors, f"Index column {name} must have type {expected_type}")
        elif field.nullable:
            _add_error(errors, f"Required index column must be non-nullable: {name}")

    declarations = index_section.get("columns")
    if not isinstance(declarations, list):
        _add_error(errors, "Manifest index columns declaration is missing or invalid")
        return

    declared_names: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, dict) or not isinstance(declaration.get("name"), str):
            _add_error(errors, "Invalid column declaration in manifest")
            continue
        name = declaration["name"]
        declared_names.add(name)
        field = fields.get(name)
        if field is None:
            _add_error(errors, f"Manifest declares missing Parquet column: {name}")
            continue
        if declaration.get("type") != str(field.type):
            _add_error(errors, f"Manifest type differs from Parquet type for column {name}")
        if declaration.get("nullable") != field.nullable:
            _add_error(errors, f"Manifest nullability differs for column {name}")

    undeclared = set(fields) - declared_names
    if undeclared:
        _add_error(errors, f"Parquet columns are not declared in manifest: {sorted(undeclared)}")

    for field in table.schema:
        if not field.nullable and table[field.name].null_count:
            _add_error(errors, f"Non-nullable column contains nulls: {field.name}")


def _validate_rows(package_dir: Path, table: pa.Table, errors: list[str], *, deep: bool) -> int:
    required = {"id", "path", "byte_size"}
    if not required.issubset(table.column_names):
        return 0

    ids = table["id"].to_pylist()
    if len(set(ids)) != len(ids):
        _add_error(errors, "Index contains duplicate item ids")

    paths = table["path"].to_pylist()
    sizes = table["byte_size"].to_pylist()
    checked = 0
    for item_id, raw_path, declared_size in zip(ids, paths, sizes, strict=True):
        if not isinstance(raw_path, str) or not _safe_relative_path(raw_path):
            _add_error(errors, f"Unsafe item path for {item_id}: {raw_path}")
            continue
        path = package_dir.joinpath(*PurePosixPath(raw_path).parts)
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            _add_error(errors, f"Indexed file does not exist: {raw_path}")
            continue
        if not resolved.is_relative_to(package_dir):
            _add_error(errors, f"Indexed path resolves outside package: {raw_path}")
            continue
        if not resolved.is_file():
            _add_error(errors, f"Indexed path is not a file: {raw_path}")
            continue
        checked += 1
        if resolved.stat().st_size != declared_size:
            _add_error(errors, f"File size differs from index for {raw_path}")
        if deep:
            _validate_image(resolved, raw_path, errors)
    return checked


def _validate_image(path: Path, display_path: str, errors: list[str]) -> None:
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
        if size != (224, 224):
            _add_error(errors, f"Expected 224x224 image at {display_path}, found {size}")
    except (OSError, ValueError) as exc:
        _add_error(errors, f"Cannot verify image {display_path}: {exc}")


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value.replace("\\", "/"))
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and not (path.parts and ":" in path.parts[0])
    )


def _add_error(errors: list[str], message: str) -> None:
    if len(errors) < MAX_REPORTED_ERRORS:
        errors.append(message)
    elif len(errors) == MAX_REPORTED_ERRORS:
        errors.append("Additional validation errors were suppressed")
