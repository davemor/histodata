"""Adapter for the PatchCamelyon (PCam) benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import pyarrow as pa

from histodata.models import DatasetMetadata, DatasetSource

ZENODO_RECORD = "https://zenodo.org/records/2546921"
ZENODO_FILES = "https://zenodo.org/api/records/2546921/files"

SPLIT_COUNTS = {"train": 262_144, "validation": 32_768, "test": 32_768}
LABELS = {"normal": 0, "tumor": 1}

INDEX_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("media_type", pa.string(), nullable=False),
        pa.field("item_index", pa.int32(), nullable=False),
        pa.field("label_path", pa.string(), nullable=False),
        pa.field("metadata_path", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("label_index", pa.int8(), nullable=False),
        pa.field("slide_id", pa.string(), nullable=False),
        pa.field("coord_x", pa.int32(), nullable=False),
        pa.field("coord_y", pa.int32(), nullable=False),
        pa.field("tumor_patch", pa.bool_(), nullable=False),
        pa.field("width", pa.int16(), nullable=False),
        pa.field("height", pa.int16(), nullable=False),
        pa.field("mpp_x", pa.float32(), nullable=False),
        pa.field("mpp_y", pa.float32(), nullable=False),
        pa.field("stain", pa.string(), nullable=False),
        pa.field("byte_size", pa.int64(), nullable=False),
    ]
)


def _source(
    filename: str,
    *,
    size: int,
    md5: str,
    split: str,
    kind: str,
) -> DatasetSource:
    target = filename.removesuffix(".gz")
    return DatasetSource(
        filename=filename,
        url=f"{ZENODO_FILES}/{filename}/content",
        size=size,
        checksum=f"md5:{md5}",
        split=split,
        cohort="PCam",
        expected_count=SPLIT_COUNTS[split],
        kind=kind,  # type: ignore[arg-type]
        target=f"pcam/{target}",
    )


class PCamAdapter:
    """Build and validate a package retaining PCam's native HDF5 storage."""

    metadata = DatasetMetadata(
        identifier="pcam",
        title="PatchCamelyon (PCam)",
        version="1.0",
        kind="patch",
        description=(
            "327,680 96x96 H&E lymph-node patches for binary metastasis classification, "
            "stored in the publisher's HDF5 containers."
        ),
        homepage="https://github.com/basveeling/pcam",
        doi="10.1007/978-3-030-00934-2_24",
        license_id="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        creators=(
            {"name": "Veeling, Bastiaan S.", "affiliation": "University of Amsterdam"},
            {"name": "Linmans, Jasper", "affiliation": "University of Amsterdam"},
            {"name": "Winkens, Jim", "affiliation": "University of Amsterdam"},
            {"name": "Cohen, Taco", "affiliation": "University of Amsterdam"},
            {"name": "Welling, Max", "affiliation": "University of Amsterdam"},
        ),
        citation=(
            "Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., & Welling, M. (2018). "
            "Rotation Equivariant CNNs for Digital Pathology. MICCAI 2018. "
            "https://doi.org/10.1007/978-3-030-00934-2_24"
        ),
    )

    sources = (
        _source(
            "camelyonpatch_level_2_split_train_x.h5.gz",
            size=6_421_353_462,
            md5="1571f514728f59376b705fc836ff4b63",
            split="train",
            kind="gzip",
        ),
        _source(
            "camelyonpatch_level_2_split_train_y.h5.gz",
            size=21_378,
            md5="35c2d7259d906cfc8143347bb8e05be7",
            split="train",
            kind="gzip",
        ),
        _source(
            "camelyonpatch_level_2_split_train_meta.csv",
            size=15_047_830,
            md5="5a3dd671e465cfd74b5b822125e65b0a",
            split="train",
            kind="file",
        ),
        _source(
            "camelyonpatch_level_2_split_valid_x.h5.gz",
            size=805_965_320,
            md5="d5b63470df7cfa627aeec8b9dc0c066e",
            split="validation",
            kind="gzip",
        ),
        _source(
            "camelyonpatch_level_2_split_valid_y.h5.gz",
            size=3_038,
            md5="2b85f58b927af9964a4c15b8f7e8f179",
            split="validation",
            kind="gzip",
        ),
        _source(
            "camelyonpatch_level_2_split_valid_meta.csv",
            size=1_850_810,
            md5="67589e00a4a37ec317f2d1932c7502ca",
            split="validation",
            kind="file",
        ),
        _source(
            "camelyonpatch_level_2_split_test_x.h5.gz",
            size=800_875_929,
            md5="d8c2d60d490dbd479f8199bdfa0cf6ec",
            split="test",
            kind="gzip",
        ),
        _source(
            "camelyonpatch_level_2_split_test_y.h5.gz",
            size=3_040,
            md5="60a7035772fbdb7f34eb86d4420cf66a",
            split="test",
            kind="gzip",
        ),
        _source(
            "camelyonpatch_level_2_split_test_meta.csv",
            size=1_608_584,
            md5="3455fd69135b66734e1008f3af684566",
            split="test",
            kind="file",
        ),
    )

    @property
    def labels(self) -> dict[str, int]:
        return dict(LABELS)

    def build_index(self, package_dir: Path) -> pa.Table:
        rows: list[dict[str, object]] = []
        for split in SPLIT_COUNTS:
            publisher_split = "valid" if split == "validation" else split
            prefix = f"camelyonpatch_level_2_split_{publisher_split}"
            image_path = package_dir / "data" / "pcam" / f"{prefix}_x.h5"
            label_path = package_dir / "data" / "pcam" / f"{prefix}_y.h5"
            metadata_path = package_dir / "data" / "pcam" / f"{prefix}_meta.csv"
            for path in (image_path, label_path, metadata_path):
                if not path.is_file():
                    raise ValueError(f"Expected PCam source file not found: {path}")

            image_relative = image_path.relative_to(package_dir).as_posix()
            label_relative = label_path.relative_to(package_dir).as_posix()
            metadata_relative = metadata_path.relative_to(package_dir).as_posix()
            image_size = image_path.stat().st_size
            with metadata_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for item_index, record in enumerate(reader):
                    label_index = int(_parse_bool(record, "center_tumor_patch"))
                    rows.append(
                        {
                            "id": f"{split}/{item_index:06d}",
                            "path": image_relative,
                            "split": split,
                            "media_type": "application/x-hdf5",
                            "item_index": item_index,
                            "label_path": label_relative,
                            "metadata_path": metadata_relative,
                            "label": "tumor" if label_index else "normal",
                            "label_index": label_index,
                            "slide_id": _required(record, "wsi"),
                            "coord_x": int(_required(record, "coord_x")),
                            "coord_y": int(_required(record, "coord_y")),
                            "tumor_patch": _parse_bool(record, "tumor_patch"),
                            "width": 96,
                            "height": 96,
                            "mpp_x": 0.972,
                            "mpp_y": 0.972,
                            "stain": "H&E",
                            "byte_size": image_size,
                        }
                    )

        rows.sort(key=lambda row: str(row["id"]))
        return pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)

    def manifest_dataset_metadata(self) -> dict[str, object]:
        return {
            "patch": {
                "width": 96,
                "height": 96,
                "label_region": {"width": 32, "height": 32, "position": "center"},
                "mpp_x": 0.972,
                "mpp_y": 0.972,
                "stain": "H&E",
            },
            "storage_layout": "hdf5-row",
            "labels": [{"name": label, "index": index} for label, index in LABELS.items()],
        }

    def validate_table(self, table: pa.Table) -> list[str]:
        errors: list[str] = []
        if table.schema != INDEX_SCHEMA:
            errors.append(f"PCam index schema differs from the required schema: {INDEX_SCHEMA}")
            return errors
        rows = table.to_pylist()
        for split, expected in SPLIT_COUNTS.items():
            split_rows = [row for row in rows if row["split"] == split]
            if len(split_rows) != expected:
                errors.append(f"Expected {expected} PCam {split} rows, found {len(split_rows)}")
            if [row["item_index"] for row in split_rows] != list(range(expected)):
                errors.append(f"PCam {split} item indexes are not contiguous from zero")
            if expected % 2 == 0:
                label_counts = {
                    label: sum(row["label"] == label for row in split_rows) for label in LABELS
                }
                for label, count in label_counts.items():
                    if count != expected // 2:
                        errors.append(
                            f"Expected {expected // 2} PCam {split} rows for {label}, found {count}"
                        )
            for row in split_rows:
                label = str(row["label"])
                if label not in LABELS or row["label_index"] != LABELS.get(label):
                    errors.append(f"Incorrect PCam label metadata for {row['id']}")
                if row["media_type"] != "application/x-hdf5":
                    errors.append(f"Incorrect PCam media_type for {row['id']}")
        return errors

    def deep_validate(self, package_dir: Path, table: pa.Table) -> list[str]:
        """Verify HDF5 shapes and that stored targets match the Parquet labels."""
        errors: list[str] = []
        for split, expected in SPLIT_COUNTS.items():
            rows = table.filter(pa.array([value == split for value in table["split"].to_pylist()]))
            if not rows.num_rows:
                continue
            image_path = package_dir / rows["path"][0].as_py()
            label_path = package_dir / rows["label_path"][0].as_py()
            try:
                with h5py.File(image_path, "r") as images:
                    dataset = images["x"]
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError("HDF5 member 'x' is not a dataset")
                    if dataset.shape != (expected, 96, 96, 3):
                        errors.append(f"Unexpected PCam {split} image shape: {dataset.shape}")
                    for start in range(0, expected, 4096):
                        _ = dataset[start : min(expected, start + 4096)]
                with h5py.File(label_path, "r") as labels:
                    dataset = labels["y"]
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError("HDF5 member 'y' is not a dataset")
                    if dataset.shape[0] != expected:
                        errors.append(f"Unexpected PCam {split} label shape: {dataset.shape}")
                    actual = dataset[:].reshape(-1).astype("int8").tolist()
                    if actual != rows["label_index"].to_pylist():
                        errors.append(f"PCam {split} HDF5 labels differ from the index")
            except (OSError, KeyError, ValueError) as exc:
                errors.append(f"Cannot verify PCam {split} HDF5 containers: {exc}")
        return errors


def _required(record: dict[str, str | None], name: str) -> str:
    value = record.get(name)
    if value is None or not value.strip():
        raise ValueError(f"PCam metadata field is missing or empty: {name}")
    return value.strip()


def _parse_bool(record: dict[str, str | None], name: str) -> bool:
    value = _required(record, name).lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValueError(f"Invalid PCam boolean value for {name}: {value}")
