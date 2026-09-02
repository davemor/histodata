"""Adapters for the Kather colorectal patch datasets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow as pa

from histodata.models import DatasetMetadata, DatasetSource

ZENODO_RECORD = "https://zenodo.org/records/1214456"

LABELS = {
    "ADI": 0,
    "BACK": 1,
    "DEB": 2,
    "LYM": 3,
    "MUC": 4,
    "MUS": 5,
    "NORM": 6,
    "STR": 7,
    "TUM": 8,
}

INDEX_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("media_type", pa.string(), nullable=False),
        pa.field("cohort", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("label_index", pa.int8(), nullable=False),
        pa.field("width", pa.int16(), nullable=False),
        pa.field("height", pa.int16(), nullable=False),
        pa.field("mpp_x", pa.float32(), nullable=False),
        pa.field("mpp_y", pa.float32(), nullable=False),
        pa.field("stain", pa.string(), nullable=False),
        pa.field("color_normalized", pa.bool_(), nullable=False),
        pa.field("byte_size", pa.int64(), nullable=False),
    ]
)


class Kather100KAdapter:
    """Build and validate a standardized Kather100K package."""

    color_normalized = True

    metadata = DatasetMetadata(
        identifier="kather100k",
        title="100,000 histological images of human colorectal cancer and healthy tissue",
        version="v0.1",
        kind="patch",
        tissue="colorectal",
        image_size=(224, 224),
        resolution_mpp=0.5,
        train_count=100_000,
        test_count=0,
        valid_count=7_180,
        description="Color-normalized H&E colorectal tissue patches.",
        homepage=ZENODO_RECORD,
        doi="10.5281/zenodo.1214456",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        creators=(
            {
                "name": "Kather, Jakob Nikolas",
                "affiliation": "National Center for Tumor Diseases, Heidelberg",
                "orcid": "0000-0002-3730-5348",
            },
            {
                "name": "Halama, Niels",
                "affiliation": "National Center for Tumor Diseases, Heidelberg",
            },
            {
                "name": "Marx, Alexander",
                "affiliation": (
                    "Institute of Pathology, University Medical Center Mannheim, Mannheim, Germany"
                ),
            },
        ),
        citation=(
            "Kather, J. N., Halama, N., & Marx, A. (2018). 100,000 histological "
            "images of human colorectal cancer and healthy tissue (v0.1). Zenodo. "
            "https://doi.org/10.5281/zenodo.1214456"
        ),
    )

    sources = (
        DatasetSource(
            filename="NCT-CRC-HE-100K.zip",
            url=("https://zenodo.org/api/records/1214456/files/NCT-CRC-HE-100K.zip/content"),
            size=11_690_284_003,
            checksum="md5:6fd702d11df6292bc054397ae038a464",
            split="train",
            cohort="NCT-CRC-HE-100K",
            expected_count=100_000,
        ),
        DatasetSource(
            filename="CRC-VAL-HE-7K.zip",
            url=("https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content"),
            size=800_276_929,
            checksum="md5:2fd1651b4f94ebd818ebf90ad2b6ce06",
            split="validation",
            cohort="CRC-VAL-HE-7K",
            expected_count=7_180,
        ),
    )

    @property
    def labels(self) -> dict[str, int]:
        return dict(LABELS)

    def build_index(self, package_dir: Path) -> pa.Table:
        columns: dict[str, list[object]] = {field.name: [] for field in INDEX_SCHEMA}

        for source in self.sources:
            cohort_root = package_dir / "data" / source.cohort
            self._validate_cohort_layout(cohort_root, source)

            for label, label_index in LABELS.items():
                for image_path in sorted((cohort_root / label).glob("*.tif")):
                    relative_path = image_path.relative_to(package_dir).as_posix()
                    values: dict[str, object] = {
                        "id": f"{source.split}/{label}/{image_path.stem}",
                        "path": relative_path,
                        "split": source.split,
                        "media_type": "image/tiff",
                        "cohort": source.cohort,
                        "label": label,
                        "label_index": label_index,
                        "width": 224,
                        "height": 224,
                        "mpp_x": 0.5,
                        "mpp_y": 0.5,
                        "stain": "H&E",
                        "color_normalized": self.color_normalized,
                        "byte_size": image_path.stat().st_size,
                    }
                    for name, value in values.items():
                        columns[name].append(value)

        arrays = [pa.array(columns[field.name], type=field.type) for field in INDEX_SCHEMA]
        return pa.Table.from_arrays(arrays, schema=INDEX_SCHEMA)

    def manifest_dataset_metadata(self) -> dict[str, object]:
        return {
            "patch": {
                "width": 224,
                "height": 224,
                "mpp_x": 0.5,
                "mpp_y": 0.5,
                "stain": "H&E",
                "color_normalized": self.color_normalized,
            },
            "labels": [{"name": label, "index": index} for label, index in LABELS.items()],
        }

    def validate_table(self, table: pa.Table) -> list[str]:
        errors: list[str] = []
        if table.schema != INDEX_SCHEMA:
            errors.append(
                f"Kather100K index schema differs from the required schema: {INDEX_SCHEMA}"
            )

        rows = table.to_pylist()
        source_by_cohort = {source.cohort: source for source in self.sources}
        counts: dict[str, int] = {}
        for row in rows:
            cohort = str(row.get("cohort"))
            source = source_by_cohort.get(cohort)
            if source is None:
                errors.append(f"Unknown Kather100K cohort in index: {cohort}")
                continue
            counts[cohort] = counts.get(cohort, 0) + 1
            label = str(row.get("label"))
            if label not in LABELS:
                errors.append(f"Unknown Kather100K label in index: {label}")
            elif row.get("label_index") != LABELS[label]:
                errors.append(f"Incorrect label_index for {label}: {row.get('label_index')}")
            if row.get("split") != source.split:
                errors.append(f"Incorrect split for cohort {cohort}: {row.get('split')}")
            if row.get("media_type") != "image/tiff":
                errors.append(f"Incorrect media_type for {row.get('id')}")

        for source in self.sources:
            actual = counts.get(source.cohort, 0)
            if actual != source.expected_count:
                errors.append(
                    f"Expected {source.expected_count} rows for {source.cohort}, found {actual}"
                )
        return _deduplicate(errors)

    def deep_validate(self, package_dir: Path, table: pa.Table) -> list[str]:
        """Kather images are verified by the format-level image validator."""
        return []

    def _validate_cohort_layout(self, cohort_root: Path, source: DatasetSource) -> None:
        if not cohort_root.is_dir():
            raise ValueError(f"Expected cohort directory not found: {cohort_root}")

        actual_directories = {path.name for path in cohort_root.iterdir() if path.is_dir()}
        if actual_directories != set(LABELS):
            raise ValueError(
                f"Unexpected class directories in {cohort_root}: "
                f"expected {sorted(LABELS)}, found {sorted(actual_directories)}"
            )

        unexpected_files = [
            path
            for path in cohort_root.rglob("*")
            if path.is_file() and path.suffix.lower() != ".tif"
        ]
        if unexpected_files:
            raise ValueError(f"Unexpected non-TIFF file in {cohort_root}: {unexpected_files[0]}")

        image_count = sum(1 for path in cohort_root.rglob("*.tif") if path.is_file())
        if image_count != source.expected_count:
            raise ValueError(
                f"Expected {source.expected_count} TIFF files in {source.cohort}, "
                f"found {image_count}"
            )


class Kather100KNonNormAdapter(Kather100KAdapter):
    """Build and validate the non-color-normalized Kather100K training package."""

    color_normalized = False
    metadata = replace(
        Kather100KAdapter.metadata,
        identifier="kather100k-nonorm",
        valid_count=0,
        description="Non-color-normalized H&E colorectal tissue patches.",
    )
    sources = (
        DatasetSource(
            filename="NCT-CRC-HE-100K-NONORM.zip",
            url=("https://zenodo.org/api/records/1214456/files/NCT-CRC-HE-100K-NONORM.zip/content"),
            size=11_726_652_899,
            checksum="md5:035777cf327776a71a05c95da6d6325f",
            split="train",
            cohort="NCT-CRC-HE-100K-NONORM",
            expected_count=100_000,
        ),
    )


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
