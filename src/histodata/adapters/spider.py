"""Adapters for the four SPIDER organ datasets."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

import pyarrow as pa
from PIL import Image

from histodata.models import DatasetMetadata, DatasetSource

PAPER = "https://arxiv.org/abs/2503.02876"
CREATORS = (
    {"name": "Nechaev, Dmitry", "affiliation": "HistAI"},
    {"name": "Pchelnikov, Alexey", "affiliation": "HistAI"},
    {"name": "Ivanova, Ekaterina", "affiliation": "HistAI"},
)
CITATION = (
    "Nechaev, D., Pchelnikov, A., & Ivanova, E. (2025). SPIDER: A Comprehensive "
    "Multi-Organ Supervised Pathology Dataset and Baseline Models. arXiv:2503.02876."
)

INDEX_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("media_type", pa.string(), nullable=False),
        pa.field("central_path", pa.string(), nullable=False),
        pa.field("context_paths", pa.list_(pa.string()), nullable=False),
        pa.field("organ", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("label_index", pa.int16(), nullable=False),
        pa.field("slide_id", pa.string(), nullable=False),
        pa.field("width", pa.int16(), nullable=False),
        pa.field("height", pa.int16(), nullable=False),
        pa.field("mpp_x", pa.float32(), nullable=False),
        pa.field("mpp_y", pa.float32(), nullable=False),
        pa.field("stain", pa.string(), nullable=False),
        pa.field("byte_size", pa.int64(), nullable=False),
    ]
)


class SpiderAdapter:
    """Base adapter for publisher-native SPIDER metadata and context patches."""

    organ: ClassVar[str]
    expected_count: ClassVar[int]
    class_counts: ClassVar[dict[str, int]]
    revision: ClassVar[str]
    archive_size: ClassVar[int]
    part_count: ClassVar[int]

    def __init__(self) -> None:
        identifier = f"spider-{self.organ}"
        repository = f"histai/SPIDER-{self.organ}"
        self.metadata = DatasetMetadata(
            identifier=identifier,
            title=f"SPIDER-{self.organ.capitalize()}",
            version=self.revision[:12],
            kind="patch",
            description=(
                f"SPIDER {self.organ} dataset: expert-labelled 224x224 central H&E patches "
                "with 24 surrounding context patches at 20X."
            ),
            homepage=f"https://huggingface.co/datasets/{repository}",
            doi="10.48550/arXiv.2503.02876",
            license_id="CC-BY-NC-4.0",
            license_url="https://creativecommons.org/licenses/by-nc/4.0/",
            creators=CREATORS,
            citation=CITATION,
        )
        self.sources = (
            DatasetSource(
                filename=f"spider-{self.organ}.tar.*",
                url=f"https://huggingface.co/datasets/{repository}",
                size=self.archive_size,
                checksum=f"git:{self.revision}",
                split="train,test",
                cohort=identifier,
                expected_count=self.expected_count,
                kind="huggingface-split-tar",
                target=identifier,
                repository=repository,
                revision=self.revision,
                part_count=self.part_count,
            ),
        )

    @property
    def labels(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.class_counts)}

    def build_index(self, package_dir: Path) -> pa.Table:
        cohort_root = package_dir / "data" / f"spider-{self.organ}"
        metadata_path = cohort_root / "metadata.json"
        images_root = cohort_root / "images"
        if not metadata_path.is_file() or not images_root.is_dir():
            raise ValueError(f"Expected SPIDER images/ and metadata.json under {cohort_root}")
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Cannot read SPIDER metadata: {metadata_path}") from exc

        rows: list[dict[str, object]] = []
        for record, inherited_split in _iter_records(value):
            split = _split(record, inherited_split)
            label = _label(record, self.labels)
            central_raw, context_raw = _paths(record)
            central = _package_image_path(
                central_raw, package_dir=package_dir, images_root=images_root
            )
            contexts = [
                _package_image_path(path, package_dir=package_dir, images_root=images_root)
                for path in context_raw
            ]
            if len(contexts) != 24:
                raise ValueError(
                    f"SPIDER item {central_raw!r} has {len(contexts)} context patches; expected 24"
                )
            central_file = package_dir.joinpath(*PurePosixPath(central).parts)
            if not central_file.is_file():
                raise ValueError(f"SPIDER central patch does not exist: {central}")
            slide_id = _string_field(record, "slide_id", "slide", "wsi", "wsi_id")
            if slide_id is None:
                slide_id = central_file.parent.name
            central_key = PurePosixPath(central).relative_to(
                PurePosixPath("data") / f"spider-{self.organ}" / "images"
            )
            rows.append(
                {
                    "id": f"{split}/{central_key.with_suffix('').as_posix()}",
                    "path": central,
                    "split": split,
                    "media_type": _media_type(central_file.suffix),
                    "central_path": central,
                    "context_paths": contexts,
                    "organ": self.organ,
                    "label": label,
                    "label_index": self.labels[label],
                    "slide_id": slide_id,
                    "width": 224,
                    "height": 224,
                    "mpp_x": 0.5,
                    "mpp_y": 0.5,
                    "stain": "H&E",
                    "byte_size": central_file.stat().st_size,
                }
            )

        if len(rows) != self.expected_count:
            raise ValueError(
                f"Expected {self.expected_count} SPIDER-{self.organ} metadata items, "
                f"found {len(rows)}"
            )
        rows.sort(key=lambda row: str(row["id"]))
        return pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)

    def manifest_dataset_metadata(self) -> dict[str, object]:
        return {
            "organ": self.organ,
            "patch": {
                "width": 224,
                "height": 224,
                "mpp_x": 0.5,
                "mpp_y": 0.5,
                "magnification": "20X",
                "stain": "H&E",
                "context_grid": {"width": 5, "height": 5, "central_index": 12},
            },
            "labels": [
                {"name": label, "index": index, "expected_count": count}
                for index, (label, count) in enumerate(self.class_counts.items())
            ],
            "access": {
                "gated": True,
                "token_environment_variable": "HF_TOKEN",
                "conditions_url": self.metadata.homepage,
            },
        }

    def validate_table(self, table: pa.Table) -> list[str]:
        errors: list[str] = []
        if table.schema != INDEX_SCHEMA:
            return [
                f"SPIDER-{self.organ} index schema differs from the required schema: {INDEX_SCHEMA}"
            ]
        rows = table.to_pylist()
        if len(rows) != self.expected_count:
            errors.append(
                f"Expected {self.expected_count} SPIDER-{self.organ} rows, found {len(rows)}"
            )
        counts: dict[str, int] = {}
        for row in rows:
            label = str(row["label"])
            counts[label] = counts.get(label, 0) + 1
            if label not in self.labels:
                errors.append(f"Unknown SPIDER-{self.organ} label: {label}")
            elif row["label_index"] != self.labels[label]:
                errors.append(f"Incorrect SPIDER label_index for {label}")
            if row["split"] not in {"train", "test"}:
                errors.append(f"Unexpected SPIDER split: {row['split']}")
            if len(row["context_paths"]) != 24:
                errors.append(f"SPIDER item {row['id']} does not have 24 context paths")
        for label, expected in self.class_counts.items():
            if counts.get(label, 0) != expected:
                errors.append(
                    f"Expected {expected} SPIDER-{self.organ} rows for {label}, "
                    f"found {counts.get(label, 0)}"
                )
        return list(dict.fromkeys(errors))

    def deep_validate(self, package_dir: Path, table: pa.Table) -> list[str]:
        errors: list[str] = []
        checked: set[str] = set()
        for paths in table["context_paths"].to_pylist():
            for raw_path in paths:
                if raw_path in checked:
                    continue
                checked.add(raw_path)
                path = package_dir.joinpath(*PurePosixPath(raw_path).parts)
                try:
                    with Image.open(path) as image:
                        size = image.size
                        image.verify()
                    if size != (224, 224):
                        errors.append(
                            f"Expected 224x224 SPIDER context patch at {raw_path}, found {size}"
                        )
                except (OSError, ValueError) as exc:
                    errors.append(f"Cannot verify SPIDER context patch {raw_path}: {exc}")
                if len(errors) >= 100:
                    return errors
        return errors


class SpiderSkinAdapter(SpiderAdapter):
    organ = "skin"
    expected_count = 159_854
    revision = "7ab1badb5a32a7b6de809a37561e42968afd4b51"
    archive_size = 241_591_910_400
    part_count = 45
    class_counts: ClassVar[dict[str, int]] = {
        "Actinic keratosis": 4_936,
        "Apocrine glands": 6_739,
        "Basal cell carcinoma": 6_446,
        "Carcinoma in situ": 5_478,
        "Collagen": 6_262,
        "Epidermis": 7_449,
        "Fat": 6_525,
        "Follicle": 8_343,
        "Inflammation": 5_856,
        "Invasive melanoma": 9_101,
        "Kaposi's sarcoma": 4_778,
        "Keratin": 6_418,
        "Melanoma in situ": 4_545,
        "Mercel cell carcinoma": 5_968,
        "Muscle": 6_051,
        "Necrosis": 6_842,
        "Nerves": 4_735,
        "Nevus": 8_937,
        "Sebaceous gland": 6_639,
        "Seborrheic keratosis": 10_311,
        "Solar elastosis": 7_613,
        "Squamous cell carcinoma": 6_051,
        "Vessels": 7_673,
        "Wart": 6_158,
    }


class SpiderColorectalAdapter(SpiderAdapter):
    organ = "colorectal"
    expected_count = 77_182
    revision = "a180079e24b53bb0900038eee855c7f50b768edc"
    archive_size = 88_802_785_280
    part_count = 17
    class_counts: ClassVar[dict[str, int]] = {
        "Adenocarcinoma high grade": 6_299,
        "Adenocarcinoma low grade": 6_066,
        "Adenoma high grade": 5_493,
        "Adenoma low grade": 5_693,
        "Fat": 6_081,
        "Hyperplastic polyp": 5_893,
        "Inflammation": 5_523,
        "Mucus": 5_711,
        "Muscle": 5_866,
        "Necrosis": 5_481,
        "Sessile serrated lesion": 4_993,
        "Stroma healthy": 8_001,
        "Vessels": 6_082,
    }


class SpiderThoraxAdapter(SpiderAdapter):
    organ = "thorax"
    expected_count = 78_307
    revision = "0583a651de34e88293d6924a0081fc2a41d6c1ab"
    archive_size = 56_372_961_280
    part_count = 11
    class_counts: ClassVar[dict[str, int]] = {
        "Alveoli": 6_652,
        "Bronchial cartilage": 5_685,
        "Bronchial glands": 4_412,
        "Chronic inflammation + fibrosis": 6_070,
        "Detritus": 5_146,
        "Fibrosis": 6_494,
        "Hemorrhage": 5_247,
        "Lymph node": 6_088,
        "Pigment": 5_177,
        "Pleura": 4_560,
        "Tumor non-small cell": 6_445,
        "Tumor small cell": 5_061,
        "Tumor soft": 5_894,
        "Vessel": 5_376,
    }


class SpiderBreastAdapter(SpiderAdapter):
    organ = "breast"
    expected_count = 92_892
    revision = "cd697eb93ff4490eee721d60d24c670cf3db6b7e"
    archive_size = 86_138_654_720
    part_count = 17
    class_counts: ClassVar[dict[str, int]] = {
        "Adenosis": 2_899,
        "Benign phyllodes tumor": 4_526,
        "Ductal carcinoma in situ (high-grade)": 5_632,
        "Ductal carcinoma in situ (low-grade)": 5_017,
        "Fat": 6_286,
        "Fibroadenoma": 5_243,
        "Fibrocystic changes": 5_027,
        "Fibrosis": 6_260,
        "Invasive non-special type carcinoma": 6_142,
        "Lipogranuloma": 4_941,
        "Lobular invasive carcinoma": 5_102,
        "Malignant phyllodes tumor": 5_271,
        "Necrosis": 5_396,
        "Normal ducts": 4_891,
        "Normal lobules": 5_821,
        "Sclerosing adenosis": 3_423,
        "Typical ductal hyperplasia": 5_546,
        "Vessels": 5_469,
    }


def _iter_records(
    value: Any, inherited_split: str | None = None
) -> Iterable[tuple[dict[str, Any], str | None]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_records(item, inherited_split)
        return
    if not isinstance(value, dict):
        return
    if _record_has_paths(value):
        yield value, inherited_split
        return
    for key, nested in value.items():
        nested_split = inherited_split
        if key.lower() in {"train", "training", "test", "testing"}:
            nested_split = key
        yield from _iter_records(nested, nested_split)


def _record_has_paths(record: Mapping[str, Any]) -> bool:
    names = {
        "central_patch",
        "central_path",
        "center_patch",
        "center_path",
        "image",
        "image_path",
        "path",
        "context_patches",
        "context_paths",
        "surrounding_patches",
        "surrounding_paths",
        "images",
        "patches",
        "image_paths",
    }
    return bool(names.intersection(record))


def _split(record: Mapping[str, Any], inherited: str | None) -> str:
    value = _string_field(record, "split", "subset", "set") or inherited
    if value is None:
        raise ValueError("SPIDER metadata item has no train/test split")
    normalised = value.lower()
    if normalised in {"train", "training"}:
        return "train"
    if normalised in {"test", "testing"}:
        return "test"
    raise ValueError(f"Unexpected SPIDER split: {value}")


def _label(record: Mapping[str, Any], labels: Mapping[str, int]) -> str:
    value: Any = None
    for name in ("label", "class", "class_name", "target", "category"):
        if name in record:
            value = record[name]
            break
    if isinstance(value, int) and not isinstance(value, bool):
        by_index = {index: label for label, index in labels.items()}
        if value in by_index:
            return by_index[value]
    if not isinstance(value, str):
        raise ValueError("SPIDER metadata item has no usable class label")
    key = _normalise_label(value)
    matches = {_normalise_label(label): label for label in labels}
    try:
        return matches[key]
    except KeyError as exc:
        raise ValueError(f"Unknown SPIDER class label: {value}") from exc


def _normalise_label(value: str) -> str:
    value = value.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'").replace("\N{EN DASH}", "-")
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _paths(record: Mapping[str, Any]) -> tuple[str, list[str]]:
    central = _string_field(
        record,
        "central_patch",
        "central_path",
        "center_patch",
        "center_path",
        "image",
        "image_path",
        "path",
    )
    context_value: Any = None
    for name in (
        "context_patches",
        "context_paths",
        "surrounding_patches",
        "surrounding_paths",
        "images",
        "patches",
        "image_paths",
    ):
        if name in record:
            context_value = record[name]
            break
    paths = list(_flatten_paths(context_value))
    if central is None and len(paths) == 25:
        central = paths[12]
        paths = paths[:12] + paths[13:]
    elif central is not None:
        paths = [path for path in paths if path != central]
    if central is None:
        raise ValueError("SPIDER metadata item has no central patch path")
    return central, paths


def _flatten_paths(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_paths(item)


def _string_field(record: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _package_image_path(raw: str, *, package_dir: Path, images_root: Path) -> str:
    path = PurePosixPath(raw.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise ValueError(f"Unsafe SPIDER image path: {raw}")
    parts = path.parts
    if parts and parts[0] == "images":
        parts = parts[1:]
    candidate = images_root.joinpath(*parts)
    return candidate.relative_to(package_dir).as_posix()


def _media_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix.lower(), "application/octet-stream")
