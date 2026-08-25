"""Internal dataset adapter models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyarrow as pa


@dataclass(frozen=True)
class DatasetSource:
    """A pinned externally hosted archive belonging to a dataset."""

    filename: str
    url: str
    size: int
    checksum: str
    split: str
    cohort: str
    expected_count: int

    @property
    def checksum_algorithm(self) -> str:
        return self.checksum.split(":", 1)[0]

    @property
    def checksum_value(self) -> str:
        return self.checksum.split(":", 1)[1]


@dataclass(frozen=True)
class DatasetMetadata:
    """Descriptive metadata shown by the registry and stored in manifests."""

    identifier: str
    title: str
    version: str
    kind: str
    description: str
    homepage: str
    doi: str
    license_id: str
    license_url: str
    creators: tuple[dict[str, str], ...]
    citation: str


class DatasetAdapter(Protocol):
    """Internal contract implemented by built-in dataset adapters."""

    metadata: DatasetMetadata
    sources: tuple[DatasetSource, ...]

    @property
    def labels(self) -> dict[str, int]: ...

    def build_index(self, package_dir: Path) -> pa.Table: ...

    def manifest_dataset_metadata(self) -> dict[str, object]: ...

    def validate_table(self, table: pa.Table) -> list[str]: ...
