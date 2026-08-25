"""Built-in dataset adapter registry."""

from __future__ import annotations

from histodata.adapters import Kather100KAdapter
from histodata.models import DatasetAdapter

_ADAPTERS: dict[str, DatasetAdapter] = {"kather100k": Kather100KAdapter()}


class UnknownDatasetError(KeyError):
    """Raised when a dataset name is not registered."""


def available_adapters() -> tuple[DatasetAdapter, ...]:
    return tuple(_ADAPTERS[name] for name in sorted(_ADAPTERS))


def get_adapter(identifier: str) -> DatasetAdapter:
    try:
        return _ADAPTERS[identifier]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise UnknownDatasetError(
            f"Unknown dataset '{identifier}'. Available datasets: {available}"
        ) from exc
