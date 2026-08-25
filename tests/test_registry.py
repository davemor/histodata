from __future__ import annotations

import pytest

from histodata.adapters.kather100k import LABELS
from histodata.registry import UnknownDatasetError, available_adapters, get_adapter


def test_registry_contains_kather100k() -> None:
    adapter = get_adapter("kather100k")
    assert adapter.metadata.kind == "patch"
    assert adapter.labels == LABELS
    assert [item.metadata.identifier for item in available_adapters()] == ["kather100k"]


def test_kather_sources_are_pinned() -> None:
    sources = get_adapter("kather100k").sources
    assert [(source.filename, source.size, source.checksum) for source in sources] == [
        (
            "NCT-CRC-HE-100K.zip",
            11_690_284_003,
            "md5:6fd702d11df6292bc054397ae038a464",
        ),
        (
            "CRC-VAL-HE-7K.zip",
            800_276_929,
            "md5:2fd1651b4f94ebd818ebf90ad2b6ce06",
        ),
    ]


def test_unknown_dataset_lists_available_names() -> None:
    with pytest.raises(UnknownDatasetError, match="kather100k"):
        get_adapter("missing")
