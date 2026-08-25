from __future__ import annotations

from typing import cast

import pytest

from histodata.adapters.kather100k import LABELS
from histodata.registry import UnknownDatasetError, available_adapters, get_adapter


def test_registry_contains_kather100k() -> None:
    adapter = get_adapter("kather100k")
    assert adapter.metadata.kind == "patch"
    assert adapter.labels == LABELS
    assert [item.metadata.identifier for item in available_adapters()] == [
        "kather100k",
        "kather100k-nonorm",
        "pcam",
        "spider-breast",
        "spider-colorectal",
        "spider-skin",
        "spider-thorax",
    ]


def test_pcam_and_spider_sources_are_pinned() -> None:
    pcam = get_adapter("pcam")
    assert len(pcam.sources) == 9
    assert (
        sum(source.expected_count for source in pcam.sources if "_x." in source.filename) == 327_680
    )
    spider = get_adapter("spider-skin")
    assert spider.sources[0].repository == "histai/SPIDER-skin"
    assert spider.sources[0].revision == "7ab1badb5a32a7b6de809a37561e42968afd4b51"
    assert spider.sources[0].part_count == 45


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


def test_kather_nonorm_source_is_pinned() -> None:
    adapter = get_adapter("kather100k-nonorm")
    assert adapter.metadata.kind == "patch"
    assert adapter.labels == LABELS
    metadata = adapter.manifest_dataset_metadata()
    patch_metadata = cast(dict[str, object], metadata["patch"])
    assert patch_metadata["color_normalized"] is False
    assert [(source.filename, source.size, source.checksum) for source in adapter.sources] == [
        (
            "NCT-CRC-HE-100K-NONORM.zip",
            11_726_652_899,
            "md5:035777cf327776a71a05c95da6d6325f",
        )
    ]


def test_unknown_dataset_lists_available_names() -> None:
    with pytest.raises(UnknownDatasetError, match="kather100k"):
        get_adapter("missing")
