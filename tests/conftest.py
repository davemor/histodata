from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, replace

import httpx
import pytest
from PIL import Image

from histodata.adapters.kather100k import LABELS, Kather100KAdapter
from histodata.models import DatasetSource


@dataclass(frozen=True)
class MiniDataset:
    adapter: Kather100KAdapter
    archives: dict[str, bytes]


class MiniKatherAdapter(Kather100KAdapter):
    def __init__(self, sources: tuple[DatasetSource, ...]) -> None:
        self.metadata = replace(Kather100KAdapter.metadata, identifier="mini-kather")
        self.sources = sources


@pytest.fixture
def mini_dataset() -> MiniDataset:
    specifications = (
        ("train.zip", "TRAIN", "train"),
        ("validation.zip", "VALIDATION", "validation"),
    )
    archives: dict[str, bytes] = {}
    sources: list[DatasetSource] = []
    for filename, cohort, split in specifications:
        content = _make_archive(cohort)
        url = f"https://example.test/{filename}"
        archives[url] = content
        sources.append(
            DatasetSource(
                filename=filename,
                url=url,
                size=len(content),
                checksum=f"md5:{hashlib.md5(content).hexdigest()}",
                split=split,
                cohort=cohort,
                expected_count=len(LABELS),
            )
        )
    return MiniDataset(MiniKatherAdapter(tuple(sources)), archives)


@pytest.fixture
def archive_transport(mini_dataset: MiniDataset) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        content = mini_dataset.archives[str(request.url)]
        range_header = request.headers.get("Range")
        if range_header:
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            return httpx.Response(
                206,
                content=content[offset:],
                headers={"Content-Range": f"bytes {offset}-{len(content) - 1}/{len(content)}"},
            )
        return httpx.Response(200, content=content)

    return httpx.MockTransport(handler)


def _make_archive(cohort: str) -> bytes:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, index in LABELS.items():
            image_bytes = io.BytesIO()
            Image.new("RGB", (224, 224), color=(index * 20, 20, 100)).save(
                image_bytes, format="TIFF"
            )
            archive.writestr(
                f"{cohort}/{label}/{label}-{cohort.lower()}.tif", image_bytes.getvalue()
            )
    return archive_bytes.getvalue()
