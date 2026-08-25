from __future__ import annotations

import gzip
import hashlib
import stat
import zipfile
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest
from conftest import MiniDataset

from histodata.download import (
    InstallationError,
    _install_file_source,
    _obtain_archive,
    _validate_zip_member,
    install_dataset,
)
from histodata.models import DatasetSource
from histodata.validation import validate_package


def test_installs_standard_package_and_deletes_archives(
    tmp_path: Path, mini_dataset: MiniDataset, archive_transport: httpx.MockTransport
) -> None:
    progress: list[tuple[str, int, int]] = []
    package = install_dataset(
        mini_dataset.adapter,
        tmp_path,
        transport=archive_transport,
        retry_delay=0,
        progress=lambda name, current, total: progress.append((name, current, total)),
    )

    assert package == tmp_path / "mini-kather"
    assert (package / "manifest.json").is_file()
    assert (package / "index.parquet").is_file()
    assert not (package / "archives").exists()
    assert not (package / ".histodata").exists()
    table = pq.read_table(package / "index.parquet")
    assert table.num_rows == 18
    assert table["id"].to_pylist()[0] == "train/ADI/ADI-train"
    assert progress
    assert validate_package(package, adapter=mini_dataset.adapter).valid


def test_keep_archives_and_idempotent_rerun(
    tmp_path: Path, mini_dataset: MiniDataset, archive_transport: httpx.MockTransport
) -> None:
    package = install_dataset(
        mini_dataset.adapter,
        tmp_path,
        keep_archives=True,
        transport=archive_transport,
        retry_delay=0,
    )
    assert sorted(path.name for path in (package / "archives").iterdir()) == [
        "train.zip",
        "validation.zip",
    ]

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Idempotent install made a request: {request.url}")

    assert (
        install_dataset(
            mini_dataset.adapter,
            tmp_path,
            transport=httpx.MockTransport(unexpected_request),
            retry_delay=0,
        )
        == package
    )


def test_refuses_unrelated_destination(tmp_path: Path, mini_dataset: MiniDataset) -> None:
    package = tmp_path / mini_dataset.adapter.metadata.identifier
    package.mkdir()
    (package / "unrelated.txt").write_text("user data")
    with pytest.raises(InstallationError, match="not a resumable"):
        install_dataset(mini_dataset.adapter, tmp_path, retries=1, retry_delay=0)


def test_resumes_partial_download(tmp_path: Path, mini_dataset: MiniDataset) -> None:
    source = mini_dataset.adapter.sources[0]
    content = mini_dataset.archives[source.url]
    archive = tmp_path / source.filename
    partial = archive.with_suffix(".zip.part")
    partial.write_bytes(content[:100])
    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers.get("Range")
        seen_range.append(range_header)
        return httpx.Response(206, content=content[100:])

    result = _obtain_archive(
        source,
        archive,
        transport=httpx.MockTransport(handler),
        progress=None,
        retries=1,
        retry_delay=0,
    )
    assert result.read_bytes() == content
    assert seen_range == ["bytes=100-"]


def test_restarts_when_server_ignores_range(tmp_path: Path, mini_dataset: MiniDataset) -> None:
    source = mini_dataset.adapter.sources[0]
    content = mini_dataset.archives[source.url]
    archive = tmp_path / source.filename
    archive.with_suffix(".zip.part").write_bytes(content[:100])

    result = _obtain_archive(
        source,
        archive,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=content)),
        progress=None,
        retries=1,
        retry_delay=0,
    )
    assert result.read_bytes() == content


def test_retries_transient_http_failure(tmp_path: Path, mini_dataset: MiniDataset) -> None:
    source = mini_dataset.adapter.sources[0]
    content = mini_dataset.archives[source.url]
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=content)

    result = _obtain_archive(
        source,
        tmp_path / source.filename,
        transport=httpx.MockTransport(handler),
        progress=None,
        retries=2,
        retry_delay=0,
    )
    assert result.read_bytes() == content
    assert calls == 2


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    content = b"not-the-expected-content"
    source = DatasetSource(
        filename="bad.zip",
        url="https://example.test/bad.zip",
        size=len(content),
        checksum=f"md5:{hashlib.md5(b'different').hexdigest()}",
        split="train",
        cohort="BAD",
        expected_count=1,
    )
    with pytest.raises(InstallationError, match="checksum mismatch"):
        _obtain_archive(
            source,
            tmp_path / source.filename,
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=content)),
            progress=None,
            retries=1,
            retry_delay=0,
        )


@pytest.mark.parametrize("name", ["../escape.tif", "/absolute.tif", "C:/drive.tif"])
def test_rejects_unsafe_zip_paths(name: str) -> None:
    with pytest.raises(InstallationError, match="Unsafe ZIP member"):
        _validate_zip_member(zipfile.ZipInfo(name))


def test_rejects_zip_symlink() -> None:
    member = zipfile.ZipInfo("cohort/link.tif")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(InstallationError, match="Unsafe ZIP member"):
        _validate_zip_member(member)


def test_installs_and_decompresses_gzip_source(tmp_path: Path) -> None:
    extracted = b"native-container-content"
    content = gzip.compress(extracted)
    source = DatasetSource(
        filename="container.h5.gz",
        url="https://example.test/container.h5.gz",
        size=len(content),
        checksum=f"md5:{hashlib.md5(content).hexdigest()}",
        split="train",
        cohort="container",
        expected_count=1,
        kind="gzip",
        target="container/container.h5",
    )
    _install_file_source(
        source,
        tmp_path,
        keep_archives=False,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=content)),
        progress=None,
        retries=1,
        retry_delay=0,
    )
    assert (tmp_path / "data/container/container.h5").read_bytes() == extracted
    assert not (tmp_path / "archives").exists()
