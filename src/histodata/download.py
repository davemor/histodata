"""Resumable download and safe extraction of registered datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import httpx

from histodata.models import DatasetAdapter, DatasetSource
from histodata.package import MANIFEST_FILENAME, write_package
from histodata.validation import validate_package

ProgressCallback = Callable[[str, int, int], None]
CHUNK_SIZE = 1024 * 1024
DISK_HEADROOM_BYTES = 512 * 1024 * 1024


class InstallationError(RuntimeError):
    """Raised when a dataset cannot be safely installed."""


def install_dataset(
    adapter: DatasetAdapter,
    datasets_root: Path,
    *,
    keep_archives: bool = False,
    transport: httpx.BaseTransport | None = None,
    progress: ProgressCallback | None = None,
    retries: int = 3,
    retry_delay: float = 1.0,
) -> Path:
    """Install one registered dataset under its canonical identifier."""
    package_dir = datasets_root / adapter.metadata.identifier
    manifest_path = package_dir / MANIFEST_FILENAME

    if manifest_path.is_file():
        report = validate_package(package_dir, deep=False, adapter=adapter)
        if report.valid:
            return package_dir
        raise InstallationError(
            f"Existing Histodata package is invalid: {package_dir}\n" + "\n".join(report.errors)
        )

    state_dir = package_dir / ".histodata"
    state_path = state_dir / "state.json"
    if package_dir.exists() and any(package_dir.iterdir()):
        if not state_path.is_file():
            raise InstallationError(
                f"Destination exists and is not a resumable Histodata installation: {package_dir}"
            )
        state = _read_state(state_path)
        if state.get("dataset_id") != adapter.metadata.identifier:
            raise InstallationError(f"Installation state belongs to another dataset: {state_path}")

    package_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_state(state_path, adapter.metadata.identifier)

    for source in adapter.sources:
        _install_source(
            adapter,
            source,
            package_dir,
            keep_archives=keep_archives,
            transport=transport,
            progress=progress,
            retries=retries,
            retry_delay=retry_delay,
        )

    write_package(adapter, package_dir, keep_archives=keep_archives)
    shutil.rmtree(state_dir)
    return package_dir


def _install_source(
    adapter: DatasetAdapter,
    source: DatasetSource,
    package_dir: Path,
    *,
    keep_archives: bool,
    transport: httpx.BaseTransport | None,
    progress: ProgressCallback | None,
    retries: int,
    retry_delay: float,
) -> None:
    final_cohort = package_dir / "data" / source.cohort
    cohort_complete = _cohort_looks_complete(final_cohort, source, adapter.labels)

    downloads_dir = package_dir / ".histodata" / "downloads"
    retained_dir = package_dir / "archives"
    retained_archive = retained_dir / source.filename
    internal_archive = downloads_dir / source.filename

    archive: Path | None = None
    if retained_archive.is_file() and _verify_file(retained_archive, source):
        archive = retained_archive
    elif not cohort_complete or keep_archives:
        downloads_dir.mkdir(parents=True, exist_ok=True)
        archive = _obtain_archive(
            source,
            internal_archive,
            transport=transport,
            progress=progress,
            retries=retries,
            retry_delay=retry_delay,
        )

    if not cohort_complete:
        if archive is None:
            raise InstallationError(f"No verified archive available for {source.cohort}")
        _extract_source(archive, source, package_dir, adapter.labels)

    if keep_archives:
        if archive is None:
            raise InstallationError(f"No verified archive available to retain: {source.filename}")
        retained_dir.mkdir(parents=True, exist_ok=True)
        if archive != retained_archive:
            os.replace(archive, retained_archive)
    else:
        internal_archive.unlink(missing_ok=True)
        retained_archive.unlink(missing_ok=True)


def _obtain_archive(
    source: DatasetSource,
    archive_path: Path,
    *,
    transport: httpx.BaseTransport | None,
    progress: ProgressCallback | None,
    retries: int,
    retry_delay: float,
) -> Path:
    partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
    if _verify_file(archive_path, source):
        return archive_path
    archive_path.unlink(missing_ok=True)

    if _verify_file(partial_path, source):
        os.replace(partial_path, archive_path)
        return archive_path

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            _stream_download(
                source,
                partial_path,
                transport=transport,
                progress=progress,
                retries=retries,
                retry_delay=retry_delay,
            )
            if not _verify_file(partial_path, source):
                raise InstallationError(
                    f"Size or {source.checksum_algorithm} checksum mismatch for {source.filename}"
                )
            os.replace(partial_path, archive_path)
            return archive_path
        except (InstallationError, httpx.HTTPError, OSError) as exc:
            last_error = exc
            if isinstance(exc, InstallationError):
                partial_path.unlink(missing_ok=True)
            if attempt + 1 < retries and retry_delay:
                time.sleep(retry_delay * (2**attempt))

    raise InstallationError(f"Failed to download {source.filename}: {last_error}") from last_error


def _stream_download(
    source: DatasetSource,
    partial_path: Path,
    *,
    transport: httpx.BaseTransport | None,
    progress: ProgressCallback | None,
    retries: int,
    retry_delay: float,
) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    if partial_path.exists() and partial_path.stat().st_size > source.size:
        partial_path.unlink()

    timeout = httpx.Timeout(connect=30.0, read=60.0, write=60.0, pool=30.0)
    last_error: Exception | None = None

    with httpx.Client(follow_redirects=True, timeout=timeout, transport=transport) as client:
        for attempt in range(retries):
            offset = partial_path.stat().st_size if partial_path.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with client.stream("GET", source.url, headers=headers) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == httpx.codes.PARTIAL_CONTENT
                    mode = "ab" if append else "wb"
                    downloaded = offset if append else 0
                    with partial_path.open(mode) as file:
                        for chunk in response.iter_bytes(CHUNK_SIZE):
                            file.write(chunk)
                            downloaded += len(chunk)
                            if progress is not None:
                                progress(source.filename, downloaded, source.size)
                if partial_path.stat().st_size == source.size:
                    return
                raise InstallationError(
                    f"Expected {source.size} bytes for {source.filename}, "
                    f"received {partial_path.stat().st_size}"
                )
            except (httpx.HTTPError, OSError, InstallationError) as exc:
                last_error = exc
                if attempt + 1 < retries and retry_delay:
                    time.sleep(retry_delay * (2**attempt))

    raise InstallationError(f"Download failed for {source.filename}: {last_error}") from last_error


def _verify_file(path: Path, source: DatasetSource) -> bool:
    if not path.is_file() or path.stat().st_size != source.size:
        return False
    digest = hashlib.new(source.checksum_algorithm)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower() == source.checksum_value.lower()


def _extract_source(
    archive_path: Path,
    source: DatasetSource,
    package_dir: Path,
    labels: dict[str, int],
) -> None:
    staging_root = package_dir / ".histodata" / "extracted" / f"{source.cohort}.partial"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            _validate_zip_member(member)

        required_bytes = sum(member.file_size for member in members if not member.is_dir())
        available_bytes = shutil.disk_usage(package_dir).free
        if available_bytes < required_bytes + DISK_HEADROOM_BYTES:
            raise InstallationError(
                f"Insufficient disk space to extract {source.filename}: need at least "
                f"{required_bytes + DISK_HEADROOM_BYTES} free bytes, found {available_bytes}"
            )

        for member in members:
            relative = PurePosixPath(member.filename.replace("\\", "/"))
            destination = staging_root.joinpath(*relative.parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_file, destination.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file, length=CHUNK_SIZE)

    top_level = {path.name for path in staging_root.iterdir()}
    if top_level != {source.cohort}:
        raise InstallationError(
            f"Unexpected archive layout in {source.filename}: expected only "
            f"{source.cohort}, found {sorted(top_level)}"
        )

    extracted_cohort = staging_root / source.cohort
    if not _cohort_looks_complete(extracted_cohort, source, labels):
        raise InstallationError(
            f"Extracted cohort {source.cohort} does not contain the expected classes and files"
        )

    final_cohort = package_dir / "data" / source.cohort
    final_cohort.parent.mkdir(parents=True, exist_ok=True)
    if final_cohort.exists():
        shutil.rmtree(final_cohort)
    os.replace(extracted_cohort, final_cohort)
    shutil.rmtree(staging_root)


def _validate_zip_member(member: zipfile.ZipInfo) -> None:
    raw_name = member.filename.replace("\\", "/")
    path = PurePosixPath(raw_name)
    file_type = (member.external_attr >> 16) & 0o170000
    if (
        not raw_name
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
        or file_type == stat.S_IFLNK
    ):
        raise InstallationError(f"Unsafe ZIP member rejected: {member.filename}")


def _cohort_looks_complete(
    cohort_root: Path, source: DatasetSource, labels: dict[str, int]
) -> bool:
    if not cohort_root.is_dir():
        return False
    directories = {path.name for path in cohort_root.iterdir() if path.is_dir()}
    if directories != set(labels):
        return False
    files = [path for path in cohort_root.rglob("*") if path.is_file()]
    return len(files) == source.expected_count and all(
        path.suffix.lower() == ".tif" for path in files
    )


def _write_state(path: Path, dataset_id: str) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump({"dataset_id": dataset_id}, file)


def _read_state(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise InstallationError(f"Cannot read installation state: {path}") from exc
    if not isinstance(value, dict):
        raise InstallationError(f"Invalid installation state: {path}")
    return value
