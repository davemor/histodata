from __future__ import annotations

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image

import histodata.download as download_module
from histodata.adapters.spider import SpiderAdapter
from histodata.download import InstallationError, _validate_tar_member, install_dataset


class MiniSpiderAdapter(SpiderAdapter):
    organ = "mini"
    expected_count = 1
    revision = "a" * 40
    archive_size = 0
    part_count = 2
    class_counts: ClassVar[dict[str, int]] = {"Tumor": 1}


def _spider_tar() -> bytes:
    buffer = io.BytesIO()
    metadata = [
        {
            "split": "train",
            "label": "Tumor",
            "slide_id": "slide-1",
            "image_paths": [f"images/slide-1/patch-{index:02d}.png" for index in range(25)],
        }
    ]
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        metadata_bytes = json.dumps(metadata).encode()
        member = tarfile.TarInfo("metadata.json")
        member.size = len(metadata_bytes)
        archive.addfile(member, io.BytesIO(metadata_bytes))
        for index in range(25):
            image = io.BytesIO()
            Image.new("RGB", (224, 224), color=(index, 10, 20)).save(image, format="PNG")
            content = image.getvalue()
            member = tarfile.TarInfo(f"images/slide-1/patch-{index:02d}.png")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def test_installs_multipart_spider_and_indexes_context(tmp_path: Path, monkeypatch) -> None:
    content = _spider_tar()
    midpoint = len(content) // 2
    parts = [content[:midpoint], content[midpoint:]]
    adapter = MiniSpiderAdapter()
    adapter.sources = (replace(adapter.sources[0], size=len(content), part_count=2),)
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    def fake_download(*, filename: str, local_dir: Path, **_kwargs) -> str:
        index = int(filename.rsplit(".", 1)[1])
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(parts[index])
        return str(destination)

    monkeypatch.setattr(download_module, "hf_hub_download", fake_download)
    package = install_dataset(adapter, tmp_path)
    table = adapter.build_index(package)
    assert table.num_rows == 1
    assert table["path"][0].as_py().endswith("patch-12.png")
    assert len(table["context_paths"][0].as_py()) == 24
    assert adapter.deep_validate(package, table) == []
    assert not (package / "archives").exists()
    assert not (package / ".histodata").exists()


def test_rejects_unsafe_tar_members() -> None:
    with pytest.raises(InstallationError, match="Unsafe TAR member"):
        _validate_tar_member(tarfile.TarInfo("../escape.png"))
    symlink = tarfile.TarInfo("images/link.png")
    symlink.type = tarfile.SYMTYPE
    with pytest.raises(InstallationError, match="Unsafe TAR member"):
        _validate_tar_member(symlink)
