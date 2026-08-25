from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import histodata.__main__ as cli_module
from histodata.__main__ import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.startswith("histodata 0.1.0")


def test_list_and_info() -> None:
    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert "kather100k" in listed.stdout
    detailed = runner.invoke(app, ["info", "kather100k"])
    assert detailed.exit_code == 0
    assert "10.5281/zenodo.1214456" in detailed.stdout
    assert "CC-BY-4.0" in detailed.stdout


def test_unknown_dataset() -> None:
    result = runner.invoke(app, ["info", "missing"])
    assert result.exit_code == 2
    assert "Available datasets: kather100k" in result.stderr


def test_download_requires_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HISTODATA_DATASETS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["download", "kather100k"])
    assert result.exit_code == 2
    assert "Pass --output ROOT" in result.stderr


def test_download_uses_environment_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    monkeypatch.setenv("HISTODATA_DATASETS_ROOT", str(root))
    received: list[Path] = []

    def fake_install(adapter, datasets_root: Path, **_kwargs) -> Path:
        received.append(datasets_root)
        package = datasets_root / adapter.metadata.identifier
        package.mkdir(parents=True)
        return package

    monkeypatch.setattr(cli_module, "install_dataset", fake_install)
    result = runner.invoke(app, ["download", "kather100k"])
    assert result.exit_code == 0
    assert received == [root]
    assert f"Installed: {root / 'kather100k'}" in result.stdout


def test_output_overrides_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HISTODATA_DATASETS_ROOT", str(tmp_path / "environment"))
    explicit = tmp_path / "explicit"
    received: list[Path] = []

    def fake_install(adapter, datasets_root: Path, **_kwargs) -> Path:
        received.append(datasets_root)
        return datasets_root / adapter.metadata.identifier

    monkeypatch.setattr(cli_module, "install_dataset", fake_install)
    result = runner.invoke(app, ["download", "kather100k", "--output", str(explicit)])
    assert result.exit_code == 0
    assert received == [explicit]
