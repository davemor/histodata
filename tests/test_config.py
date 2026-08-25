from __future__ import annotations

from pathlib import Path

import pytest

from histodata.config import ConfigurationError, resolve_datasets_root


def test_explicit_output_has_highest_precedence(tmp_path: Path) -> None:
    result = resolve_datasets_root(
        Path("explicit"), cwd=tmp_path, environ={"HISTODATA_DATASETS_ROOT": "environment"}
    )
    assert result == tmp_path / "explicit"


def test_environment_precedes_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("HISTODATA_DATASETS_ROOT=dotenv\n")
    result = resolve_datasets_root(
        None, cwd=tmp_path, environ={"HISTODATA_DATASETS_ROOT": "environment"}
    )
    assert result == tmp_path / "environment"


def test_loads_dotenv_from_current_directory_only(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (tmp_path / ".env").write_text("HISTODATA_DATASETS_ROOT=parent\n")
    with pytest.raises(ConfigurationError, match="No datasets root"):
        resolve_datasets_root(None, cwd=child, environ={})

    (child / ".env").write_text("HISTODATA_DATASETS_ROOT=local-data\n")
    assert resolve_datasets_root(None, cwd=child, environ={}) == child / "local-data"


def test_expands_user_directory(tmp_path: Path) -> None:
    result = resolve_datasets_root(
        None, cwd=tmp_path, environ={"HISTODATA_DATASETS_ROOT": "~/histodata-test"}
    )
    assert result == Path("~/histodata-test").expanduser().resolve()


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_rejects_empty_configured_value(tmp_path: Path, source: str) -> None:
    environment: dict[str, str] = {}
    if source == "environment":
        environment["HISTODATA_DATASETS_ROOT"] = "  "
    else:
        (tmp_path / ".env").write_text("HISTODATA_DATASETS_ROOT=\n")
    with pytest.raises(ConfigurationError, match="empty"):
        resolve_datasets_root(None, cwd=tmp_path, environ=environment)


def test_missing_configuration_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Pass --output ROOT"):
        resolve_datasets_root(None, cwd=tmp_path, environ={})
