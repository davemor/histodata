"""Configuration resolution for the Histodata CLI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

DATASETS_ROOT_ENV = "HISTODATA_DATASETS_ROOT"


class ConfigurationError(ValueError):
    """Raised when the datasets root cannot be resolved."""


def resolve_datasets_root(
    output: Path | None,
    *,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the parent directory used to store canonical datasets.

    Precedence is an explicit CLI value, the process environment, then ``.env``
    in the current working directory. Relative values are interpreted from that
    working directory.
    """
    working_directory = (cwd or Path.cwd()).resolve()
    environment = os.environ if environ is None else environ

    if output is not None:
        return _normalise_path(output, working_directory)

    if DATASETS_ROOT_ENV in environment:
        value = environment[DATASETS_ROOT_ENV]
        if not value.strip():
            raise ConfigurationError(f"{DATASETS_ROOT_ENV} is set but empty")
        return _normalise_path(Path(value), working_directory)

    dotenv_path = working_directory / ".env"
    if dotenv_path.is_file():
        value = dotenv_values(dotenv_path).get(DATASETS_ROOT_ENV)
        if value is not None:
            if not value.strip():
                raise ConfigurationError(
                    f"{DATASETS_ROOT_ENV} is present in {dotenv_path} but empty"
                )
            return _normalise_path(Path(value), working_directory)

    raise ConfigurationError(
        "No datasets root configured. Pass --output ROOT or set "
        f"{DATASETS_ROOT_ENV} in the environment or ./.env."
    )


def _normalise_path(path: Path, cwd: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = cwd / expanded
    return expanded.resolve()
