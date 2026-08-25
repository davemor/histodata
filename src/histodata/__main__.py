"""Histodata command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from histodata import __version__
from histodata.config import ConfigurationError, resolve_datasets_root
from histodata.download import InstallationError, install_dataset
from histodata.models import DatasetAdapter
from histodata.registry import UnknownDatasetError, available_adapters, get_adapter
from histodata.validation import validate_package

app = typer.Typer(
    name="histodata",
    help="Download, standardize, and validate histopathology datasets.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"histodata {__version__}")
        raise typer.Exit()


@app.callback()
def cli(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the Histodata version and exit.",
        ),
    ] = False,
) -> None:
    """Manage standardized local copies of externally hosted datasets."""


@app.command("list")
def list_datasets() -> None:
    """List available built-in dataset adapters."""
    typer.echo("ID           KIND   VERSION  DESCRIPTION")
    for adapter in available_adapters():
        metadata = adapter.metadata
        typer.echo(
            f"{metadata.identifier:<12} {metadata.kind:<6} {metadata.version:<8} "
            f"{metadata.description}"
        )


@app.command()
def info(dataset: Annotated[str, typer.Argument(help="Registered dataset identifier.")]) -> None:
    """Show metadata and pinned sources for a dataset."""
    adapter = _adapter_or_exit(dataset)
    metadata = adapter.metadata
    total_size = sum(source.size for source in adapter.sources)
    typer.echo(f"ID:          {metadata.identifier}")
    typer.echo(f"Title:       {metadata.title}")
    typer.echo(f"Kind:        {metadata.kind}")
    typer.echo(f"Version:     {metadata.version}")
    typer.echo(f"DOI:         {metadata.doi}")
    typer.echo(f"Licence:     {metadata.license_id} ({metadata.license_url})")
    typer.echo(f"Download:    {_format_bytes(total_size)}")
    typer.echo("Sources:")
    for source in adapter.sources:
        typer.echo(
            f"  {source.filename}  split={source.split}  items={source.expected_count}  "
            f"size={_format_bytes(source.size)}  {source.checksum}"
        )


@app.command()
def download(
    dataset: Annotated[str, typer.Argument(help="Registered dataset identifier.")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=("Parent datasets directory. Overrides HISTODATA_DATASETS_ROOT and ./.env."),
        ),
    ] = None,
    keep_archives: Annotated[
        bool,
        typer.Option(
            "--keep-archives",
            help="Retain verified source ZIP archives inside the dataset package.",
        ),
    ] = False,
) -> None:
    """Download and install a dataset in the standardized Histodata format."""
    adapter = _adapter_or_exit(dataset)
    try:
        datasets_root = resolve_datasets_root(output)
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    typer.echo(f"Dataset: {adapter.metadata.identifier}")
    typer.echo(f"Root:    {datasets_root}")
    reporter = _CliProgress()
    try:
        package = install_dataset(
            adapter,
            datasets_root,
            keep_archives=keep_archives,
            progress=reporter,
        )
    except (InstallationError, OSError, ValueError) as exc:
        typer.echo(f"Installation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Installed: {package}")


@app.command("validate")
def validate_command(
    package: Annotated[Path, typer.Argument(help="Standardized dataset package directory.")],
    deep: Annotated[
        bool,
        typer.Option(
            "--deep",
            help="Open every indexed image and verify its integrity and dimensions.",
        ),
    ] = False,
) -> None:
    """Validate a standardized Histodata dataset package."""
    report = validate_package(package, deep=deep)
    if report.valid:
        mode = "deep" if deep else "fast"
        typer.echo(
            f"Valid Histodata package ({mode} validation, "
            f"{report.checked_files} files checked): {report.package}"
        )
        return
    typer.echo(f"Invalid Histodata package: {report.package}", err=True)
    for error in report.errors:
        typer.echo(f"  - {error}", err=True)
    raise typer.Exit(1)


def _adapter_or_exit(identifier: str) -> DatasetAdapter:
    try:
        return get_adapter(identifier)
    except UnknownDatasetError as exc:
        typer.echo(str(exc).strip("'"), err=True)
        raise typer.Exit(2) from exc


class _CliProgress:
    def __init__(self) -> None:
        self._last_percent: dict[str, int] = {}

    def __call__(self, filename: str, downloaded: int, total: int) -> None:
        percent = min(100, int(downloaded * 100 / total)) if total else 0
        previous = self._last_percent.get(filename, -10)
        if percent == 100 or percent >= previous + 10:
            typer.echo(
                f"Downloading {filename}: {percent:>3}% "
                f"({_format_bytes(downloaded)} / {_format_bytes(total)})"
            )
            self._last_percent[filename] = percent


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"  # pragma: no cover


if __name__ == "__main__":
    app()
