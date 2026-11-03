"""Command-line entry point for cnn-classifier."""

from __future__ import annotations

import platform

import typer

from cnn_classifier._version import __version__

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Config-driven PyTorch toolkit for CNN image classification.",
)


def _version_callback(value: bool) -> None:
    """Print the package version and exit when --version is supplied."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the package version and exit.",
    ),
) -> None:
    """Run the cnn-classifier command line."""
    _ = version


@app.command()
def doctor() -> None:
    """Print package and runtime information."""
    typer.echo(f"cnn-classifier {__version__}")
    typer.echo(f"python {platform.python_version()}")
    typer.echo(f"platform {platform.platform()}")


def main() -> None:
    """Console-script entry point."""
    app()
