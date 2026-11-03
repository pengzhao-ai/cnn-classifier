from typer.testing import CliRunner

from cnn_classifier.cli import app

runner = CliRunner()


def test_cli_help_lists_doctor():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
