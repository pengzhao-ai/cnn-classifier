from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(config: dict) -> object:
    return config.get("on", config.get(True))


def test_ci_has_quality_test_docs_build_and_smoke_jobs():
    config = _workflow("ci.yml")
    jobs = config["jobs"]
    assert {"quality", "test", "docs", "build", "smoke"} <= set(jobs)

    matrix = jobs["test"]["strategy"]["matrix"]
    entries = matrix["include"]
    assert {entry["python-version"] for entry in entries} == {"3.11", "3.12", "3.13"}
    assert {entry["os"] for entry in entries} == {
        "ubuntu-latest",
        "macos-latest",
        "macos-15-intel",
    }
    assert len(entries) == 6


def test_ci_uses_non_editable_install_and_no_full_training():
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "uv sync --all-extras --no-editable" in text
    assert "--no-sync" in text
    assert "--cov-fail-under=90" in text
    for forbidden in (
        "run_scratch.py\n",
        "run_transfer.py",
        "run_ablations.py",
        "run_optics.py",
        "generate_dataset.py\n",
    ):
        assert forbidden not in text, forbidden


def test_docs_workflow_builds_strictly_and_deploys_pages():
    config = _workflow("docs.yml")
    triggers = _triggers(config)
    assert triggers
    text = (WORKFLOWS / "docs.yml").read_text(encoding="utf-8")
    assert "mkdocs build --strict" in text
    assert "actions/deploy-pages" in text
    assert "pages: write" in text
    assert "id-token: write" in text


def test_release_workflow_uses_tag_trigger_and_trusted_publishing():
    config = _workflow("release.yml")
    triggers = _triggers(config)
    assert "v*" in triggers["push"]["tags"]
    permissions = config["permissions"]
    assert permissions["contents"] == "write"
    assert permissions["id-token"] == "write"
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    for phrase in (
        "uv build",
        "twine check",
        "environment: pypi",
        "pypa/gh-action-pypi-publish",
        "gh release create",
        "GH_REPO: ${{ github.repository }}",
    ):
        assert phrase in text, phrase
    assert "password:" not in text.lower()
