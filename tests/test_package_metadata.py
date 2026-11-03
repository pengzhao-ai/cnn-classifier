from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = (
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "NOTICE",
    "THIRD_PARTY.md",
    "docs/release.md",
)


def _project() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_package_metadata_urls_and_entry_point():
    metadata = _project()
    assert metadata["scripts"]["cnncls"] == "cnn_classifier.cli:main"
    assert metadata["urls"]["Repository"] == "https://github.com/pengzhao-ai/cnn-classifier"
    assert metadata["urls"]["Documentation"] == "https://pengzhao-ai.github.io/cnn-classifier/"


def test_package_metadata_declares_supported_python_and_license():
    metadata = _project()
    assert metadata["requires-python"] == ">=3.11,<3.14"
    assert metadata["license"] == "Apache-2.0"
    assert metadata["dynamic"] == ["version"]
    assert "docs" in metadata["optional-dependencies"]
    assert "dev" in metadata["optional-dependencies"]


def test_release_files_exist_and_have_content():
    for name in RELEASE_FILES:
        path = REPO_ROOT / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_citation_and_changelog_are_parseable():
    citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["title"]
    assert citation["version"]
    assert citation["repository-code"] == "https://github.com/pengzhao-ai/cnn-classifier"

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.lstrip().startswith("#")
    assert "Unreleased" in changelog or "0.1.0" in changelog


def test_release_docs_contain_owner_checklist():
    text = (REPO_ROOT / "docs/release.md").read_text(encoding="utf-8").lower()
    for phrase in ("trusted publisher", "pages", "owner", "twine check"):
        assert phrase in text, phrase


def test_readme_uses_canonical_absolute_docs_url():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://pengzhao-ai.github.io/cnn-classifier/" in text
