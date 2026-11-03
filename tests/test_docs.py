from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

REQUIRED_PAGES = (
    "docs/index.md",
    "docs/quickstart.md",
    "docs/how_it_works.md",
    "docs/models.md",
    "docs/configuration.md",
    "docs/extending.md",
    "docs/cli.md",
    "docs/api.md",
    "docs/release.md",
    "docs/bring_your_own_data.md",
    "docs/techniques/augmentation.md",
    "docs/techniques/optimization.md",
    "docs/techniques/transfer.md",
    "docs/techniques/evaluation.md",
    "docs/techniques/diagnostics.md",
    "docs/case_studies/cifar10.md",
    "docs/case_studies/transfer.md",
    "docs/case_studies/optics_degradation.md",
    "docs/case_studies/ablations.md",
)

ARCHITECTURE_IMAGES = (
    "architecture_cifar10.png",
    "architecture_transfer.png",
    "architecture_ablations.png",
    "architecture_optics.png",
    "architecture_byo.png",
)


def test_required_pages_exist_and_have_content():
    for page in REQUIRED_PAGES:
        path = REPO_ROOT / page
        assert path.is_file(), page
        assert len(path.read_text(encoding="utf-8")) > 200, page


def _nav_paths(items):
    if isinstance(items, dict):
        for value in items.values():
            yield from _nav_paths(value)
    elif isinstance(items, list):
        for item in items:
            yield from _nav_paths(item)
    elif isinstance(items, str) and not items.startswith(("http://", "https://")):
        yield items


class _MkDocsLoader(yaml.SafeLoader):
    pass


_MkDocsLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    lambda loader, suffix, node: suffix,
)


def test_nav_covers_required_sections_and_pages():
    config = yaml.load(
        (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"),
        Loader=_MkDocsLoader,
    )
    nav_text = yaml.safe_dump(config["nav"])
    for label in (
        "Home",
        "Quickstart",
        "How it works",
        "Overview",
        "Data & configuration",
        "Models",
        "Techniques",
        "Extending",
        "Demos",
        "CIFAR-10 scratch",
        "Transfer benchmark",
        "Ablations",
        "Optical degradation",
        "Bring your own data",
        "Reference",
        "CLI",
        "API",
    ):
        assert label in nav_text, label
    for page in _nav_paths(config["nav"]):
        assert (DOCS / page).is_file(), page


def test_internal_docs_are_excluded_from_the_public_site():
    config = yaml.load(
        (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"),
        Loader=_MkDocsLoader,
    )
    excluded = config.get("exclude_docs", "")
    assert "superpowers/" in excluded
    assert "release.md" in excluded
    nav_pages = list(_nav_paths(config["nav"]))
    assert "release.md" not in nav_pages
    assert not any("superpowers/" in page for page in nav_pages)


def test_technique_pages_follow_required_structure():
    for name in ("augmentation", "optimization", "transfer", "evaluation", "diagnostics"):
        text = (DOCS / "techniques" / f"{name}.md").read_text(encoding="utf-8").lower()
        for heading in (
            "## what",
            "## why",
            "## config",
            "## implementation",
            "## evidence",
            "## caveat",
        ):
            assert heading in text, (name, heading)


def test_architecture_images_exist():
    for name in ARCHITECTURE_IMAGES:
        path = DOCS / "images" / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_demo_pages_report_measured_results():
    expected = {
        "case_studies/cifar10.md": "92.25",
        "case_studies/transfer.md": "94.98",
        "case_studies/optics_degradation.md": "85.71",
        "case_studies/ablations.md": "19.37",
        "bring_your_own_data.md": "60.00",
    }
    for relative, value in expected.items():
        text = (DOCS / relative).read_text(encoding="utf-8")
        assert value in text, (relative, value)
