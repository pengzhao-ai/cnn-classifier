# Release

## Local verification

Run the same gates CI runs before tagging:

```bash
make gate
uv build
uvx twine check dist/*
```

`make gate` covers lint, formatting, mypy, tests, the coverage gate, the strict
docs build, and the wheel/sdist build. `twine check` verifies that the package
metadata renders on PyPI. The artifacts are named
`cnn_classifier_toolkit-<version>-py3-none-any.whl` and
`cnn_classifier_toolkit-<version>.tar.gz`.

## Versioning

The version is dynamic from `src/cnn_classifier/_version.py`. A release is a
`v*` Git tag. The changelog is in `CHANGELOG.md`; the citation metadata is in
`CITATION.cff`.

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push to `main`, pull request, manual | quality, tests on Python 3.11–3.13 / Ubuntu and macOS arm64, synthetic and BYO smoke, strict docs, build |
| `docs.yml` | push to `main`, manual | strict MkDocs build and GitHub Pages deployment |
| `release.yml` | `v*` tag | tests, wheel/sdist, `twine check`, GitHub release assets, PyPI trusted publishing |

CI never runs a full training suite. It uses the synthetic config and the
committed bring-your-own-data fixture.

## PyPI trusted publishing

The release workflow uses PyPI trusted publishing through OpenID Connect:

- the job has `id-token: write`;
- it targets the `pypi` environment;
- `pypa/gh-action-pypi-publish` publishes the built distributions;
- no long-lived PyPI token or password is stored in the repository.

The owner must configure the PyPI trusted publisher for the distribution
`cnn-classifier-toolkit`, workflow `release.yml`, and environment `pypi` before
the first release. The import package remains `cnn_classifier`; the PyPI
distribution name is different because `cnn-classifier` collides with an
existing project under PyPI's name-similarity rule.

For a new PyPI project, register a pending publisher with these exact values:

| Field | Value |
|---|---|
| PyPI project name | `cnn-classifier-toolkit` |
| Owner | `pengzhao-ai` |
| Repository name | `cnn-classifier` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

## Owner checklist

These steps require the repository owner's accounts and are not automated by
the local build:

- [x] Create or confirm `pengzhao-ai/cnn-classifier`.
- [x] Enable GitHub Pages and select GitHub Actions as the source.
- [x] Confirm the `docs.yml` workflow can deploy to the `github-pages`
      environment.
- [x] Configure the PyPI trusted publisher for project
      `cnn-classifier-toolkit`, workflow `release.yml`, and environment `pypi`.
- [x] Push the `v*` tag.
- [x] Verify the GitHub release assets contain the wheel and sdist.
- [x] Verify the PyPI page
      `https://pypi.org/project/cnn-classifier-toolkit/` renders, including
      README links and images.
- [x] Verify the documentation URL
      `https://pengzhao-ai.github.io/cnn-classifier/` is live.

Live PyPI and Pages publication was verified on 2026-11-03:
`v0.1.0` published both artifacts, and the docs site is serving the five-section
public navigation.
