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
metadata renders on PyPI.

## Versioning

The version is dynamic from `src/cnn_classifier/_version.py`. A release is a
`v*` Git tag. The changelog is in `CHANGELOG.md`; the citation metadata is in
`CITATION.cff`.

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push to `main`, pull request, manual | quality, tests on Python 3.11–3.13 / Ubuntu and macOS, synthetic and BYO smoke, strict docs, build |
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

The owner must configure the trusted publisher on PyPI with the repository,
workflow filename, and environment name before the first release.

## Owner checklist

These steps require the repository owner's accounts and are not automated by
the local build:

- [ ] Create or confirm `pengzhao-ai/cnn-classifier`.
- [ ] Enable GitHub Pages and select GitHub Actions as the source.
- [ ] Confirm the `docs.yml` workflow can deploy to the `github-pages`
      environment.
- [ ] Configure the PyPI trusted publisher for `release.yml` and the `pypi`
      environment.
- [ ] Push the `v*` tag.
- [ ] Verify the GitHub release assets contain the wheel and sdist.
- [ ] Verify the PyPI page renders, including README links and images.
- [ ] Verify the documentation URL
      `https://pengzhao-ai.github.io/cnn-classifier/` is live.

Live PyPI and Pages publication is verified by the owner after account setup.
The local build does not assume it succeeded.
