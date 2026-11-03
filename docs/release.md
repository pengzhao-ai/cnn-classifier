# Release

## Local verification

Before a release, run the same gates CI runs:

```bash
make gate
uv build
uvx twine check dist/*
```

`make gate` covers lint, formatting, mypy, tests, the coverage gate, the strict
docs build, and the wheel/sdist build. `twine check` verifies the package
metadata renders.

## Versioning and tags

The version is dynamic from `src/cnn_classifier/_version.py`. A release is a
`v*` tag. The release workflow is designed to:

1. run the test suite on Python 3.11–3.13;
2. build the wheel and sdist;
3. run `twine check`;
4. publish the GitHub release assets;
5. publish to PyPI through trusted publishing (OIDC), with no long-lived token.

## CI

The quality workflow runs ruff, mypy, tests on Ubuntu and macOS for Python
3.11–3.13, the docs strict build, and the package build. Full training is never
run in CI; the synthetic smoke and the bring-your-own-data quickstart are the
CI-scale checks.

## Owner checklist

These steps require the repository owner's accounts and cannot be automated by
the build:

- [ ] Create or confirm `pengzhao-ai/cnn-classifier`.
- [ ] Enable GitHub Pages with the docs workflow as the source.
- [ ] Configure the PyPI trusted publisher for the release workflow.
- [ ] Push the `v*` tag.
- [ ] Verify the docs URL and the PyPI page render, including README images.
- [ ] Confirm the release assets contain the wheel and sdist.

## Documentation

The docs site is built with MkDocs Material and deployed from `main` through
GitHub Pages. The canonical URLs are:

- Repository: `https://github.com/pengzhao-ai/cnn-classifier`
- Documentation: `https://pengzhao-ai.github.io/cnn-classifier/`

Live publishing is verified by the owner after account setup; the local build
does not assume it succeeded.
