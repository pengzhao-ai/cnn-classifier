# Contributing

## Setup

```bash
make sync
source .venv/bin/activate
cnncls doctor
```

The package is intentionally non-editable. Run `make sync` after changing
library code; if the installed copy is stale, use:

```bash
uv sync --all-extras --no-editable --reinstall-package cnn-classifier-toolkit
```

## Development workflow

1. Read the relevant design notes under `docs/`.
2. Write a failing test first. Run it and confirm the failure is for the right
   reason.
3. Implement the smallest change that passes.
4. Run the focused test, then the full suite.
5. Run the quality gate before opening a pull request:

```bash
make gate
```

## Style

- `ruff check` and `ruff format` are required.
- `mypy src` must pass.
- Keep training headless: do not import `matplotlib` or `jinja2` from training.
- Keep demo names generic; no notebook-derived or dataset-specific class names.
- Add or update tests with every behavior change.

## Tests

```bash
make test        # full suite
make coverage    # library coverage gate
make docs        # strict docs build
make build       # wheel and sdist
```

Slow data downloads and full training runs are not part of CI. Use the
synthetic config and the bring-your-own-data quickstart for fast checks.

## Commits and pull requests

- Keep commits focused and explain the why in the message.
- Reference the task or issue when applicable.
- Do not commit checkpoints, raw datasets, or runs.
- Include the verification commands you ran in the pull request description.

## License

By contributing, you agree that your contribution is licensed under
Apache-2.0.
