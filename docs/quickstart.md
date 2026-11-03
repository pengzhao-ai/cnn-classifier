# Quickstart

## Install

### From PyPI

```bash
pip install cnn-classifier-toolkit
cnncls doctor
```

### From a checkout

The commands below use the repository's configs and demos, so run them from a
checkout. The package installs non-editable on purpose. On macOS, an editable
`.pth` can be hidden after PyTorch loads; the non-editable install keeps the
`cnncls` entry point reliable.

```bash
make sync
source .venv/bin/activate
cnncls doctor
```

Run `make sync` after changing library code. If the installed copy still has old
code, force it:

```bash
uv sync --all-extras --no-editable --reinstall-package cnn-classifier-toolkit
```

Without activating the environment, use `uv run --no-sync ...` rather than
plain `uv run`.

## First smoke test

The synthetic config exercises the whole pipeline without downloading data:

```bash
cnncls train -c configs/tests/synthetic.yaml --set training.epochs=1
```

Inspect the contract and debug before a longer run:

```bash
cnncls inspect-data -c configs/tests/synthetic.yaml
cnncls debug -c configs/tests/synthetic.yaml --overfit-batch --steps 20
cnncls train -c configs/tests/synthetic.yaml --dry-run
```

`train --dry-run` builds the dataset and model, runs one forward/backward pass,
prints the raw-to-intake contract and parameter counts, and writes nothing.

## Bring your own data

The committed fixture is the fastest real-data path:

```bash
python demos/bring_your_own_data/scripts/run_quickstart.py
```

It uses 60 16-bit grayscale images, trains for two epochs, evaluates the
validation split, and writes a report in a few seconds. See
[Bring your own data](bring_your_own_data.md) for the exact contract.

## CIFAR-10

Quick smoke (two epochs on the real split):

```bash
python demos/cifar10/scripts/run_scratch.py --quick
```

Full reference baseline (50 epochs):

```bash
python demos/cifar10/scripts/run_scratch.py
```

The first data access downloads CIFAR-10 (~170 MB) unless it is already under
`data/`. The `data/` and `runs/` directories are gitignored.

## Other demos

```bash
# three 32x32 ResNet-18 transfer runs, then a fairness comparison
python demos/cifar10/scripts/run_transfer.py

# four paired ablations on a reduced CIFAR split
python demos/ablations/run_ablations.py

# generate the optical-degradation dataset, then train
python demos/optics_degradation/generate_dataset.py --quick
python demos/optics_degradation/scripts/run_optics.py
```

The transfer, ablation, and optical full suites are compute-heavy; the demos
support quick iteration, while the committed results come from the full runs
described in the [demo pages](case_studies/cifar10.md).

## Where things go

- `runs/` — self-describing run directories, gitignored.
- `results/<demo>/` — committed summaries, figures, and reports.
- `configs/` — experiment YAML.
- `docs/` — this site.

## Next steps

- [Configuration](configuration.md) for the schema and overrides.
- [Models & architectures](models.md) for the built-in model families.
- [Techniques](techniques/augmentation.md) for the method explanations.
- [CLI](cli.md) for the full command reference.
