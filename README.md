# cnn-classifier

A config-driven, reproducible PyTorch toolkit for CNN image classification.

`cnn-classifier` is built for practitioners who want a reusable training and
evaluation pipeline rather than another notebook. It provides typed YAML
configuration, dataset adapters, a configurable residual CNN, torchvision
transfer learning, run artifacts, evaluation reports, and diagnostics.

The project is in active development. The sharp-v1 design and implementation
plan live under `docs/superpowers/`.

## Install

```bash
uv sync --all-extras
```

## Development

```bash
make test
make lint
make typecheck
make docs
make build
```

## Documentation

The documentation site will be published at
<https://pengzhao-ai.github.io/cnn-classifier/>.

## License

Apache-2.0. See `LICENSE`.
