# CLI

The `cnncls` command is the orchestration layer. Everything it does is also
available through the library API.

## Commands

| Command | Purpose |
|---|---|
| `cnncls init DIR` | Scaffold a config, README, and data layout |
| `cnncls doctor` | Versions, device selection, optional dependencies |
| `cnncls inspect-data -c CONFIG` | Class balance, duplicates, corruption, leakage, model contract |
| `cnncls debug -c CONFIG` | One-batch overfit, shape trace, gradients, determinism |
| `cnncls train -c CONFIG` | Train, evaluate, write figures and the report |
| `cnncls evaluate --checkpoint PATH` | Evaluate on `test` or `val`, write predictions and metrics |
| `cnncls predict --checkpoint PATH --input PATH` | Image, folder, or manifest inference |
| `cnncls report --run DIR` | Rebuild HTML and Markdown from artifacts |
| `cnncls compare RUN... --out DIR` | Fair comparison with test-fingerprint enforcement |

## Common flows

```bash
cnncls doctor

cnncls inspect-data -c configs/tests/synthetic.yaml
cnncls debug -c configs/tests/synthetic.yaml --overfit-batch --steps 20
cnncls train -c configs/tests/synthetic.yaml --dry-run
cnncls train -c configs/tests/synthetic.yaml --set training.epochs=1

cnncls evaluate --checkpoint runs/<run>/checkpoints/best.pt --split test
cnncls predict --checkpoint runs/<run>/checkpoints/best.pt --input image.png
cnncls report --run runs/<run>
cnncls compare runs/<run-a> runs/<run-b> --out results/comparison
```

## Training options

```text
-c, --config PATH          experiment YAML
--set KEY=VALUE            repeatable typed override
--resume PATH              resume from a checkpoint
--dry-run                  build data/model, one forward/backward, write nothing
--runs-dir PATH            run directory root (default: runs)
--allow-config-change      allow resume with a changed config
--no-evaluate              skip the automatic post-training evaluation
```

## Exit behavior

- Configuration and validation errors exit non-zero with a readable message.
- A failed run is marked `failed` in its `status.json`.
- `compare` refuses runs with mismatched test fingerprints.
- `predict` accepts an image, a folder, or a manifest, and writes CSV/JSON.

See [Configuration](configuration.md) for the YAML contract and
[API](api.md) for the library entry points.
