# Bring your own data — results

Status: fixture, config, and quickstart are complete. The quickstart is a
pipeline smoke check, not a meaningful accuracy benchmark.

## Fixture

- 3 classes: circle, square, stripe.
- 60 images: 45 train / 15 validation.
- 16×16, 16-bit grayscale PNGs; seed 7.
- Manifest: `path,label,split,group`; no group leakage.

## Contract

| Field | Value |
|---|---|
| dtype | `float32` |
| one sample | `[1, 16, 16]` |
| one batch | `[B, 1, 16, 16]` |
| value range | `[0.0, 1.0]` |
| label map | circle: 0, square: 1, stripe: 2 |
| normalization | computed from the training split |

## Quickstart result

The run completes in a few seconds on CPU or MPS and writes `report.html`,
`evaluation.json`, predictions, figures, and checkpoints under its run
directory. These artifacts are local (`runs/` is gitignored); the quickstart is
the CI smoke path, not a published benchmark.

## Notes

- The fixture is committed so the demo needs no network.
- The 16-bit reader scales the full stored range to `[0, 1]`.
- The test-set discipline still applies: this fixture uses train/validation, and
  the trainer never iterates a test loader.
