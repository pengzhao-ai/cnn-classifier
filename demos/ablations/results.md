# CIFAR-10 core ablations — results

Status: configurations and runner are complete; the full eight-run MPS suite and
committed artifacts are produced in Task 28 of the sharp-v1 plan.

## Budget

- CIFAR-10 reduced split: 10,000 train / 2,000 validation; 10,000 test.
- Seed 42; 8 epochs; batch 128; AdamW 1e-3; weight decay 5e-4.
- Residual CNN with widths 32/64/128, dropout 0.2.

## Pairs (pending)

| Pair | Knob on | Knob off | Accuracy difference | 95% CI overlap |
|---|---|---|---|---|
| Augmentation | random crop + flip | none | pending | pending |
| BatchNorm | batch | none | pending | pending |
| Residual | true | false | pending | pending |
| Schedule | cosine | constant | pending | pending |

## Artifacts

After Task 28, this file will link to `results/ablations/comparison.md`,
`comparison.json`, `comparison.png`, and the per-run reports.

## Notes

- Per-run bootstrap intervals are recorded in each run's `summary.json`.
- No paired significance claim is made; overlapping intervals are not evidence
  of no effect.
- Timings are measurements on the author's Apple M3 Max MPS machine, not
  guarantees.
