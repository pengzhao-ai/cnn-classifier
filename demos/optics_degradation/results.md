# Optical degradation classification — results

Status: generator, config, and runner are complete; the full MPS run and
committed artifacts are produced in Task 28 of the sharp-v1 plan.

## Dataset

- 32×32 CIFAR-10 base images with synthetic degradations.
- Seven classes: clean, defocus, motion_blur, chromatic_aberration, vignetting,
  sensor_noise, jpeg.
- Group-aware split by base image; seed 1; severity sampled per sample.
- Quick mode: 60 base images / 420 samples. Full mode: 300 base images /
  2,100 samples.

## Results (pending)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| clean | pending | pending | pending | pending |
| defocus | pending | pending | pending | pending |
| motion_blur | pending | pending | pending | pending |
| chromatic_aberration | pending | pending | pending | pending |
| vignetting | pending | pending | pending | pending |
| sensor_noise | pending | pending | pending | pending |
| jpeg | pending | pending | pending | pending |

## Artifacts

After Task 28, this file will link to `results/optics_degradation/` (report,
figures, summary, and confusion analysis). Checkpoints remain local.

## Notes

- Group leakage is required to be empty; the generator records the source group
  in every manifest row.
- The test split is held out until after training.
- The demo is a framework and diagnostics showcase, not a production
  image-quality benchmark.
