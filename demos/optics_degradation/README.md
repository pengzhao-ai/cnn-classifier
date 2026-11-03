# Optical degradation classification

## Question

Can the toolkit classify the dominant optical degradation in an image, and can
it do so with leakage-safe splits and useful confusion diagnostics?

This is a framework and diagnostics showcase. The images are 32×32 CIFAR-10
base images with synthetic degradations, so it is not a claim about production
image-quality assessment. The point is that a second, manifest-based dataset
runs through the same trainer, evaluator, and report path as CIFAR-10.

## Degradations

| Label | Optical analogue | Implementation |
|---|---|---|
| `clean` | no dominant degradation | unchanged image |
| `defocus` | defocused point-spread function | Gaussian blur |
| `motion_blur` | camera or subject motion during exposure | oriented line kernel |
| `chromatic_aberration` | wavelength-dependent magnification | per-channel radial scale |
| `vignetting` | off-axis illumination falloff | radial gain mask |
| `sensor_noise` | photon and read noise | Gaussian + Poisson noise |
| `jpeg` | compression artifacts | JPEG round-trip at reduced quality |

Severity is sampled per image from a configured range (`0.2–1.0` for the six
degradations, `0.0` for clean) and recorded in the manifest.

## Data contract

The generator writes:

```text
data/optics_degradation/
├── base/                 # selected CIFAR-10 base images (gitignored)
├── images/               # generated samples (gitignored)
└── manifest.csv          # path,label,split,group,severity
```

Each base image is a **group**. Groups are assigned to train/validation/test
before any degraded sample is generated, so the same base image never appears
in two splits. The manifest adapter reads the CSV and the explicit split
column; the test split is evaluated only after training.

`generate_fixture` is deterministic for a given seed. The CLI uses selected
CIFAR-10 training images as the base source and supports `--quick` for a small
dataset.

## Run

From an activated environment at the repository root:

```bash
# quick synthetic dataset for iteration
python demos/optics_degradation/generate_dataset.py --quick

# inspect the manifest and model contract
cnncls train -c configs/optics_degradation/degradation_classifier.yaml --dry-run

# train, evaluate, and report
python demos/optics_degradation/scripts/run_optics.py
```

A full generation without `--quick` uses 300 base images (2,100 samples). The
quick path uses 60 base images (420 samples). The full MPS run has been
completed; committed results are in `results/optics_degradation/`.

## Results

Full results, confusion analysis, and committed artifacts are in
`results/optics_degradation/` and described in
`demos/optics_degradation/results.md`.

## What to notice

- **Group leakage is checked before generation.** The manifest records the
  source group, and the split never mixes groups.
- **A second dataset needs no code changes.** It uses the manifest adapter and
  the same trainer/evaluator/report path as the CIFAR-10 demos.
- **The diagnostics matter.** The confusion matrix shows which degradations are
  confused; the report includes per-class metrics, calibration, and error
  galleries.
- **The caveat is explicit.** 32×32 synthetic degradations are a framework
  demonstration, not a production image-quality benchmark.
