# Third-party software

`cnn-classifier` depends on the following open-source projects. The license
listed is the primary license of the upstream project; consult the installed
package metadata for the authoritative text and any bundled notices.

| Package | Purpose | License |
|---|---|---|
| PyTorch (`torch`) | tensor and neural-network runtime | BSD-3-Clause |
| TorchVision (`torchvision`) | datasets, transforms, pretrained models | BSD-3-Clause |
| Pydantic | typed configuration schema | MIT |
| PyYAML | YAML loading | MIT |
| NumPy | numerical arrays | BSD-3-Clause |
| Pillow | image decoding and encoding | HPND |
| tifffile | TIFF I/O including 16-bit data | BSD-3-Clause |
| Matplotlib | figures | PSF-based (Matplotlib license) |
| Jinja2 | report templates | BSD-3-Clause |
| Typer | command line interface | MIT |
| Rich | terminal output | MIT |
| MkDocs Material | documentation site | MIT |
| mkdocstrings | generated API reference | ISC |
| pytest | tests | MIT |
| pytest-cov | coverage measurement | MIT |
| Hypothesis | property-based tests | MPL-2.0 |
| ruff | lint and formatting | MIT |
| mypy | static typing | MIT |
| pre-commit | Git hooks | MIT |
| pandas | test-side tabular helpers | BSD-3-Clause |
| TensorBoard | optional tracking backend | Apache-2.0 |
| scikit-learn | optional extra metrics | BSD-3-Clause |
| torchinfo | optional model summaries | MIT |

Demo datasets are downloaded by the user through TorchVision (CIFAR-10). They
are not redistributed with this package. Generated demo fixtures are original
synthetic data.
