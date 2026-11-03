# Result packaging

These scripts regenerate the committed result artifacts and the demo
architecture figures. They are not needed at runtime.

Run them after the demo suites have completed:

```bash
python scripts/package_results.py
python scripts/make_comparisons.py
python scripts/paired_comparisons.py
python scripts/make_architecture_figures.py
```

`package_results.py` copies summaries, evaluations, reports, and selected
figures from `runs/` into `results/<demo>/`, excluding checkpoints and resolved
configs with absolute paths. `make_comparisons.py` writes the labeled transfer
and ablation comparison tables/JSON. `paired_comparisons.py` computes paired
bootstrap intervals from the saved test predictions. `make_architecture_figures.py`
builds each demo model from its config and writes the stage-pipeline figures
under `docs/images/`.

The committed results were produced by these scripts on the author's Apple M3
Max MPS machine. Timings are measurements, not guarantees.
