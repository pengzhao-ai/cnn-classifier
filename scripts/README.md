# Result packaging

These scripts turn local run directories into the committed `results/` tree and
regenerate the labeled comparison artifacts. They are not needed at runtime.

Run them after the demo suites have completed:

```bash
python scripts/package_results.py
python scripts/make_comparisons.py
```

`package_results.py` copies summaries, evaluations, reports, and selected
figures from `runs/` into `results/<demo>/`, excluding checkpoints and resolved
configs with absolute paths. `make_comparisons.py` loads the packaged summaries,
writes the labeled transfer and ablation comparison tables/JSON, and mirrors
selected figures into `docs/images/`.

The committed results were produced by these scripts on the author's Apple M3
Max MPS machine. Timings are measurements, not guarantees.
