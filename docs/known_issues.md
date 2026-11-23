# Known issues and revision backlog

Observed while walking the codebase and running the demos. Validated on
2026-11-22 against `e6460c0`. Items marked **Fixed in 0.1.1** are resolved;
**Closed** items were not reproducible or are by design; the rest remain open.
This file is internal and is excluded from the published documentation site.

Each entry records: what was observed, why it happens, what was actually run to
confirm it, and a proposed fix. Line references describe the code at the time of
writing.

Severity legend: **High** = wrong or missing output in a normal run; **Medium** =
inconsistent behaviour or a trap for the next reader; **Low** = cosmetic,
hygiene, or documentation.

| # | Status | Decision |
|---|--------|----------|
| 1 | Fixed in 0.1.1 | Batch-level gradient norm is written to history, metrics, summary, and reports. |
| 2 | Fixed in 0.1.1 | Nonfinite gradients stop the run before the optimizer step. |
| 3 | Open — cosmetic | Checkpoint tracks strict best; document or thread `min_delta` later. |
| 4 | Closed — not reproducible | A `val_loss`/`min` run wrote `best.pt` with `best_metric` 1.1154; the claim is incorrect. |
| 5 | Fixed in 0.1.1 | Added null sample-limit keys so the quick overrides apply. |
| 6 | Deferred — conditional | EMA checkpoint/resume contract; no shipped config enables EMA. |
| 7 | Open — cosmetic | Torchvision pseudo-path display. |
| 8 | Closed — by design | Non-editable install is intentional and documented in README/CONTRIBUTING. |
| 9 | Closed — resolved | No duplicate ` 2.py` files remain; `.DS_Store` is ignored. |
| 10 | Deferred — design | Callback resume dispatch; revisit with #11. |
| 11 | Deferred — conditional | Unfreeze schedule on resume; no shipped config uses it. |
| 12 | Open — decision | `predict` output default; document or change later. |

---

## 1. `GradientMonitor` never records `grad_norm` — High

**Symptom.** `grad_norm` never appears in `history.csv`, `metrics.jsonl`,
`summary.json`, or any other artifact, even though `GradientMonitor` is the first
callback registered in the default pipeline.

**Root cause.** `Trainer._train_epoch` ends every optimiser window with

```python
self.optimizer.zero_grad(set_to_none=True)   # trainer.py, after optimizer.step()
```

`set_to_none=True` assigns `None` to every `parameter.grad`. `GradientMonitor.on_epoch_end`
then iterates:

```python
for name, parameter in state.model.named_parameters():
    if parameter.grad is not None:           # never true
        report[name] = ...
...
if report:                                   # never true
    state.metrics["grad_norm"] = ...
```

so the metrics entry is silently skipped. The callback is wired correctly; it
simply runs at the wrong moment.

**Evidence.** A two-epoch run produced:

```text
step,epoch,lr,train_accuracy,train_loss,val_accuracy,val_loss
1,1.0,0.001,0.2999,1.8636528692245484,0.385,1.693062391281128
2,2.0,0.0,0.3858,1.6484640106201172,0.415,1.589163942337036
```

No `grad_norm` column, and `grep -r grad_norm` over all 31 artifacts found
nothing.

**Proposed fix.** Capture the norm while the gradients still exist.

1. *Preferred.* Add `on_batch_end(self, state)` to the `Callback` protocol and
   `_NoOpCallback`, call it at the end of each optimiser window in
   `_train_epoch` before `zero_grad`, and compute the norm there. Keep the last
   window's value and publish it from `on_epoch_end`.
2. *Minimal.* Have `_train_epoch` store `clip_grad_norm_`'s return value on
   `state.extra["grad_norm"]` and let the callback read it.

Option 1 is the right shape: it also gives `NaNDetector` a correct place to run
(issue 2) and future per-step telemetry a home.

**Tests to add.**

- Train one epoch on the synthetic adapter and assert `"grad_norm" in state.metrics`.
- Assert `grad_norm` increases when the learning rate is raised by an order of magnitude.

---

## 2. `NaNDetector`'s gradient check is dead code — High

**Symptom.** A run whose gradients become `nan`/`inf` is not stopped by the
gradient half of `NaNDetector`; only the metric half can trigger.

**Root cause.** Same as issue 1:

```python
gradients = [p.grad for p in state.model.parameters() if p.grad is not None]
...
any(not bool(torch.isfinite(gradient).all()) for gradient in gradients)
```

`gradients` is always `[]`, so `any(...)` is `False`. The metric check
(`math.isfinite` over `state.metrics`) still works, so a diverged **loss** is
caught. A `nan` confined to gradients that has not yet reached the epoch's
aggregate loss is not.

**Proposed fix.** Fold into issue 1's `on_batch_end` hook and check gradients
there, before `scaler.step()` runs.

**Tests to add.**

- Inject `inf` into one parameter's `.grad` and assert `state.stop_requested` is set.
- Assert a healthy short run leaves `failed is False`.

---

## 3. `ModelCheckpoint` ignores `min_delta` while `EarlyStopping` honours it — Medium

**Symptom.** Two definitions of "better" in one run. A +0.005 validation-accuracy
improvement is ignored by early stopping (with `min_delta=0.01`) but still
overwrites `best.pt`.

**Root cause.** `callbacks.py`, `ModelCheckpoint.on_epoch_end`:

```python
improved = _is_better(value, self.best, self.mode, 0.0)   # min_delta hardcoded
```

`ModelCheckpoint.__init__` has no `min_delta` parameter, whereas
`EarlyStopping.__init__` does and passes `self.min_delta` through. Arguably
intentional — the checkpoint should track the true best ever — but the
inconsistency is undocumented, and `_is_better` accepts the argument either way.

**Proposed fix.** Decide and document one policy. If the checkpoint should track
the strict best, say so in a comment and rename the local (`strict_improved`).
Otherwise thread `training.checkpoint.min_delta` from the schema into
`ModelCheckpoint`.

**Tests to add.**

- With `min_delta=0.01`, a +0.005 epoch must not rewrite `best.pt`.
- Assert the documented policy for `state.best_metric`.

---

## 4. `best_metric` starts at `0.0` but the checkpoint's `best` starts at `±inf` — Medium

**Symptom.** For a metric whose good values are negative (any loss, or a negative
score), the first epoch is compared against `0.0` instead of `+inf`, so no epoch
can be recorded as best and `best.pt` is never written.

**Root cause.** Two independent trackers with different initial values:

| Tracker | Initial value | Where |
|---|---|---|
| `TrainingState.best_metric` | `0.0` | `trainer.py`, `Trainer.__init__` |
| `ModelCheckpoint.best` | `-inf` (max) / `+inf` (min) | `callbacks._initial_best` |

For `val_accuracy` in `[0, 1]` this is harmless. With
`training.checkpoint.monitor` set to a loss and `mode: min`, `0.0 < inf`, so the
first epoch is already "best" and nothing ever improves on it.

**Proposed fix.** Initialise `TrainingState.best_metric` from `_initial_best(mode)`
using the configured monitor mode. Then `fit()`'s fallback (`best.pt` missing →
use `last.pt`) is only reachable in genuine edge cases.

**Tests to add.**

- Train with `checkpoint.monitor: val_loss`, `mode: min` for two epochs and
  assert `checkpoints/best.pt` exists and holds the lower-loss weights.

---

## 5. `run_scratch.py --quick` fails before training starts — High

**Symptom.**

```text
Error: unknown override path: data.adapter_params.max_train_samples
```

`--quick` is unusable as shipped, yet the demo README documents it.

**Root cause.** Two facts combine.

1. `src/cnn_classifier/config/presets/default.yaml` (47 lines) contains **no
   `data:` block**. `DataConfig.adapter_params` therefore falls back to the
   schema default, an empty dict:

   ```python
   # config/schema.py
   adapter_params: dict[str, Any] = Field(default_factory=dict)
   ```

2. `apply_overrides` deliberately refuses to create keys:

   ```python
   # config/loader.py
   leaf = parts[-1]
   if not isinstance(cursor, dict) or leaf not in cursor:
       raise ConfigError(f"unknown override path: {dotted}")
   ```

   Its docstring states the intent: *"Apply `a.b=value` overrides without
   silently growing the schema."* A free-form dict therefore cannot be extended
   from the command line.

The three other overrides in `QUICK_OVERRIDES` (`training.epochs`,
`evaluation.bootstrap.enabled`, `evaluation.latency.enabled`) are valid paths
and would work. Only the two sample-limit overrides fail. The adapter code that
consumes them is fine — `_sample_limit` reads `config.adapter_params.get(key)`.

**Evidence.** The demo reproduces the error at config load, before any data is
touched. A hand-written config with the two keys present in YAML trains normally
and produces a complete run directory.

**Proposed fix.** Pick one:

1. *Lowest risk.* Add the keys, commented out, to
   `configs/cifar10/scratch_baseline.yaml`:

   ```yaml
   adapter_params:
     dataset: CIFAR10
     download: true
     # max_train_samples: 10000
     # max_val_samples: 2000
   ```

   Then either enable them for quick mode or drop the two overrides from
   `QUICK_OVERRIDES` and rely on `training.epochs=2` alone (once the dataset is
   cached, training time is dominated by epochs, not sample count).

2. *More general.* Allow `apply_overrides` to create leaves inside declared
   free-form mapping fields (`adapter_params`, `model.params`). This needs a
   schema-side declaration of which paths are extensible, otherwise it defeats
   the typo protection.

3. *Alternative.* Give `--quick` a dedicated config file
   (`configs/cifar10/scratch_quick.yaml`) instead of using overrides.

**Tests to add.**

- `run_scratch.py --quick` exits 0 and writes a run directory.
- `apply_overrides` still rejects `training.epoch=5`, preserving typo protection.

---

## 6. `best.pt` stores EMA weights but live optimiser state — Medium

**Symptom.** When `training.ema.enabled` is true, resuming from `best.pt` pairs
averaged weights with the optimiser and scheduler state of the live model.

**Root cause.** `callbacks.py`, `ModelCheckpoint._save`:

```python
save_checkpoint(
    ...,
    model=state.eval_model or state.model,     # EMA weights when enabled
    optimizer=state.optimizer,                 # live optimiser state
    scheduler=state.scheduler,
    scaler=state.scaler,
    ...,
)
```

The weights come from the EMA copy; the optimiser moments belong to the live
model. `last.pt` has the same property. Resuming normally uses `last.pt`, so the
impact is limited, but the pairing is inconsistent.

**Proposed fix.** Save `model=state.model` for resumption and record the EMA
weights under a separate key (for example `"ema_model"`), or write a distinct
`best_ema.pt`. Document which checkpoint is for inference and which for resume.

**Tests to add.**

- With EMA enabled, assert the checkpoint used for resume reloads weights that
  match `state.model`, and the deployment checkpoint matches `state.eval_model`.

---

## 7. `torchvision` adapter has no real sample paths — Low

**Symptom.** Error galleries and any output showing a file path display
`<torchvision:1234>` instead of an image path.

**Root cause.** `data/adapters/torchvision.py`:

```python
metadata = SampleMetadata(
    paths=tuple(Path(f"<torchvision:{index}>") for index in range(len(train_base))),
    ...
)
```

There are no files to point at — CIFAR-10 lives in pickled batches inside
`data/cifar-10-batches-py/`.

**Proposed fix.** Document the placeholder, or synthesise a
`<torchvision:CIFAR10/train/{split_index}>` style pseudo-path that at least names
the dataset and split.

---

## 8. Development environment installs a copy, not an editable link — Medium

**Symptom.** Edits to `src/cnn_classifier/**` have no effect on
`python demos/...` or `cnncls` runs.

**Root cause.** The package is installed into the virtual environment as a
regular copy at `.venv/lib/python3.11/site-packages/cnn_classifier/`, and
`import cnn_classifier` resolves there rather than to `src/cnn_classifier`. A
content diff between the two trees currently shows no differences, so this
checkout is in sync — but the workflow invites silent divergence.

**Proposed fix.** Install in editable mode (`pip install -e .`) for development,
or inject `src/` onto `sys.path` for the test suite. Document the reinstall step
in `CONTRIBUTING.md`.

**Tests to add.** A smoke check that `cnn_classifier.__file__` starts with the
repository root when running from a checkout.

---

## 9. Stray duplicated files inside `site-packages` — Low

**Symptom.** `site-packages/cnn_classifier/` contains second copies of many
modules named with a ` 2` suffix (`cli 2.py`, `core/device 2.py`,
`data/adapters/manifest 2.py`, ...), and `.DS_Store` files are present in
`src/cnn_classifier`.

**Root cause.** Most likely a file-sync conflict retained as duplicates during an
install, not a packaging bug. They are inert — nothing imports them — but they
appear in every directory diff.

**Proposed fix.** Delete them, and add `.DS_Store` to `.gitignore`.

---

## 10. `_restore_callback_state` dispatches with `isinstance` — Low (design)

**Symptom.** A new stateful callback silently loses its state on resume unless
someone remembers to edit `Trainer`.

**Root cause.** `trainer.py`, `_restore_callback_state`:

```python
for callback in self.callbacks:
    if isinstance(callback, ModelCheckpoint):
        callback.restore_best(self.state.best_metric)
    elif isinstance(callback, EarlyStopping):
        callback.restore_from_history(self.state.history, self.state.best_metric)
```

The callback protocol has three hooks but no resume hook, so resume knowledge
leaks back into the trainer and must be extended by editing it.

**Proposed fix.** Add `on_resume(self, state) -> None` to the protocol and
`_NoOpCallback`, then call `callback.on_resume(self.state)` for every callback.
`ModelCheckpoint` and `EarlyStopping` implement it; the rest inherit the no-op.

**Tests to add.**

- Resume a run with early stopping and assert patience is not reset.
- Resume with a custom stateful callback and assert `on_resume` was called.

---

## 11. `UnfreezeCallback` never fires after a resume past its epoch — Low

**Symptom.** If training is interrupted after the scheduled unfreeze epoch and
resumed, the unfreeze event does not re-fire.

**Root cause.** `callbacks.py`:

```python
if event.epoch == state.epoch:
    unfreeze_modules(state.model, event.modules)
```

Exact equality against the current epoch. On resume the loop starts at
`_start_epoch + 1`, so an event whose epoch already passed is skipped.

**Assessment.** Usually harmless, because the unfrozen state is part of the saved
weights. It becomes a real problem if `requires_grad` differs between the saved
and freshly built model.

**Proposed fix.** Fire on `state.epoch >= event.epoch` and track which events have
already fired, deriving the initial set from `_start_epoch`.

---

## 12. `predict` output location needs an explicit decision — Low

**Symptom.** `cnncls predict` defaults to writing `predictions.csv` into the
caller's working directory rather than the run directory, which scatters outputs.

**Assessment.** Not confirmed as a defect — the default may be deliberate. Worth
an explicit decision and a line in `docs/cli.md`.

---

## Reproduction notes

Environment used for these observations:

```text
device        mps (Apple Silicon)
python        3.11.8
torch         2.15.0
torchvision   0.29.2
typer         0.27.3
```

Commands:

```bash
# CIFAR-10 quick check (the shipped --quick flag fails; issue 5)
python demos/cifar10/scripts/run_scratch.py --quick

# equivalent hand-written config used to confirm the run itself works
cnncls train -c /tmp/cifar_quick.yaml --runs-dir /tmp/cifar_quick

# inspect for issue 1
cat <run-dir>/history.csv
grep -rl grad_norm <run-dir>/
```