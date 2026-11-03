# Extending

The library exposes three registries. A custom component registers itself and
becomes available to the same config, trainer, evaluator, and report path.

## Custom dataset adapters

```python
from cnn_classifier.core.types import DatasetBundle
from cnn_classifier.data.adapters import register_dataset


@register_dataset("my_format")
class MyFormatAdapter:
    @staticmethod
    def build(config) -> DatasetBundle:
        # Read files, build train/val/test datasets, resolve class names,
        # normalization, split info, sample metadata, and the model contract.
        ...
```

Then set `data.adapter: my_format`. The module must be imported before the run;
demos can do this in their own entry-point script, so no library edit is needed.
The bring-your-own-data guide has a worked example and the full contract.

## Custom model builders

```python
from torch import nn
from cnn_classifier.models import ModelBuildContext, register_model


@register_model("my_model")
class MyModelBuilder:
    Params = MyParams  # a Pydantic model

    @staticmethod
    def build(params, context: ModelBuildContext) -> nn.Module:
        return MyModel(params, context.num_classes, context.in_channels)
```

The builder receives the class count, input channels, image size, and transfer
settings. `model.params` is validated against `Params` before `build`.

## Custom transforms

```python
from cnn_classifier.data.transforms import register_transform


@register_transform("my_transform")
def build_my_transform(spec, normalization):
    strength = spec.params().get("strength", 1.0)
    return lambda image: ...
```

The factory receives the validated transform config and the run's normalization
statistics (or `None`). Return a callable that accepts a PIL image / NumPy array
/ tensor and returns a tensor.

## What stays fixed

The `DatasetBundle` and `ModelBuildContext` contracts are the integration
surface. A custom adapter must produce the same bundle fields; a custom model
must accept the context. Keeping those contracts stable is what lets the
trainer, evaluator, and report work unchanged.

## Testing extensions

- Build the component directly and assert its contract.
- Run `cnncls inspect-data -c config.yaml` and `train --dry-run`.
- Run a one-epoch smoke before a full budget.
- Add the config to the repo's config tests if it is a committed demo.

## Where to go next

- [Bring your own data](bring_your_own_data.md) for the raw-to-intake contract.
- [Configuration](configuration.md) for the schema.
- [API](api.md) for the typed signatures.
