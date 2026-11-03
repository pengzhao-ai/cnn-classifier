"""Stage-level architecture descriptors for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from torch import nn

from cnn_classifier.diagnostics.tracing import LayerShape, trace_shapes


@dataclass(frozen=True)
class ArchitectureStage:
    name: str
    kind: str
    output_shape: tuple[int, ...]
    params: int
    trainable_params: int
    residual: bool = False

    @property
    def frozen(self) -> bool:
        return self.trainable_params == 0


def _output_shape(record: LayerShape | None) -> tuple[int, ...]:
    if record is None:
        return ()
    shape = tuple(int(value) for value in record.output_shape)
    if len(shape) > 1 and shape[0] == 1:
        return shape[1:]
    return shape


def _stage(
    name: str,
    kind: str,
    modules: list[nn.Module],
    records: dict[str, LayerShape],
    record_name: str,
    *,
    residual: bool = False,
) -> ArchitectureStage:
    params = sum(parameter.numel() for module in modules for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return ArchitectureStage(
        name=name,
        kind=kind,
        output_shape=_output_shape(records.get(record_name)),
        params=params,
        trainable_params=trainable,
        residual=residual,
    )


def _sequential_stages(model: nn.Module, records: dict[str, LayerShape]) -> list[ArchitectureStage]:
    features = cast(nn.Sequential, model.features)
    groups: list[list[tuple[int, nn.Module]]] = []
    current: list[tuple[int, nn.Module]] = []
    for index, module in enumerate(features.children()):
        current.append((index, module))
        if isinstance(module, nn.MaxPool2d):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    stages = [
        _stage(
            f"stage {index}",
            "stage",
            [module for _, module in group],
            records,
            f"features.{group[-1][0]}",
        )
        for index, group in enumerate(groups, start=1)
    ]
    stages.append(_stage("head", "head", _children(cast(nn.Module, model.head)), records, "head"))
    return stages


def _children(module: nn.Module) -> list[nn.Module]:
    if isinstance(module, nn.Sequential):
        return list(module.children())
    return [module]


def _residual_stages(model: nn.Module, records: dict[str, LayerShape]) -> list[ArchitectureStage]:
    features = cast(nn.Sequential, model.features)
    spans = cast(tuple[tuple[str, int, int, bool], ...], model.stage_spans)
    stages = [
        _stage(
            name,
            "stem" if name == "stem" else "stage",
            [features[index] for index in range(start, end)],
            records,
            f"features.{end - 1}",
            residual=residual,
        )
        for name, start, end, residual in spans
    ]
    stages.append(_stage("head", "head", _children(cast(nn.Module, model.head)), records, "head"))
    return stages


def _torchvision_stages(
    model: nn.Module, records: dict[str, LayerShape]
) -> list[ArchitectureStage]:
    children = dict(model.named_children())
    stem_names = [name for name in ("conv1", "bn1", "relu", "maxpool") if name in children]
    stages = [
        _stage(
            "stem",
            "stem",
            [children[name] for name in stem_names],
            records,
            stem_names[-1],
        )
    ]
    for name in ("layer1", "layer2", "layer3", "layer4"):
        if name in children:
            stages.append(
                _stage(
                    name.replace("layer", "layer "),
                    "stage",
                    [children[name]],
                    records,
                    name,
                    residual=True,
                )
            )
    head_names: list[str] = [name for name in ("avgpool", "fc") if name in children]
    if not head_names:
        head_names = list(children)
    stages.append(
        _stage(
            "head",
            "head",
            [children[name] for name in head_names],
            records,
            head_names[-1],
        )
    )
    return stages


def architecture_stages(
    model: nn.Module, input_shape: tuple[int, int, int]
) -> list[ArchitectureStage]:
    """Return concise stage descriptors for a built-in CNN.

    The built-in ``simple_cnn`` and ``residual_cnn`` families and torchvision
    ResNet models get precise grouping. Other models fall back to their
    top-level children.
    """
    records = {record.name: record for record in trace_shapes(model, input_shape)}
    if hasattr(model, "stage_spans") and hasattr(model, "features"):
        return _residual_stages(model, records)
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        return _sequential_stages(model, records)
    children = dict(model.named_children())
    if "conv1" in children and "layer1" in children:
        return _torchvision_stages(model, records)
    stages = [_stage(name, "stage", [module], records, name) for name, module in children.items()]
    if not stages:
        raise ValueError("model has no submodules to visualize")
    return stages
