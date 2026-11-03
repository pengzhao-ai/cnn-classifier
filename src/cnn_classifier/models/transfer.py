"""Transfer-learning helpers for stem adaptation, freezing, and param groups."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import nn

from cnn_classifier.config.schema import TransferConfig
from cnn_classifier.core.errors import ConfigError

StemMode = Literal["none", "keep_7x7_stride1", "adapt_3x3"]


def _head_name(model: nn.Module) -> str:
    if hasattr(model, "fc") and isinstance(model.fc, (nn.Linear, nn.Sequential)):
        return "fc"
    if hasattr(model, "classifier") and isinstance(model.classifier, (nn.Linear, nn.Sequential)):
        return "classifier"
    raise ConfigError("cannot identify the classifier head for this model")


def _last_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, nn.Sequential):
        for child in reversed(module):
            if isinstance(child, nn.Linear):
                return child
    raise ConfigError("classifier does not contain a Linear layer")


def replace_classifier(model: nn.Module, num_classes: int, dropout: float = 0.0) -> nn.Module:
    """Replace the final classifier with a task-specific head."""
    if hasattr(model, "fc") and isinstance(model.fc, (nn.Linear, nn.Sequential)):
        in_features = _last_linear(model.fc).in_features
        head = nn.Linear(in_features, num_classes)
        model.fc = nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head
        return model
    if hasattr(model, "classifier") and isinstance(model.classifier, (nn.Linear, nn.Sequential)):
        classifier = model.classifier
        in_features = _last_linear(classifier).in_features
        new_layer = nn.Linear(in_features, num_classes)
        if isinstance(classifier, nn.Linear):
            model.classifier = new_layer
        else:
            for index in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[index], nn.Linear):
                    classifier[index] = new_layer
                    break
        return model
    raise ConfigError("unsupported torchvision classifier head")


def apply_stem_adaptation(
    model: nn.Module,
    mode: StemMode,
    in_channels: int = 3,
) -> nn.Module:
    """Adapt a torchvision stem for native 32x32 inputs."""
    if mode == "none":
        return model
    if not hasattr(model, "conv1") or not isinstance(model.conv1, nn.Conv2d):
        raise ConfigError(f"model {type(model).__name__} has no supported conv1 stem")
    conv = model.conv1
    if conv.in_channels != in_channels:
        raise ConfigError(
            f"stem expects {conv.in_channels} input channels but data has {in_channels}"
        )
    if mode == "keep_7x7_stride1":
        conv.stride = (1, 1)
        model.maxpool = nn.Identity()
        return model
    if mode == "adapt_3x3":
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            3,
            1,
            1,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        kernel = conv.weight.data
        height, width = kernel.shape[-2:]
        top = (height - 3) // 2
        left = (width - 3) // 2
        with torch.no_grad():
            new_conv.weight.copy_(kernel[:, :, top : top + 3, left : left + 3])
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias.data)
        model.conv1 = new_conv
        model.maxpool = nn.Identity()
        return model
    raise ConfigError(f"unknown stem adaptation {mode!r}")


def _head_parameters(model: nn.Module) -> set[str]:
    head = _head_name(model)
    return {
        name for name, _ in model.named_parameters() if name == head or name.startswith(f"{head}.")
    }


def apply_transfer(
    model: nn.Module,
    transfer: TransferConfig,
    *,
    stem_adaptation: str,
) -> nn.Module:
    """Apply freezing, BatchNorm policy, and validation for a transfer run."""
    if transfer.freeze in {"backbone", "all_but_head"}:
        if stem_adaptation == "adapt_3x3":
            raise ConfigError(
                "frozen adapt_3x3 is invalid: the uninitialized stem would block ImageNet features"
            )
        head_names = _head_parameters(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in head_names
        if transfer.train_bn:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
    return model


def build_param_groups(
    model: nn.Module,
    transfer: TransferConfig,
    *,
    base_lr: float,
    base_weight_decay: float,
) -> list[dict[str, Any]]:
    """Build disjoint optimizer groups, validating every regex pattern."""
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not trainable:
        raise ConfigError("model has no trainable parameters")
    remaining = dict(trainable)
    groups: list[dict[str, Any]] = []
    for group_config in transfer.param_groups:
        pattern = re.compile(group_config.pattern)
        matched = [
            (name, parameter) for name, parameter in remaining.items() if pattern.search(name)
        ]
        if not matched:
            raise ConfigError(
                f"param group pattern {group_config.pattern!r} did not match "
                "any trainable parameter"
            )
        groups.append(
            {
                "params": [parameter for _, parameter in matched],
                "lr": base_lr * group_config.lr_multiplier,
                "weight_decay": base_weight_decay * group_config.weight_decay_multiplier,
            }
        )
        for name, _ in matched:
            del remaining[name]
    if remaining:
        groups.insert(
            0,
            {
                "params": list(remaining.values()),
                "lr": base_lr,
                "weight_decay": base_weight_decay,
            },
        )
    if not groups:
        groups = [
            {
                "params": list(trainable.values()),
                "lr": base_lr,
                "weight_decay": base_weight_decay,
            }
        ]
    return groups


def unfreeze_modules(model: nn.Module, names: Sequence[str]) -> None:
    """Unfreeze every parameter under the named submodules."""
    modules = dict(model.named_modules())
    for name in names:
        if name not in modules:
            raise ConfigError(f"cannot unfreeze unknown module {name!r}")
        for parameter in modules[name].parameters():
            parameter.requires_grad = True
