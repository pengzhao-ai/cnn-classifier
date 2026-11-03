"""Model builder protocol, registry, and construction entry point."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel
from torch import nn

from cnn_classifier.config.schema import ModelConfig, TransferConfig
from cnn_classifier.core.errors import ConfigError, RegistryError
from cnn_classifier.core.registry import Registry


@dataclass(frozen=True)
class ModelBuildContext:
    num_classes: int
    in_channels: int
    image_size: tuple[int, int]
    transfer: TransferConfig


class ModelBuilder(Protocol):
    Params: type[BaseModel]

    @staticmethod
    def build(params: Any, context: ModelBuildContext) -> nn.Module:
        """Construct a model from validated parameters and context."""
        ...


MODELS: Registry[type[Any]] = Registry("model")


def register_model(name: str) -> Callable[[type[Any]], type[Any]]:
    """Register a model builder class under ``name``."""
    return MODELS.register(name)


def build_model(
    config: ModelConfig,
    *,
    num_classes: int,
    in_channels: int,
    image_size: tuple[int, int],
) -> nn.Module:
    """Resolve, validate, and construct the configured model."""
    try:
        builder = MODELS.get(config.name)
    except RegistryError as exc:
        raise ConfigError(f"unknown model {config.name!r}") from exc
    params = builder.Params.model_validate(config.params)
    context = ModelBuildContext(
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=image_size,
        transfer=config.transfer,
    )
    model: nn.Module = builder.build(params, context)
    return model
