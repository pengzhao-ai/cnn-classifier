"""Torchvision model builder with transfer-learning controls."""

from __future__ import annotations

from typing import Literal

import torchvision
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from cnn_classifier.core.errors import ConfigError
from cnn_classifier.models.base import ModelBuildContext, register_model
from cnn_classifier.models.transfer import (
    apply_stem_adaptation,
    apply_transfer,
    replace_classifier,
)


class TorchvisionParams(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: str = "resnet18"
    stem_adaptation: Literal["none", "keep_7x7_stride1", "adapt_3x3"] = "none"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


@register_model("torchvision")
class TorchvisionBuilder:
    Params = TorchvisionParams

    @staticmethod
    def build(params: TorchvisionParams, context: ModelBuildContext) -> nn.Module:
        try:
            model = torchvision.models.get_model(
                params.model,
                weights=context.transfer.weights,
            )
        except (RuntimeError, ValueError) as exc:
            raise ConfigError(f"cannot build torchvision model {params.model!r}: {exc}") from exc
        if not isinstance(model, nn.Module):
            raise ConfigError(f"torchvision model {params.model!r} did not return an nn.Module")
        replace_classifier(model, context.num_classes, dropout=params.dropout)
        apply_stem_adaptation(model, params.stem_adaptation, context.in_channels)
        apply_transfer(model, context.transfer, stem_adaptation=params.stem_adaptation)
        return model
