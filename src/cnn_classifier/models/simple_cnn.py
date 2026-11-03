"""Minimal stacked CNN baseline."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from cnn_classifier.models.base import (
    ModelBuildContext,
    register_model,
)


class SimpleCNNParams(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    channels: tuple[int, ...] = Field(default=(32, 64, 128), min_length=1)
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)
    norm: Literal["batch", "none"] = "batch"


class SimpleCNN(nn.Module):
    def __init__(self, params: SimpleCNNParams, context: ModelBuildContext) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = context.in_channels
        for out_channels in params.channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=params.norm == "none"))
            if params.norm == "batch":
                layers.append(nn.BatchNorm2d(out_channels))
            layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(params.dropout),
            nn.Linear(in_channels, context.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.head(self.features(x))
        return result


@register_model("simple_cnn")
class SimpleCNNBuilder:
    Params = SimpleCNNParams

    @staticmethod
    def build(params: SimpleCNNParams, context: ModelBuildContext) -> nn.Module:
        return SimpleCNN(params, context)
