"""Configurable residual CNN with an explicit stem and stage description."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from cnn_classifier.models.base import ModelBuildContext, register_model


class StemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    channels: int = Field(gt=0)
    kernel_size: int = Field(default=3, ge=1, le=7)
    stride: int = Field(default=1, ge=1)
    pool: bool = False


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    channels: int = Field(gt=0)
    conv: bool = True
    blocks: int = Field(default=1, ge=1)
    downsample: Literal["none", "maxpool", "stride"] = "none"


class ResidualCNNParams(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    stem: StemConfig
    stages: tuple[StageConfig, ...] = Field(min_length=1)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    norm: Literal["batch", "none"] = "batch"
    residual: bool = True


def _norm_layer(channels: int, norm: str) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm: str, residual: bool) -> None:
        super().__init__()
        bias = norm == "none"
        self.residual = residual
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=bias)
        self.norm1 = _norm_layer(channels, norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=bias)
        self.norm2 = _norm_layer(channels, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.residual:
            out = out + x
        return torch.relu(out)


class ResidualCNN(nn.Module):
    def __init__(self, params: ResidualCNNParams, context: ModelBuildContext) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        stem = params.stem
        layers.append(
            nn.Conv2d(
                context.in_channels,
                stem.channels,
                stem.kernel_size,
                stem.stride,
                stem.kernel_size // 2,
                bias=params.norm == "none",
            )
        )
        layers.append(_norm_layer(stem.channels, params.norm))
        layers.append(nn.ReLU(inplace=True))
        if stem.pool:
            layers.append(nn.MaxPool2d(2))
        stage_spans: list[tuple[str, int, int, bool]] = [("stem", 0, len(layers), False)]
        current = stem.channels
        for stage_index, stage in enumerate(params.stages, start=1):
            start = len(layers)
            if stage.conv:
                layers.append(
                    nn.Conv2d(current, stage.channels, 3, 1, 1, bias=params.norm == "none")
                )
                layers.append(_norm_layer(stage.channels, params.norm))
                layers.append(nn.ReLU(inplace=True))
                current = stage.channels
            elif stage.channels != current:
                raise ValueError("a stage without conv must keep the channel count")
            for _ in range(stage.blocks):
                layers.append(ResidualBlock(current, params.norm, params.residual))
            if stage.downsample == "maxpool":
                layers.append(nn.MaxPool2d(2))
            elif stage.downsample == "stride":
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(current, current, 3, 2, 1, bias=params.norm == "none"),
                        _norm_layer(current, params.norm),
                        nn.ReLU(inplace=True),
                    )
                )
            stage_spans.append(
                (
                    f"stage {stage_index}",
                    start,
                    len(layers),
                    params.residual and stage.blocks > 0,
                )
            )
        self.stage_spans = tuple(stage_spans)
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(params.dropout),
            nn.Linear(current, context.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.head(self.features(x))
        return result


@register_model("residual_cnn")
class ResidualCNNBuilder:
    Params = ResidualCNNParams

    @staticmethod
    def build(params: ResidualCNNParams, context: ModelBuildContext) -> nn.Module:
        return ResidualCNN(params, context)
