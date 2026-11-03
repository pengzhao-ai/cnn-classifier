"""Explicit 8-bit and 16-bit image readers with a documented value range."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import torch
from PIL import Image

from cnn_classifier.config.schema import ValueRangeConfig
from cnn_classifier.core.errors import DataError

ChannelPolicy = Literal["rgb", "gray", "as_is"]
_INTEGER_MAX = {
    np.dtype("uint8"): 255.0,
    np.dtype("uint16"): 65535.0,
    np.dtype("uint32"): float(2**32 - 1),
    np.dtype("int16"): 32767.0,
    np.dtype("int32"): float(2**31 - 1),
}


def _read_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    try:
        if suffix in {".tif", ".tiff"}:
            return np.asarray(tifffile.imread(path))
        with Image.open(path) as image:
            return np.asarray(image)
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read image {path}: {exc}") from exc


def _to_channels(array: np.ndarray, channels: ChannelPolicy) -> np.ndarray:
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise DataError(f"image must be 2-D or 3-D; got shape {array.shape}")
    if channels == "as_is":
        return array
    if channels == "gray":
        if array.shape[2] == 1:
            return array
        if array.dtype == np.uint8:
            return np.asarray(Image.fromarray(array).convert("L"), dtype=np.uint8)[:, :, None]
        result: np.ndarray = array.astype(np.float64).mean(axis=2, keepdims=True)
        return result
    if array.shape[2] == 1:
        return np.repeat(array, 3, axis=2)
    if array.shape[2] < 3:
        raise DataError(f"RGB image needs at least 3 channels; got {array.shape[2]}")
    return array[:, :, :3]


def _scale(array: np.ndarray, value_range: ValueRangeConfig | None) -> np.ndarray:
    original_dtype = array.dtype
    data = array.astype(np.float64)
    if value_range is not None and value_range.mode == "fixed":
        if value_range.min is None or value_range.max is None:
            raise DataError("fixed value_range requires min and max")
        low, high = float(value_range.min), float(value_range.max)
        if low >= high:
            raise DataError("fixed value_range min must be smaller than max")
        return np.clip((data - low) / (high - low), 0.0, 1.0)
    if np.issubdtype(original_dtype, np.integer):
        maximum = _INTEGER_MAX.get(original_dtype)
        if maximum is None:
            maximum = float(np.iinfo(original_dtype).max)
        return data / maximum
    return np.clip(data, 0.0, 1.0)


def read_image(
    path: Path,
    channels: ChannelPolicy = "rgb",
    value_range: ValueRangeConfig | None = None,
) -> torch.Tensor:
    """Read an image as ``float32 [C,H,W]`` in ``[0,1]``.

    The reader never silently converts a 16-bit source to 8 bits. TIFF files use
    ``tifffile``; other formats use Pillow, including 16-bit PNG mode ``I;16``.
    """
    if not path.is_file():
        raise DataError(f"image does not exist: {path}")
    array = _read_array(path)
    array = _to_channels(array, channels)
    scaled = _scale(array, value_range)
    tensor = torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1))).to(torch.float32)
    if tensor.ndim != 3:
        raise DataError(f"unexpected image shape after conversion: {tuple(tensor.shape)}")
    return tensor
