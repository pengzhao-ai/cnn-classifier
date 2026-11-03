import struct
import zlib

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from cnn_classifier.config.schema import ValueRangeConfig
from cnn_classifier.core.errors import DataError
from cnn_classifier.data.image_io import read_image


def _write_png16_rgb(path, pixels: np.ndarray) -> None:
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].astype(">u2").tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_8bit_rgb_reads_float32_chw(tmp_path):
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(path)
    image = read_image(path, channels="rgb")
    assert image.shape == (3, 4, 5)
    assert image.dtype == torch.float32
    assert float(image.max()) == 0.0


def test_8bit_gray_reads_one_channel(tmp_path):
    path = tmp_path / "gray.png"
    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8)).save(path)
    image = read_image(path, channels="gray")
    assert image.shape == (1, 3, 4)
    assert float(image.max()) == pytest.approx(1.0)


def test_16bit_tiff_preserves_range(tmp_path):
    path = tmp_path / "gray.tiff"
    tifffile.imwrite(path, np.array([[0, 65535]], dtype=np.uint16))
    image = read_image(path, channels="gray")
    assert image.shape == (1, 1, 2)
    assert float(image[0, 0, 1]) == pytest.approx(1.0)


def test_16bit_png_reads_without_uint8_truncation(tmp_path):
    path = tmp_path / "gray16.png"
    Image.fromarray(np.array([[0, 65535]], dtype=np.uint16)).save(path)
    image = read_image(path, channels="gray")
    assert image.dtype == torch.float32
    assert float(image[0, 0, 1]) == pytest.approx(1.0)


def test_16bit_rgb_png_is_rejected_instead_of_truncated(tmp_path):
    path = tmp_path / "rgb16.png"
    _write_png16_rgb(path, np.array([[[0, 32768, 65535]]], dtype=np.uint16))

    with pytest.raises(DataError, match="16-bit RGB PNG"):
        read_image(path, channels="rgb")


def test_16bit_rgb_tiff_preserves_range(tmp_path):
    path = tmp_path / "rgb16.tiff"
    tifffile.imwrite(path, np.array([[[0, 32768, 65535]]], dtype=np.uint16))
    image = read_image(path, channels="rgb")
    assert image.shape == (3, 1, 1)
    assert float(image[2, 0, 0]) == pytest.approx(1.0)
    assert float(image[1, 0, 0]) == pytest.approx(32768 / 65535)


def test_fixed_range_scales_16bit(tmp_path):
    path = tmp_path / "gray.tiff"
    tifffile.imwrite(path, np.array([[1000, 2000]], dtype=np.uint16))
    image = read_image(
        path,
        channels="gray",
        value_range=ValueRangeConfig(mode="fixed", min=1000, max=2000),
    )
    assert image.tolist() == [[[0.0, 1.0]]]


def test_corrupt_image_raises_data_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(DataError, match=r"broken\.png"):
        read_image(path)


def test_missing_image_raises_data_error(tmp_path):
    with pytest.raises(DataError, match=r"missing\.png"):
        read_image(tmp_path / "missing.png")
