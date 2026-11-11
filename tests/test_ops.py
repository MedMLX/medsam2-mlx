"""Small network-operation parity against independent CPU Torch behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest

from medsam2_mlx._ops import Conv2d, fill_small_holes_fp32, interpolate_bilinear
from medsam2_mlx._runtime import mx
from medsam2_mlx.mlx_model import resize_mask
from medsam2_mlx.video_predictor import fill_small_holes
from tests.reference_types import FloatArray, TorchRuntime

if TYPE_CHECKING:
    from torch import Tensor


class TorchConvolution(Protocol):
    def conv2d(self, input: Tensor, weight: Tensor, bias: Tensor, *, padding: int) -> Tensor:
        """Apply the Torch oracle convolution with the requested padding."""
        ...


@pytest.mark.parametrize("padding", [0, 1])
def test_pointwise_convolution_padding_matches_torch(padding: int) -> None:
    """Compare pointwise convolution with and without padding to Torch."""
    torch = cast(TorchRuntime, pytest.importorskip("torch"))
    functional = cast(TorchConvolution, torch.nn.functional)
    image = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    weights = np.array([[[[2]], [[-1]]], [[[1]], [[3]]]], dtype=np.float32)
    bias = np.array([0.5, -0.5], dtype=np.float32)
    expected = functional.conv2d(
        torch.from_numpy(image), torch.from_numpy(weights), torch.from_numpy(bias), padding=padding
    ).numpy()
    convolution = Conv2d(2, 2, 1, padding=padding)
    convolution.weight = mx.array(weights)
    convolution.bias = mx.array(bias)
    actual = convolution(mx.array(image))
    mx.eval(actual)
    np.testing.assert_allclose(cast(FloatArray, np.asarray(actual)), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("shape", "target", "dtype"),
    [
        ((2, 3, 13, 17), (31, 29), np.float32),
        ((2, 3, 13, 17), (5, 8), np.float32),
        ((1, 2, 1, 7), (11, 3), np.float32),
        ((1, 2, 9, 1), (3, 15), np.float32),
        ((1, 2, 1, 1), (7, 5), np.float32),
        ((1, 2, 9, 7), (1, 1), np.float32),
        ((1, 2, 9, 7), (9, 7), np.float32),
        ((2, 3, 4, 5, 7), (11, 3), np.float32),
        ((13, 17), (31, 29), np.float32),
        # Low-precision mask values retain the existing FP32 coefficient promotion.
        ((1, 2, 7, 9), (17, 11), np.float16),
    ],
)
def test_bilinear_geometry_and_strided_planes_match_torch(
    shape: tuple[int, ...], target: tuple[int, int], dtype: type[np.floating]
) -> None:
    """Cover half-pixel boundaries, singleton axes, FP16 input and independently resized planes."""
    torch = cast(TorchRuntime, pytest.importorskip("torch"))
    image = np.random.default_rng(17).normal(size=shape).astype(dtype)
    source = torch.nn.functional.interpolate(
        torch.from_numpy(image.astype(np.float32).reshape(-1, 1, *shape[-2:])),
        size=target, mode="bilinear", align_corners=False, antialias=False,
    ).numpy().reshape(*shape[:-2], *target)
    # The model hands interpolation views as well as dense arrays. Transpose a
    # materialized reverse-layout copy so the kernel must handle a strided input.
    axes = (*range(len(shape) - 2), len(shape) - 1, len(shape) - 2)
    value = mx.transpose(mx.array(image.swapaxes(-1, -2).copy()), axes)
    actual = interpolate_bilinear(value, target)
    mx.eval(actual)
    assert actual.dtype == mx.float32
    np.testing.assert_allclose(
        cast(FloatArray, np.asarray(actual)), source, atol=3e-6, rtol=2e-6,
    )


@pytest.mark.parametrize("shape,target", [((1, 1, 13, 17), (31, 29)), ((1, 1, 64, 64), (16, 16))])
def test_mask_antialias_resize_matches_torch(
    shape: tuple[int, ...], target: tuple[int, int]
) -> None:
    """Compare antialiased mask resizing with the Torch reference."""
    torch = cast(TorchRuntime, pytest.importorskip("torch"))
    image = np.random.default_rng(17).normal(size=shape).astype(np.float32)
    source = torch.nn.functional.interpolate(
        torch.from_numpy(image), size=target, mode="bilinear", align_corners=False, antialias=True
    ).numpy()
    result = resize_mask(mx.array(image), target, antialias=True)
    mx.eval(result)
    np.testing.assert_allclose(cast(FloatArray, np.asarray(result)), source, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("max_area", [1, 8, 16])
def test_metal_hole_fill_matches_host_bits(max_area: int) -> None:
    """Preserve 8-connectivity, area cutoffs and IEEE values on strided planes."""
    scores = np.ones((3, 2, 19, 23), dtype=np.float32)
    # A diagonal component connects eight pixels; the adjacent plane has nine.
    for size, plane in [(8, 0), (9, 1)]:
        for index in range(size):
            scores[0, plane, index, index] = -1
    scores[1] = np.random.default_rng(23).choice(
        np.array([-1, 0, 1], dtype=np.float32), size=(2, 19, 23)
    )
    # Build the bits directly so the host never flushes a subnormal before upload.
    bits = np.array([
        0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000,
        0x7fc00001, 0xffc00001,
    ], dtype=np.uint32)
    scores[2, 0, 0, :len(bits)] = bits.view(np.float32)
    scores[2, 1, :, :] = -1
    expected = fill_small_holes(scores, max_area)
    device = mx.transpose(mx.array(scores.swapaxes(-1, -2).copy()), (0, 1, 3, 2))
    actual = fill_small_holes_fp32(device, max_area)
    mx.eval(actual)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), expected.view(np.uint32)
    )
