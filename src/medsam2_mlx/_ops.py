# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source files: sam2/modeling/sam2_utils.py, sam2/modeling/position_encoding.py,
# sam2/modeling/backbones/utils.py, plus PyTorch interpolation semantics.
"""Shared MLX layers and PyTorch-compatible resampling used by the MedSAM2 graph.

Parameters keep PyTorch names, shapes, and layouts. Ordinary convolutions hold
``(out, in / groups, kh, kw)`` weights and transpose them at call time;
transposed convolutions use ``(in, out, kh, kw)``.
Spatial tensors that cross module boundaries use PyTorch's NCHW layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from medsam2_mlx._types import FloatArray, HostArray

if TYPE_CHECKING:
    from mlx.core import array

import math
from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np
from medmlx_core.typing import MetalKernel

from medsam2_mlx._runtime import mx, nn


def gelu(x: array) -> array:
    """Exact (erf) GELU, as ``torch.nn.GELU()``."""
    return nn.gelu(x)


def relu(x: array) -> array:
    """Rectified linear unit."""
    return mx.maximum(x, 0)


class Conv2d(nn.Module):
    """``torch.nn.Conv2d`` on NCHW inputs, holding the PyTorch OIHW weight."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        """Initialize OIHW weights and bias for NCHW convolution."""
        super().__init__()
        scale = 1.0 / math.sqrt(in_channels // groups * kernel_size * kernel_size)
        self.weight = mx.random.uniform(
            -scale, scale, (out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = mx.random.uniform(-scale, scale, (out_channels,))
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def nhwc(self, x: array) -> array:
        """Apply to an NHWC input and return NHWC."""
        if (self.weight.shape[2:] == (1, 1) and self.groups == 1
                and self.stride == 1 and self.padding == 0):
            weight = mx.reshape(self.weight, (self.weight.shape[0], -1))
            return x @ weight.T + self.bias
        y = mx.conv2d(
            x,
            mx.transpose(self.weight, (0, 2, 3, 1)),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        return y + self.bias

    def __call__(self, x: array) -> array:
        """Transpose NCHW input for MLX convolution and restore NCHW output."""
        return mx.transpose(self.nhwc(mx.transpose(x, (0, 2, 3, 1))), (0, 3, 1, 2))


class ConvTranspose2x2(nn.Module):
    """``torch.nn.ConvTranspose2d(kernel_size=2, stride=2)`` on NCHW inputs.

    With stride equal to the kernel, every output pixel receives exactly one input
    pixel, so the operation is a per-pixel matrix product followed by a pixel shuffle.
    The weight keeps PyTorch's ``(in, out, 2, 2)`` layout.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize transposed-convolution weights in PyTorch layout."""
        super().__init__()
        scale = 1.0 / math.sqrt(out_channels * 4)
        self.weight = mx.random.uniform(-scale, scale, (in_channels, out_channels, 2, 2))
        self.bias = mx.random.uniform(-scale, scale, (out_channels,))

    def __call__(self, x: array) -> array:
        """Double each spatial axis using a projection and pixel shuffle."""
        b, c, h, w = x.shape
        out = self.weight.shape[1]
        y = mx.reshape(mx.transpose(x, (0, 2, 3, 1)), (b * h * w, c)) @ mx.reshape(
            self.weight, (c, out * 4)
        )
        y = mx.transpose(mx.reshape(y, (b, h, w, out, 2, 2)), (0, 3, 1, 4, 2, 5))
        return mx.reshape(y, (b, out, 2 * h, 2 * w)) + self.bias[None, :, None, None]


class LayerNorm2d(nn.Module):
    """Channel LayerNorm on NCHW tensors (SAM's ``LayerNorm2d``)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Initialize per-channel scale, bias and normalization epsilon."""
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x: array) -> array:
        """Normalize across channels independently at every NCHW pixel."""
        u = mx.mean(x, axis=1, keepdims=True)
        s = mx.mean(mx.square(x - u), axis=1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LayerNorm(nn.Module):
    """``torch.nn.LayerNorm`` over the last axis."""

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        """Initialize last-axis scale, bias and normalization epsilon."""
        super().__init__()
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: array) -> array:
        """Apply MLX layer normalization over the last axis."""
        return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)


class Activation(nn.Module):
    """Parameter-free activation slot, keeping ``nn.Sequential`` indices aligned."""

    def __init__(self, fn: Callable[[array], array]) -> None:
        """Store the activation function in a parameter-free module slot."""
        super().__init__()
        self._fn = fn

    def __call__(self, x: array) -> array:
        """Apply the stored activation to its input."""
        return self._fn(x)


class Embedding(nn.Module):
    """``torch.nn.Embedding`` used only through its ``weight`` table."""

    def __init__(self, num: int, dims: int) -> None:
        """Initialize a normally distributed embedding weight table."""
        super().__init__()
        self.weight = mx.random.normal((num, dims))


class MLP(nn.Module):
    """SAM's MLP: ``num_layers`` linears, activation between them."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: Callable[[array], array] = relu,
        sigmoid_output: bool = False,
    ) -> None:
        """Build the requested linear layers, activation and optional output sigmoid."""
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = [
            nn.Linear(n, k)
            for n, k in zip([input_dim, *hidden], [*hidden, output_dim], strict=True)
        ]
        self._activation = activation
        self.sigmoid_output = sigmoid_output

    def __call__(self, x: array) -> array:
        """Apply each linear layer and intermediate activation to the input."""
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            x = layer(x) if index == last else self._activation(layer(x))
        return mx.sigmoid(x) if self.sigmoid_output else x


def sequential(layers: Sequence[Callable[[array], array]], x: array) -> array:
    """Apply ``layers`` in order (``nn.Sequential`` for module lists held as attributes)."""
    for layer in layers:
        x = layer(x)
    return x


# --- Positional encodings -----------------------------------------------------------


@lru_cache(maxsize=32)
def _sine_position(num_pos_feats: int, height: int, width: int) -> HostArray:
    """``PositionEmbeddingSine(num_pos_feats, normalize=True)`` for one image, (C, H, W)."""
    half = num_pos_feats // 2
    scale = np.float32(2 * math.pi)
    eps = np.float32(1e-6)
    y = np.arange(1, height + 1, dtype=np.float32)
    x = np.arange(1, width + 1, dtype=np.float32)
    y = y / (y[-1] + eps) * scale
    x = x / (x[-1] + eps) * scale
    dim_t = np.arange(half, dtype=np.float32)
    dim_t = np.float32(10000) ** (2 * (dim_t // 2) / np.float32(half))

    def encode(values: FloatArray) -> FloatArray:
        """Interleave sine and cosine features for one spatial coordinate axis."""
        pos = values[:, None] / dim_t
        return np.stack((np.sin(pos[:, 0::2]), np.cos(pos[:, 1::2])), axis=2).reshape(
            len(values), -1
        )

    pos_y = np.broadcast_to(encode(y)[:, None, :], (height, width, half))
    pos_x = np.broadcast_to(encode(x)[None, :, :], (height, width, half))
    return np.ascontiguousarray(np.concatenate([pos_y, pos_x], axis=2).transpose(2, 0, 1))


def sine_position_encoding(num_pos_feats: int, batch: int, height: int, width: int) -> array:
    """Sine position encoding broadcast to (batch, num_pos_feats, height, width).

    The encoding is the same for every batch element, so it is computed once per size.
    """
    pos = mx.array(_sine_position(num_pos_feats, height, width))
    return mx.broadcast_to(pos[None], (batch, *pos.shape))


def get_1d_sine_pe(pos_inds: array, dim: int, temperature: float = 10000) -> array:
    """Encode positions ``pos_inds`` (...,) as (..., 2 * (dim // 2)).

    The first ``dim // 2`` channels are sines and the rest cosines (not interleaved);
    an odd ``dim`` therefore produces one fewer channel.
    """
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = np.float32(temperature) ** (2 * (dim_t // 2) / np.float32(pe_dim))
    pos_embed = pos_inds[..., None] / mx.array(dim_t)
    return mx.concatenate([mx.sin(pos_embed), mx.cos(pos_embed)], axis=-1)


@lru_cache(maxsize=16)
def axial_rope(dim: int, end_x: int, end_y: int, theta: float) -> tuple[HostArray, HostArray]:
    """Return axial cosine/sine tables, x frequencies then y.

    Each table has shape (``end_x * end_y``, ``2 * (dim // 4)``), equal to
    ``(L, dim // 2)`` when ``dim`` is divisible by four. Spatial x varies fastest.
    """
    exponents = np.arange(0, dim, 4, dtype=np.float32)[: dim // 4] / np.float32(dim)
    freqs = np.float32(1.0) / (np.float32(theta) ** exponents)
    t = np.arange(end_x * end_y, dtype=np.float32)
    t_x = t % np.float32(end_x)
    t_y = np.floor(t / np.float32(end_x))
    angles = np.concatenate([np.outer(t_x, freqs), np.outer(t_y, freqs)], axis=-1)
    angles = angles.astype(np.float32)
    return np.cos(angles), np.sin(angles)


def apply_rope(x: array, cos: array, sin: array) -> array:
    """Rotate adjacent (real, imaginary) channel pairs; ``x`` is (..., L, D)."""
    pairs = mx.reshape(x, (*x.shape[:-1], -1, 2))
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = mx.stack([real * cos - imag * sin, real * sin + imag * cos], axis=-1)
    return mx.reshape(rotated, x.shape)


# --- PyTorch-compatible interpolation -------------------------------------------------


@lru_cache(maxsize=1)
def _bilinear_kernel() -> MetalKernel:
    """Build the four-sample FP32 resize kernel; MLX makes strided inputs contiguous."""
    return mx.fast.metal_kernel(
        name="medsam2_bilinear_fp32",
        input_names=["inp"],
        output_names=["out"],
        # Keep the half-pixel multiply and subtraction separately rounded, as in
        # the coefficient construction below; contraction can move sample coordinates.
        header="#pragma clang fp contract(off)\n",
        compile_options={"math_mode": "safe"},
        source="""
            uint i = thread_position_in_grid.x;
            if (i >= PLANES * OH * OW) return;
            uint ox = i % OW;
            uint oy = (i / OW) % OH;
            uint plane = i / (OW * OH);
            float sy = metal::max(
                (float(oy) + 0.5f) * (float(H) / float(OH)) - 0.5f, 0.0f);
            float sx = metal::max(
                (float(ox) + 0.5f) * (float(W) / float(OW)) - 0.5f, 0.0f);
            uint y0 = metal::min(uint(metal::floor(sy)), uint(H - 1));
            uint x0 = metal::min(uint(metal::floor(sx)), uint(W - 1));
            uint y1 = metal::min(y0 + 1, uint(H - 1));
            uint x1 = metal::min(x0 + 1, uint(W - 1));
            float dy = sy - float(y0), dx = sx - float(x0);
            size_t base = size_t(plane) * H * W;
            float left = (1.0f - dy) * inp[base + y0 * W + x0]
                + dy * inp[base + y1 * W + x0];
            float right = (1.0f - dy) * inp[base + y0 * W + x1]
                + dy * inp[base + y1 * W + x1];
            out[i] = (1.0f - dx) * left + dx * right;
        """,
    )


@lru_cache(maxsize=64)
def _bilinear_matrix(source: int, target: int, antialias: bool) -> HostArray:
    """Row-stochastic (target, source) weights of ``F.interpolate(mode="bilinear")``.

    ``align_corners=False``. With ``antialias`` and downsampling, the triangle filter
    is widened by the scale factor, as in PyTorch's antialiased kernels.
    """
    scale = np.float32(source / target)
    weights = np.zeros((target, source), dtype=np.float32)
    if antialias and scale > 1:
        centers = (np.arange(target, dtype=np.float32) + np.float32(0.5)) * scale
        support = scale
        for index, center in enumerate(centers):
            low = max(int(center - support + np.float32(0.5)), 0)
            high = min(int(center + support + np.float32(0.5)), source)
            taps = np.arange(low, high, dtype=np.float32)
            distance = np.abs((taps - center + np.float32(0.5)) / scale)
            tap_weights = np.where(distance < 1, 1 - distance, 0).astype(np.float32)
            total = tap_weights.sum(dtype=np.float32)
            if total != 0:
                tap_weights = tap_weights / total
            weights[index, low:high] = tap_weights
        return weights
    src = (np.arange(target, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    src = np.maximum(src, np.float32(0))
    i0 = np.minimum(np.floor(src).astype(np.int64), source - 1)
    i1 = np.minimum(i0 + 1, source - 1)
    l1 = (src - i0.astype(np.float32)).astype(np.float32)
    l0 = (np.float32(1) - l1).astype(np.float32)
    rows = np.arange(target)
    np.add.at(weights, (rows, i0), l0)
    np.add.at(weights, (rows, i1), l1)
    return weights


def interpolate_bilinear(x: array, size: tuple[int, int], *, antialias: bool = False) -> array:
    """``F.interpolate(x, size, mode="bilinear", align_corners=False)`` on NCHW.

    ``size`` is (height, width); any leading shape is preserved. Non-antialiased
    FP32 values use four samples per output on Metal, avoiding dense products
    with mostly zero interpolation weights. Other dtypes and antialiased resizing
    retain the matrix path. An unchanged size returns ``x`` itself.
    """
    height, width = x.shape[-2:]
    if (height, width) == tuple(size):
        return x
    out_height, out_width = size
    shape = (*x.shape[:-2], out_height, out_width)
    output_size = math.prod(shape)
    # Metal's launch index and integer template products use 32-bit arithmetic.
    # Preserve the matrix implementation for larger or empty helper inputs.
    if (
        x.dtype == mx.float32 and not antialias and x.size
        and min(height, width, *size) > 0
        and max(x.size, output_size) <= 2**31 - 1
    ):
        return _bilinear_kernel()(
            inputs=[x],
            template=[
                ("PLANES", math.prod(x.shape[:-2])),
                ("H", height), ("W", width), ("OH", out_height), ("OW", out_width),
            ],
            grid=(output_size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[shape],
            output_dtypes=[mx.float32],
        )[0]
    rows = mx.array(_bilinear_matrix(height, size[0], antialias))
    cols = mx.array(_bilinear_matrix(width, size[1], antialias))
    return rows @ x @ cols.T


@lru_cache(maxsize=8)
def _bicubic_matrix(source: int, target: int) -> HostArray:
    """(target, source) weights of ``F.interpolate(mode="bicubic", align_corners=False)``."""
    a = -0.75
    scale = source / target
    weights = np.zeros((target, source), dtype=np.float64)
    for index in range(target):
        real = scale * (index + 0.5) - 0.5
        base = math.floor(real)
        t = real - base

        def near(x: float) -> float:
            """Evaluate the inner piece of the cubic interpolation kernel."""
            return ((a + 2) * x - (a + 3)) * x * x + 1

        def far(x: float) -> float:
            """Evaluate the outer piece of the cubic interpolation kernel."""
            return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

        for offset, weight in zip(
            (-1, 0, 1, 2), (far(t + 1), near(t), near(1 - t), far(2 - t)), strict=True
        ):
            weights[index, min(max(base + offset, 0), source - 1)] += weight
    return weights.astype(np.float32)


def interpolate_bicubic(x: array, size: tuple[int, int]) -> array:
    """``F.interpolate(x, size, mode="bicubic", align_corners=False)`` on NCHW."""
    rows = mx.array(_bicubic_matrix(x.shape[-2], size[0]))
    cols = mx.array(_bicubic_matrix(x.shape[-1], size[1]))
    return rows @ x @ cols.T


def upsample_nearest2x(x: array) -> array:
    """``F.interpolate(x, scale_factor=2.0, mode="nearest")`` on NCHW."""
    b, c, h, w = x.shape
    x = mx.broadcast_to(x[:, :, :, None, :, None], (b, c, h, 2, w, 2))
    return mx.reshape(x, (b, c, 2 * h, 2 * w))
