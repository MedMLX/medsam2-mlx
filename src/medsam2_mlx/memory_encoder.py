# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source file: sam2/modeling/memory_encoder.py.
"""Memory encoder: fuse a mask with the frame's features into a spatial memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array

import math
from collections.abc import Callable

from medsam2_mlx._ops import (
    Activation,
    Conv2d,
    LayerNorm2d,
    gelu,
    sequential,
    sine_position_encoding,
)
from medsam2_mlx._runtime import mx, nn
from medsam2_mlx._types import MemoryEncoding
from medsam2_mlx.config import MemoryEncoderConfig


class MaskDownSampler(nn.Module):
    """Strided convolutions reducing a mask by ``total_stride`` (16: 512 -> 32 pixels)."""

    def __init__(
        self, embed_dim: int, kernel_size: int, stride: int, padding: int, total_stride: int
    ) -> None:
        """Build strided mask convolutions whose combined stride matches total_stride."""
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        if stride**num_layers != total_stride:
            raise ValueError("Mask stride does not divide the total stride")
        layers: list[Callable[[array], array]] = []
        mask_in_chans = mask_out_chans = 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * stride**2
            layers.append(Conv2d(mask_in_chans, mask_out_chans, kernel_size, stride, padding))
            layers.append(LayerNorm2d(mask_out_chans))
            layers.append(Activation(gelu))
            mask_in_chans = mask_out_chans
        layers.append(Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.encoder = layers

    def __call__(self, x: array) -> array:
        """Downsample the input mask through the configured convolution stack."""
        return sequential(self.encoder, x)


class CXBlock(nn.Module):
    """ConvNeXt block: depthwise conv, LayerNorm2d, pointwise MLP, layer scale."""

    def __init__(self, dim: int, kernel_size: int, padding: int) -> None:
        """Build a depthwise ConvNeXt residual with pointwise projections."""
        super().__init__()
        self.dwconv = Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = 1e-6 * mx.ones((dim,))

    def __call__(self, x: array) -> array:
        """Apply depthwise convolution and a scaled channel MLP as a residual."""
        y = mx.transpose(self.norm(self.dwconv(x)), (0, 2, 3, 1))
        y = self.gamma * self.pwconv2(gelu(self.pwconv1(y)))
        return x + mx.transpose(y, (0, 3, 1, 2))


class Fuser(nn.Module):
    """Stack of ConvNeXt blocks fusing the image features with the downsampled mask."""

    def __init__(self, dim: int, config: MemoryEncoderConfig) -> None:
        """Create the configured stack of mask and image fusion blocks."""
        super().__init__()
        self.layers = [
            CXBlock(dim, config.fuser_kernel_size, config.fuser_padding)
            for _ in range(config.fuser_num_layers)
        ]

    def __call__(self, x: array) -> array:
        """Apply the ConvNeXt fusion blocks in sequence."""
        return sequential(self.layers, x)


class MemoryEncoder(nn.Module):
    """Encode a mask and the frame's image features into a spatial memory."""

    def __init__(self, in_dim: int, out_dim: int, config: MemoryEncoderConfig) -> None:
        """Build mask downsampling, image projection and memory fusion modules."""
        super().__init__()
        self.mask_downsampler = MaskDownSampler(
            in_dim,
            config.mask_kernel_size,
            config.mask_stride,
            config.mask_padding,
            config.mask_total_stride,
        )
        self.pix_feat_proj = Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(in_dim, config)
        self.out_proj = Conv2d(in_dim, out_dim, kernel_size=1)
        self.out_dim = out_dim

    def __call__(
        self, pix_feat: array, masks: array, skip_mask_sigmoid: bool = False
    ) -> MemoryEncoding:
        """Encode image features and a mask into memory and position encodings.

        ``masks`` are logits unless ``skip_mask_sigmoid``, in which case they are used as
        given (the caller already applied the sigmoid or binarization and the scale and
        bias). With the default graph the inputs have shapes (B, 256, 32, 32) and
        (B, 1, 512, 512). Returns ``vision_features`` (B, out_dim, 32, 32) for that
        graph and a one-element ``vision_pos_enc`` list with matching sine encoding.
        """
        if not skip_mask_sigmoid:
            masks = mx.sigmoid(masks)
        masks = self.mask_downsampler(masks)
        x = self.pix_feat_proj(pix_feat) + masks
        x = self.out_proj(self.fuser(x))
        b, _, h, w = x.shape
        return {
            "vision_features": x,
            "vision_pos_enc": [sine_position_encoding(self.out_dim, b, h, w)],
        }
