# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source files: sam2/modeling/backbones/hieradet.py, sam2/modeling/backbones/utils.py,
# sam2/modeling/backbones/image_encoder.py.
"""Hiera trunk and FPN neck of the SAM 2.1 image encoder, in MLX."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array

from collections.abc import Sequence
from typing import cast

from medsam2_mlx._ops import (
    MLP,
    Conv2d,
    LayerNorm,
    gelu,
    interpolate_bicubic,
    sine_position_encoding,
    upsample_nearest2x,
)
from medsam2_mlx._runtime import mx, nn
from medsam2_mlx._types import ImageFeatures
from medsam2_mlx.config import HieraConfig, ImageEncoderConfig


def _max_pool2x2(x: array) -> array:
    """``nn.MaxPool2d(2, 2)`` on (B, H, W, C), floor mode."""
    b, h, w, c = x.shape
    x = x[:, : h // 2 * 2, : w // 2 * 2]
    return mx.max(mx.reshape(x, (b, h // 2, 2, w // 2, 2, c)), axis=(2, 4))


def window_partition(x: array, window_size: int) -> tuple[array, tuple[int, int]]:
    """Zero-pad (B, H, W, C) to whole windows and split into (B * nW, ws, ws, C)."""
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
    hp, wp = h + pad_h, w + pad_w
    x = mx.reshape(x, (b, hp // window_size, window_size, wp // window_size, window_size, c))
    windows = mx.reshape(mx.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, window_size, window_size, c))
    return windows, (hp, wp)


def window_unpartition(
    windows: array, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]
) -> array:
    """Inverse of ``window_partition``: merge windows and crop the padding back to ``hw``."""
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = mx.reshape(windows, (b, hp // window_size, wp // window_size, window_size, window_size, -1))
    x = mx.reshape(mx.transpose(x, (0, 1, 3, 2, 4, 5)), (b, hp, wp, -1))
    return x[:, :h, :w, :] if hp > h or wp > w else x


class MultiScaleAttention(nn.Module):
    """Hiera attention over (B, H, W, C) tokens; with ``q_pool`` the queries are 2x2 max-pooled."""

    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: bool) -> None:
        """Create query, key and value projections and optional pooled queries."""
        super().__init__()
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def __call__(self, x: array) -> array:
        """Attend NHWC tokens, pooling queries when the stage downsamples."""
        b, h, w, _ = x.shape
        qkv = mx.reshape(self.qkv(x), (b, h * w, 3, self.num_heads, -1))
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.q_pool:
            q = _max_pool2x2(mx.reshape(q, (b, h, w, -1)))
            h, w = q.shape[1:3]
            q = mx.reshape(q, (b, h * w, self.num_heads, -1))
        scale = q.shape[-1] ** -0.5
        out = mx.fast.scaled_dot_product_attention(
            mx.transpose(q, (0, 2, 1, 3)),
            mx.transpose(k, (0, 2, 1, 3)),
            mx.transpose(v, (0, 2, 1, 3)),
            scale=scale,
        )
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (b, h, w, -1))
        return self.proj(out)


class MultiScaleBlock(nn.Module):
    """Hiera block on (B, H, W, C): windowed (or global if ``window_size`` is 0) attention.

    When ``dim != dim_out`` the shortcut is projected (and 2x2 max-pooled if ``q_stride``
    is set); with ``q_stride`` the queries are pooled too, halving the spatial size.
    Query pooling requires a channel-changing stage and a window of at least two
    pixels. The pooled windows must also match the pooled shortcut's spatial grid.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_stride: tuple[int, int] | None,
        window_size: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Build windowed attention, projected shortcut and feed-forward residual."""
        super().__init__()
        if q_stride is not None and tuple(q_stride) != (2, 2):
            raise ValueError("Only 2x2 Hiera query pooling is supported")
        if q_stride is not None and (dim == dim_out or window_size < 2):
            raise ValueError(
                "Hiera query pooling requires changing channels and a window of at least 2"
            )
        self.dim = dim
        self.dim_out = dim_out
        self.window_size = window_size
        self.q_stride = q_stride
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, q_pool=q_stride is not None)
        self.norm2 = LayerNorm(dim_out, eps=1e-6)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, 2, activation=gelu)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def __call__(self, x: array) -> array:
        """Run Hiera attention and MLP residuals with validated pooling geometry."""
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = self.proj(x)
            if self.q_stride is not None:
                shortcut = _max_pool2x2(shortcut)
        window_size = self.window_size
        h, w = x.shape[1], x.shape[2]
        if self.q_stride is not None:
            pooled_window = window_size // 2
            source_windows = tuple((size + window_size - 1) // window_size for size in (h, w))
            pooled_windows = tuple(
                (size // 2 + pooled_window - 1) // pooled_window for size in (h, w)
            )
            if source_windows != pooled_windows:
                raise ValueError("Hiera pooled attention windows do not match the shortcut grid")
        pad_hw = (h, w)
        if window_size > 0:
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.q_stride is not None:
            window_size = self.window_size // self.q_stride[0]
            h, w = shortcut.shape[1:3]
            pad_h = (window_size - h % window_size) % window_size
            pad_w = (window_size - w % window_size) % window_size
            pad_hw = (h + pad_h, w + pad_w)
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (h, w))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """7 x 7, stride-4 convolution that turns the image into tokens."""

    def __init__(self, embed_dim: int, in_chans: int = 3) -> None:
        """Build the stride-four image-to-token convolution."""
        super().__init__()
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3)

    def __call__(self, x: array) -> array:
        """NCHW image to (B, H, W, C) tokens."""
        return self.proj.nhwc(mx.transpose(x, (0, 2, 3, 1)))


class Hiera(nn.Module):
    """Hiera trunk.

    With the default configuration and a (B, 3, 512, 512) input, ``__call__`` returns
    the last block of each stage as NCHW arrays, shallow to deep: (B, 96, 128, 128),
    (B, 192, 64, 64),
    (B, 384, 32, 32) and (B, 768, 16, 16).
    """

    def __init__(self, config: HieraConfig) -> None:
        """Construct the patch stem, positional embeddings and configured Hiera stages."""
        super().__init__()
        stages = config.stages
        if len(stages) != len(config.window_spec):
            raise ValueError("Hiera stages and window_spec lengths differ")
        self.window_spec = config.window_spec
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][: config.q_pool]
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        self.patch_embed = PatchEmbed(embed_dim)
        self.pos_embed = mx.zeros((1, embed_dim, *config.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = mx.zeros(
            (1, embed_dim, config.window_spec[0], config.window_spec[0])
        )
        cur_stage = 1
        self.blocks: list[MultiScaleBlock] = []
        for i in range(sum(stages)):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if i in config.global_att_blocks:
                window_size = 0
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * config.dim_mul)
                num_heads = int(num_heads * config.head_mul)
                cur_stage += 1
            self.blocks.append(
                MultiScaleBlock(
                    embed_dim,
                    dim_out,
                    num_heads,
                    q_stride=(2, 2) if i in q_pool_blocks else None,
                    window_size=window_size,
                )
            )
            embed_dim = dim_out
        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]]

    def _get_pos_embed(self, hw: tuple[int, int]) -> array:
        """Add a resized background embedding and a tiled window embedding.

        Patch-grid dimensions should be multiples of the window dimensions; tiling
        uses floor division and does not pad or crop the window embedding.
        """
        pos = interpolate_bicubic(self.pos_embed, hw)
        window = self.pos_embed_window
        reps = (1, 1, hw[0] // window.shape[2], hw[1] // window.shape[3])
        pos = pos + mx.tile(window, reps)
        return mx.transpose(pos, (0, 2, 3, 1))

    def __call__(self, x: array) -> list[array]:
        """Return each stage endpoint as an NCHW feature map, shallow to deep."""
        x = self.patch_embed(x)
        x = x + self._get_pos_embed((x.shape[1], x.shape[2]))
        outputs: list[array] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.stage_ends:
                outputs.append(mx.transpose(x, (0, 3, 1, 2)))
        return outputs


class _NeckLevel(nn.Module):
    """One lateral 1 x 1 convolution (``neck.convs.N.conv`` in the state dict)."""

    def __init__(self, in_channels: int, d_model: int) -> None:
        """Build one lateral projection into the neck feature width."""
        super().__init__()
        self.conv = Conv2d(in_channels, d_model, kernel_size=1)


class FpnNeck(nn.Module):
    """SAM 2 FPN neck with nearest top-down fusion and sine position encodings.

    Takes the trunk outputs shallow to deep and returns ``d_model``-channel features and
    position encodings in the same order. Levels in ``fpn_top_down_levels`` add the
    2x nearest-upsampled next-deeper level.
    """

    def __init__(
        self, d_model: int, backbone_channel_list: Sequence[int], fpn_top_down_levels: Sequence[int]
    ) -> None:
        """Create lateral projections and record the levels that receive top-down fusion."""
        super().__init__()
        self.d_model = d_model
        self.backbone_channel_list = list(backbone_channel_list)
        self.convs = [_NeckLevel(dim, d_model) for dim in backbone_channel_list]
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def __call__(self, xs: list[array]) -> tuple[list[array], list[array]]:
        """Fuse trunk levels and return features with matching sine positions."""
        n = len(self.convs) - 1
        out: list[array | None] = [None] * len(self.convs)
        pos: list[array | None] = [None] * len(self.convs)
        previous = None
        for i in range(n, -1, -1):
            lateral = self.convs[n - i].conv(xs[i])
            if i in self.fpn_top_down_levels and previous is not None:
                previous = lateral + upsample_nearest2x(previous)
            else:
                previous = lateral
            out[i] = previous
            b, _, h, w = previous.shape
            pos[i] = sine_position_encoding(self.d_model, b, h, w)
        return cast("list[array]", out), cast("list[array]", pos)


class ImageEncoder(nn.Module):
    """Hiera trunk plus FPN neck.

    With the default configuration and a (B, 3, 512, 512) input, returns ``backbone_fpn`` and
    ``vision_pos_enc`` as three NCHW levels of 256 channels (128 x 128, 64 x 64,
    32 x 32), and ``vision_features``, the 32 x 32 level. The deepest 16 x 16 level
    contributes through top-down fusion but is dropped (``scalp``).
    """

    def __init__(self, config: ImageEncoderConfig) -> None:
        """Construct compatible Hiera trunk and FPN neck modules."""
        super().__init__()
        self.trunk = Hiera(config.hiera)
        self.neck = FpnNeck(
            config.d_model, config.backbone_channel_list, config.fpn_top_down_levels
        )
        self.scalp = config.scalp
        if self.trunk.channel_list != self.neck.backbone_channel_list:
            raise ValueError("Hiera channels do not match the FPN neck")

    def __call__(self, sample: array) -> ImageFeatures:
        """Encode an image and return the selected FPN levels and positions."""
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        return {"vision_features": features[-1], "vision_pos_enc": pos, "backbone_fpn": features}
