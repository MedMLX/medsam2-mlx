# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source files: sam2/modeling/sam/prompt_encoder.py, sam2/modeling/position_encoding.py.
"""SAM prompt encoder (points, box corners, and dense mask prompts), in MLX."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array

import math

from medsam2_mlx._ops import Activation, Conv2d, Embedding, LayerNorm2d, gelu, sequential
from medsam2_mlx._runtime import mx, nn


class PositionEmbeddingRandom(nn.Module):
    """Random Fourier features of normalized coordinates.

    The Gaussian projection matrix is initialized randomly, then replaced by the
    checkpoint value when weights are loaded.
    """

    def __init__(self, num_pos_feats: int) -> None:
        """Initialize the Gaussian matrix used for coordinate Fourier features."""
        super().__init__()
        self.positional_encoding_gaussian_matrix = mx.random.normal((2, num_pos_feats))

    def _pe_encoding(self, coords: array) -> array:
        """Project normalized coordinates into sine and cosine features."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = coords * (2 * math.pi)
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)

    def __call__(self, size: tuple[int, int]) -> array:
        """Dense (C, H, W) encoding of pixel centers."""
        h, w = size
        y = (mx.arange(1, h + 1, dtype=mx.float32) - 0.5) / h
        x = (mx.arange(1, w + 1, dtype=mx.float32) - 0.5) / w
        grid = mx.stack(
            [mx.broadcast_to(x[None, :], (h, w)), mx.broadcast_to(y[:, None], (h, w))], axis=-1
        )
        return mx.transpose(self._pe_encoding(grid), (2, 0, 1))

    def forward_with_coords(self, coords: array, image_size: tuple[int, int]) -> array:
        """Encode (..., 2) (x, y) pixel coordinates of an image of ``image_size`` (h, w)."""
        coords = mx.stack([coords[..., 0] / image_size[1], coords[..., 1] / image_size[0]], axis=-1)
        return self._pe_encoding(coords.astype(mx.float32))


class PromptEncoder(nn.Module):
    """Embed point/box-corner prompts and dense mask prompts for the mask decoder."""

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
    ) -> None:
        """Create point embeddings, coordinate features and mask downscaling layers."""
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = [Embedding(1, embed_dim) for _ in range(4)]
        self.not_a_point_embed = Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = [
            Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            Activation(gelu),
            Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            Activation(gelu),
            Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        ]
        self.no_mask_embed = Embedding(1, embed_dim)

    def get_dense_pe(self) -> array:
        """Encode the embedding grid as (1, embed_dim, H, W); default H and W are 32."""
        return self.pe_layer(self.image_embedding_size)[None]

    def _embed_points(self, points: array, labels: array, pad: bool) -> array:
        """Embed (B, N, 2) pixel coordinates; ``pad`` appends one label -1 "not a point" token."""
        # Shift pixel indices to pixel centers, as upstream does.
        points = points + 0.5
        if pad:
            points = mx.concatenate([points, mx.zeros((points.shape[0], 1, 2))], axis=1)
            labels = mx.concatenate(
                [labels, -mx.ones((labels.shape[0], 1), dtype=labels.dtype)], axis=1
            )
        embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        labels = labels[..., None]
        embedding = mx.where(labels == -1, self.not_a_point_embed.weight, embedding)
        for label, table in enumerate(self.point_embeddings):
            embedding = mx.where(labels == label, embedding + table.weight, embedding)
        return embedding

    def __call__(
        self, points: tuple[array, array] | None, masks: array | None
    ) -> tuple[array, array]:
        """Sparse (B, N, C) and dense (B, C, h, w) prompt embeddings.

        ``points`` is ``(coords, labels)``: ``coords`` (B, N, 2) are (x, y) in pixels of the
        512 x 512 model input and ``labels`` (B, N) ints with 1 foreground, 0 background,
        2 and 3 for the box top-left and bottom-right corners, and -1 for padding. One
        padding point is appended to the given points. ``masks`` (B, 1, 128, 128) are
        dense logits; without them a learned "no mask" embedding fills the 32 x 32 grid.

        SAM 2 always passes boxes as labeled corner points, so there is no box input.
        """
        if points is not None:
            batch = points[0].shape[0]
        elif masks is not None:
            batch = masks.shape[0]
        else:
            batch = 1
        sparse = mx.zeros((batch, 0, self.embed_dim))
        if points is not None:
            sparse = mx.concatenate([sparse, self._embed_points(*points, pad=True)], axis=1)
        if masks is not None:
            dense = sequential(self.mask_downscaling, masks)
        else:
            dense = mx.broadcast_to(
                mx.reshape(self.no_mask_embed.weight, (1, -1, 1, 1)),
                (batch, self.embed_dim, *self.image_embedding_size),
            )
        return sparse, dense
