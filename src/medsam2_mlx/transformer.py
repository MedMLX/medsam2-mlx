# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source files: sam2/modeling/sam/transformer.py, sam2/modeling/position_encoding.py.
"""Two-way SAM transformer and (rotary) multi-head attention, in MLX."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array

import math

from medsam2_mlx._ops import MLP, LayerNorm, apply_rope, axial_rope, relu
from medsam2_mlx._runtime import mx, nn


class Attention(nn.Module):
    """Multi-head attention with an optional channel downsampling of q/k/v.

    Inputs are (B, N, C) sequences; keys and values may have ``kv_in_dim`` channels.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int | None = None,
    ) -> None:
        """Create query, key, value and output projections for multi-head attention."""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if self.internal_dim % num_heads:
            raise ValueError("num_heads must divide embedding_dim")
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: array) -> array:
        """Reshape batch-first tokens into separate attention heads."""
        b, n, c = x.shape
        return mx.transpose(
            mx.reshape(x, (b, n, self.num_heads, c // self.num_heads)), (0, 2, 1, 3)
        )

    def _recombine_heads(self, x: array) -> array:
        """Combine the attention head and channel axes."""
        b, heads, n, c = x.shape
        return mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (b, n, heads * c))

    def _attend(self, q: array, k: array, v: array) -> array:
        """Apply scaled attention and project the recombined heads."""
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]))
        return self.out_proj(self._recombine_heads(out))

    def __call__(self, q: array, k: array, v: array) -> array:
        """Project and attend batch-first query, key and value sequences."""
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))
        return self._attend(q, k, v)


class RoPEAttention(Attention):
    """Attention with 2D axial rotary position encoding (SAM 2 memory attention).

    With ``rope_k_repeat`` the query's rotary table is tiled along the key sequence,
    once per spatial memory frame (``repeat_freqs_k=True`` upstream). The trailing
    ``num_k_exclude_rope`` keys (object pointers) are not rotated. The query length
    must be a square; without repetition the full query and key lengths must match.
    For repeated spatial grids the unexcluded key length must be a multiple of
    the query length, and the excluded count must fit the key sequence.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
    ) -> None:
        """Configure axial rotary attention and optional repeated key grids."""
        super().__init__(embedding_dim, num_heads, downsample_rate, kv_in_dim)
        self.rope_theta = rope_theta
        self.rope_k_repeat = rope_k_repeat

    def __call__(self, q: array, k: array, v: array, num_k_exclude_rope: int = 0) -> array:
        """Rotate spatial query and key grids while leaving trailing pointer keys unrotated."""
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))
        if not 0 <= num_k_exclude_rope <= k.shape[-2]:
            raise ValueError("Excluded RoPE key count must fit the key sequence")
        side = math.isqrt(q.shape[-2])
        if not side or side * side != q.shape[-2]:
            raise ValueError("RoPE attention requires a nonempty square query grid")
        cos_np, sin_np = axial_rope(
            self.internal_dim // self.num_heads, side, side, self.rope_theta
        )
        cos, sin = mx.array(cos_np), mx.array(sin_np)
        if q.shape[-2] != k.shape[-2] and not self.rope_k_repeat:
            raise ValueError("Query and key lengths differ without rope_k_repeat")
        q = apply_rope(q, cos, sin)
        num_k_rope = k.shape[-2] - num_k_exclude_rope
        if num_k_rope > 0:
            if self.rope_k_repeat and num_k_rope % q.shape[-2]:
                raise ValueError("RoPE spatial keys must contain whole query grids")
            if not self.rope_k_repeat and num_k_rope != q.shape[-2]:
                raise ValueError("RoPE spatial keys must match the query grid")
            k_rope = k[:, :, :num_k_rope]
            if self.rope_k_repeat:
                repeats = num_k_rope // q.shape[-2]
                cos, sin = mx.tile(cos, (repeats, 1)), mx.tile(sin, (repeats, 1))
            k_rope = apply_rope(k_rope, cos, sin)
            k = (
                mx.concatenate([k_rope, k[:, :, num_k_rope:]], axis=2)
                if num_k_exclude_rope
                else k_rope
            )
        return self._attend(q, k, v)


class TwoWayAttentionBlock(nn.Module):
    """SAM two-way block: token self-attention plus cross-attention in both directions."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """Create token self-attention and bidirectional token-image attention."""
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2, activation=relu)
        self.norm3 = LayerNorm(embedding_dim)
        self.norm4 = LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(
        self, queries: array, keys: array, query_pe: array, key_pe: array
    ) -> tuple[array, array]:
        """Update prompt queries and image keys through the two-way block."""
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q, k, keys))
        queries = self.norm3(queries + self.mlp(queries))
        q = queries + query_pe
        k = keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(k, q, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    """SAM two-way transformer between prompt/output tokens and the image embedding."""

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int = 2,
    ) -> None:
        """Build the two-way attention stack and final token-to-image attention."""
        super().__init__()
        self.layers = [
            TwoWayAttentionBlock(
                embedding_dim,
                num_heads,
                mlp_dim,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=i == 0,
            )
            for i in range(depth)
        ]
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = LayerNorm(embedding_dim)

    def __call__(
        self, image_embedding: array, image_pe: array, point_embedding: array
    ) -> tuple[array, array]:
        """Update tokens and image features.

        ``image_embedding`` and ``image_pe`` are (B, C, H, W) (``image_pe`` may have batch
        1 or B); ``point_embedding`` is (B, N, C). Returns the refined tokens (B, N, C) and
        the refined image features as a sequence (B, H * W, C).
        """
        b, c, h, w = image_embedding.shape
        image_embedding = mx.transpose(mx.reshape(image_embedding, (b, c, h * w)), (0, 2, 1))
        image_pe = mx.transpose(mx.reshape(image_pe, (image_pe.shape[0], c, h * w)), (0, 2, 1))
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries, keys, point_embedding, image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(q, k, keys))
        return queries, keys
