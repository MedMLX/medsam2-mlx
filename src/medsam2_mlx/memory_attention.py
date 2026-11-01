# Ported from MedSAM2 (bowang-lab/MedSAM2 @ 332f30d), itself derived from SAM 2
# (Meta Platforms, Inc. and affiliates). Licensed under the Apache License, Version 2.0.
# Source file: sam2/modeling/memory_attention.py.
"""Memory attention: self-attention over the frame, cross-attention to memories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.core import array

from medsam2_mlx._ops import LayerNorm, relu
from medsam2_mlx._runtime import mx, nn
from medsam2_mlx.transformer import RoPEAttention


class MemoryAttentionLayer(nn.Module):
    """Eval-mode layer of ``sam2.1_hiera_t512`` with a ReLU feed-forward.

    Self-attention adds no separate position tensor, although its input already
    contains the stack's scaled current-frame position encoding. Cross-attention
    adds memory positions to keys, not values.
    """

    def __init__(self, d_model: int, mem_dim: int, dim_feedforward: int, rope_theta: float) -> None:
        """Build rotary self-attention, memory cross-attention and feed-forward layers."""
        super().__init__()
        self.self_attn = RoPEAttention(d_model, 1, rope_theta=rope_theta)
        self.cross_attn_image = RoPEAttention(
            d_model, 1, kv_in_dim=mem_dim, rope_theta=rope_theta, rope_k_repeat=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def __call__(
        self,
        tgt: array,
        memory: array,
        pos: array,
        num_k_exclude_rope: int = 0,
    ) -> array:
        """Condition normalized frame tokens on memory and apply residual feed-forward."""
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.self_attn(tgt2, tgt2, tgt2)
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.cross_attn_image(
            tgt2, memory + pos, memory, num_k_exclude_rope=num_k_exclude_rope
        )
        tgt2 = self.norm3(tgt)
        return tgt + self.linear2(relu(self.linear1(tgt2)))


class MemoryAttention(nn.Module):
    """Memory attention stack: per layer RoPE self-attention, cross-attention, feed-forward."""

    def __init__(
        self,
        d_model: int,
        mem_dim: int,
        num_layers: int,
        dim_feedforward: int,
        rope_theta: float,
    ) -> None:
        """Create the configured memory-attention stack and output normalization."""
        super().__init__()
        self.d_model = d_model
        self.layers = [
            MemoryAttentionLayer(d_model, mem_dim, dim_feedforward, rope_theta)
            for _ in range(num_layers)
        ]
        self.norm = LayerNorm(d_model)

    def __call__(
        self,
        curr: array,
        memory: array,
        curr_pos: array,
        memory_pos: array,
        num_obj_ptr_tokens: int = 0,
    ) -> array:
        """Condition the current frame on memories; sequence-first inputs (L, B, C), as upstream.

        Args:
            curr: Current frame tokens, (L, B, C); L = 32 * 32 in the default graph.
            memory: Memory tokens (M, B, mem_dim): spatial memories, then object pointers.
            curr_pos: Position encoding of ``curr``, (L, B, C).
            memory_pos: Position encoding of ``memory``, (M, B, mem_dim).
            num_obj_ptr_tokens: Number of trailing object-pointer tokens in ``memory``;
                they get no rotary encoding.

        Returns:
            Conditioned features (L, B, C).

        Raises:
            ValueError: The batch sizes differ, or rotary attention rejects a non-square
                query grid or incompatible query and key lengths.
        """
        if curr.shape[1] != memory.shape[1]:
            raise ValueError("Batch size must be the same for curr and memory")
        # Upstream adds the current frame's position encoding to the input at weight 0.1.
        output = curr + 0.1 * curr_pos
        output = mx.transpose(output, (1, 0, 2))
        memory = mx.transpose(memory, (1, 0, 2))
        memory_pos = mx.transpose(memory_pos, (1, 0, 2))
        for layer in self.layers:
            output = layer(output, memory, memory_pos, num_k_exclude_rope=num_obj_ptr_tokens)
        return mx.transpose(self.norm(output), (1, 0, 2))
